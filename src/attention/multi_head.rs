//! Bias-free, rotary, causal multi-head self-attention.

use std::error::Error;
use std::fmt;

use super::causal_mask::{
    CausalMaskingError, causal_scaled_dot_product_self_attention_with_context,
};
use super::qkv::{QkvError, QkvProjections};
use super::rope::{RopeError, RotaryEmbedding};
use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::nn::linear::{Linear, LinearError};
use crate::tensor::storage::Tensor;

/// A taped reshape/transpose stage in the public split and merge helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadLayoutStage {
    SplitReshape,
    SplitTranspose,
    MergeTranspose,
    MergeReshape,
}

impl fmt::Display for HeadLayoutStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SplitReshape => "split reshape",
            Self::SplitTranspose => "split transpose",
            Self::MergeTranspose => "merge transpose",
            Self::MergeReshape => "merge reshape",
        })
    }
}

/// A rejected rank or feature partition in `split_heads` or `merge_heads`.
#[derive(Clone, Debug, PartialEq)]
pub enum HeadLayoutError {
    SplitRank {
        rank: usize,
    },
    MergeRank {
        rank: usize,
    },
    ZeroHeadCount,
    ZeroHeadWidth,
    WidthNotDivisible {
        width: usize,
        heads: usize,
    },
    ModelWidthOverflow {
        heads: usize,
        head_width: usize,
    },
    Autodiff {
        stage: HeadLayoutStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for HeadLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplitRank { rank } => write!(
                formatter,
                "head split input must have rank three [batch, tokens, model_width], got rank {rank}"
            ),
            Self::MergeRank { rank } => write!(
                formatter,
                "head merge input must have rank four [batch, heads, tokens, head_width], got rank {rank}"
            ),
            Self::ZeroHeadCount => formatter.write_str("head count must be nonzero"),
            Self::ZeroHeadWidth => formatter.write_str("head width must be nonzero"),
            Self::WidthNotDivisible { width, heads } => write!(
                formatter,
                "model width {width} must be divisible by head count {heads}"
            ),
            Self::ModelWidthOverflow { heads, head_width } => write!(
                formatter,
                "merged model width overflows for {heads} heads of width {head_width}"
            ),
            Self::Autodiff { stage, source } => {
                write!(formatter, "multi-head {stage}: {source}")
            }
        }
    }
}

impl Error for HeadLayoutError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn layout_autodiff(stage: HeadLayoutStage) -> impl FnOnce(TensorAutodiffError) -> HeadLayoutError {
    move |source| HeadLayoutError::Autodiff { stage, source }
}

/// Converts `[batch, tokens, model_width]` to `[batch, heads, tokens, head_width]`
pub fn split_heads(input: &TensorValue, heads: usize) -> Result<TensorValue, HeadLayoutError> {
    split_heads_with_context(AutogradContext::recording(), input, heads)
}

/// Splits model-width rows under the caller's explicit recording policy
pub fn split_heads_with_context(
    context: AutogradContext,
    input: &TensorValue,
    heads: usize,
) -> Result<TensorValue, HeadLayoutError> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(HeadLayoutError::SplitRank { rank: shape.len() });
    }
    if heads == 0 {
        return Err(HeadLayoutError::ZeroHeadCount);
    }

    let width = shape[2];
    if width == 0 {
        return Err(HeadLayoutError::ZeroHeadWidth);
    }
    if !width.is_multiple_of(heads) {
        return Err(HeadLayoutError::WidthNotDivisible { width, heads });
    }
    let head_width = width / heads;
    let reshaped = input
        .reshape_with_context(context, &[shape[0], shape[1], heads, head_width])
        .map_err(layout_autodiff(HeadLayoutStage::SplitReshape))?;
    reshaped
        .transpose_with_context(context, 1, 2)
        .map_err(layout_autodiff(HeadLayoutStage::SplitTranspose))
}

/// Converts `[batch, heads, tokens, head_width]` back to model-width rows
pub fn merge_heads(input: &TensorValue) -> Result<TensorValue, HeadLayoutError> {
    merge_heads_with_context(AutogradContext::recording(), input)
}

/// Merges attention heads under the caller's explicit recording policy.
pub fn merge_heads_with_context(
    context: AutogradContext,
    input: &TensorValue,
) -> Result<TensorValue, HeadLayoutError> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(HeadLayoutError::MergeRank { rank: shape.len() });
    }
    if shape[1] == 0 {
        return Err(HeadLayoutError::ZeroHeadCount);
    }
    if shape[3] == 0 {
        return Err(HeadLayoutError::ZeroHeadWidth);
    }
    let model_width =
        shape[1]
            .checked_mul(shape[3])
            .ok_or(HeadLayoutError::ModelWidthOverflow {
                heads: shape[1],
                head_width: shape[3],
            })?;
    let transposed = input
        .transpose_with_context(context, 1, 2)
        .map_err(layout_autodiff(HeadLayoutStage::MergeTranspose))?;
    transposed
        .reshape_with_context(context, &[shape[0], shape[2], model_width])
        .map_err(layout_autodiff(HeadLayoutStage::MergeReshape))
}

/// A Q/K/V branch whose head-layou or rotary operation failed
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MultiHeadInput {
    Query,
    Key,
    Value,
}

impl fmt::Display for MultiHeadInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
        })
    }
}

/// A cumulative tensor stage owned by the multi-head assembly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiHeadStage {
    QueryLanes,
    KeyLanes,
    ValueLanes,
    RestoreWeights,
    RestoreHeadOutputs,
}

impl fmt::Display for MultiHeadStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::QueryLanes => "query lane flatten",
            Self::KeyLanes => "key lane flatten",
            Self::ValueLanes => "value lane flatten",
            Self::RestoreWeights => "attention-weight head restore",
            Self::RestoreHeadOutputs => "head-output restore",
        })
    }
}

/// A rejected configuration, parameter set, input, or cumulative forward stage.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiHeadAttentionError {
    ZeroModelWidth,
    ZeroHeadCount,
    ModelWidthNotDivisible {
        model_width: usize,
        heads: usize,
    },
    OddHeadWidth {
        head_width: usize,
    },
    QkvProjection(QkvError),
    QkvOutputWidthMismatch {
        model_width: usize,
        projected_width: usize,
    },
    OutputProjection(LinearError),
    OutputInputWidthMismatch {
        expected: usize,
        actual: usize,
    },
    OutputWidthMismatch {
        expected: usize,
        actual: usize,
    },
    Initialization(InitializationError),
    InputRank {
        rank: usize,
    },
    InputWidthMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyTokens,
    PositionOffsetOverflow {
        offset: usize,
        tokens: usize,
    },
    PositionRangeExceeded {
        offset: usize,
        tokens: usize,
        max_positions: usize,
    },
    BatchHeadOverflow {
        batch: usize,
        heads: usize,
    },
    HeadLayout {
        input: MultiHeadInput,
        source: HeadLayoutError,
    },
    MergeLayout(HeadLayoutError),
    RotaryConfiguration(RopeError),
    Rotary {
        input: MultiHeadInput,
        source: RopeError,
    },
    CausalAttention(CausalMaskingError),
    Autodiff {
        stage: MultiHeadStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for MultiHeadAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroModelWidth => formatter.write_str("multi-head model width must be nonzero"),
            Self::ZeroHeadCount => formatter.write_str("multi-head head count must be nonzero"),
            Self::ModelWidthNotDivisible { model_width, heads } => write!(
                formatter,
                "multi-head model width {model_width} must be divisible by head count {heads}"
            ),
            Self::OddHeadWidth { head_width } => write!(
                formatter,
                "multi-head per-head width must be even for RoPE, got {head_width}"
            ),
            Self::QkvProjection(source) => source.fmt(formatter),
            Self::QkvOutputWidthMismatch {
                model_width,
                projected_width,
            } => write!(
                formatter,
                "Q/K/V projected width must equal model width {model_width}, got {projected_width}"
            ),
            Self::OutputProjection(source) => write!(formatter, "output projection: {source}"),
            Self::OutputInputWidthMismatch { expected, actual } => write!(
                formatter,
                "output projection input width must equal model width {expected}, got {actual}"
            ),
            Self::OutputWidthMismatch { expected, actual } => write!(
                formatter,
                "output projection width must equal model width {expected}, got {actual}"
            ),
            Self::Initialization(source) => source.fmt(formatter),
            Self::InputRank { rank } => write!(
                formatter,
                "multi-head input must have rank three [batch, tokens, model_width], got rank {rank}"
            ),
            Self::InputWidthMismatch { expected, actual } => write!(
                formatter,
                "multi-head input final width must equal model width {expected}, got {actual}"
            ),
            Self::EmptyTokens => formatter.write_str(
                "multi-head attention needs at least one token so every causal row has a key",
            ),
            Self::PositionOffsetOverflow { offset, tokens } => write!(
                formatter,
                "multi-head position interval overflows: offset {offset} plus {tokens} tokens"
            ),
            Self::PositionRangeExceeded {
                offset,
                tokens,
                max_positions,
            } => write!(
                formatter,
                "multi-head position interval [{offset}, {}) exceeds capacity {max_positions}",
                offset.saturating_add(*tokens)
            ),
            Self::BatchHeadOverflow { batch, heads } => write!(
                formatter,
                "multi-head lane count overflows for batch {batch} and {heads} heads"
            ),
            Self::HeadLayout { input, source } => {
                write!(formatter, "{input} head layout: {source}")
            }
            Self::MergeLayout(source) => write!(formatter, "head output merge: {source}"),
            Self::RotaryConfiguration(source) => {
                write!(formatter, "multi-head RoPE configuration: {source}")
            }
            Self::Rotary { input, source } => write!(formatter, "{input} RoPE: {source}"),
            Self::CausalAttention(source) => source.fmt(formatter),
            Self::Autodiff { stage, source } => {
                write!(formatter, "multi-head {stage}: {source}")
            }
        }
    }
}

impl Error for MultiHeadAttentionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::QkvProjection(source) => Some(source),
            Self::OutputProjection(source) => Some(source),
            Self::Initialization(source) => Some(source),
            Self::HeadLayout { source, .. } => Some(source),
            Self::MergeLayout(source) => Some(source),
            Self::RotaryConfiguration(source) => Some(source),
            Self::Rotary { source, .. } => Some(source),
            Self::CausalAttention(source) => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<InitializationError> for MultiHeadAttentionError {
    fn from(source: InitializationError) -> Self {
        Self::Initialization(source)
    }
}

fn multi_autodiff(
    stage: MultiHeadStage,
) -> impl FnOnce(TensorAutodiffError) -> MultiHeadAttentionError {
    move |source| MultiHeadAttentionError::Autodiff { stage, source }
}

fn head_layout(input: MultiHeadInput) -> impl FnOnce(HeadLayoutError) -> MultiHeadAttentionError {
    move |source| MultiHeadAttentionError::HeadLayout { input, source }
}

fn rotary_error(input: MultiHeadInput) -> impl FnOnce(RopeError) -> MultiHeadAttentionError {
    move |source| MultiHeadAttentionError::Rotary { input, source }
}

fn validate_configuration(
    model_width: usize,
    heads: usize,
) -> Result<usize, MultiHeadAttentionError> {
    if model_width == 0 {
        return Err(MultiHeadAttentionError::ZeroModelWidth);
    }
    if heads == 0 {
        return Err(MultiHeadAttentionError::ZeroHeadCount);
    }
    if !model_width.is_multiple_of(heads) {
        return Err(MultiHeadAttentionError::ModelWidthNotDivisible { model_width, heads });
    }
    let head_width = model_width / heads;
    if !head_width.is_multiple_of(2) {
        return Err(MultiHeadAttentionError::OddHeadWidth { head_width });
    }
    Ok(head_width)
}

/// Inspectable intermediate tensors from one complete multi-head forward pass
#[derive(Clone, Debug)]
pub struct MultiHeadAttentionForward {
    projected_query_heads: TensorValue,
    projected_key_heads: TensorValue,
    projected_value_heads: TensorValue,
    rotated_query_heads: TensorValue,
    rotated_key_heads: TensorValue,
    attention_weights: TensorValue,
    head_outputs: TensorValue,
    merged: TensorValue,
    output: TensorValue,
}

impl MultiHeadAttentionForward {
    pub fn projected_query_heads(&self) -> &TensorValue {
        &self.projected_query_heads
    }

    pub fn projected_key_heads(&self) -> &TensorValue {
        &self.projected_key_heads
    }

    pub fn projected_value_heads(&self) -> &TensorValue {
        &self.projected_value_heads
    }

    pub fn rotated_query_heads(&self) -> &TensorValue {
        &self.rotated_query_heads
    }

    pub fn rotated_key_heads(&self) -> &TensorValue {
        &self.rotated_key_heads
    }

    pub fn attention_weights(&self) -> &TensorValue {
        &self.attention_weights
    }

    pub fn head_outputs(&self) -> &TensorValue {
        &self.head_outputs
    }

    pub fn merged(&self) -> &TensorValue {
        &self.merged
    }

    pub fn output(&self) -> &TensorValue {
        &self.output
    }

    pub fn into_output(self) -> TensorValue {
        self.output
    }
}

/// Full-width Q/K/V projections, per-head RoPE and causal attention, and W_O
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    qkv: QkvProjections,
    output: Linear,
    rope: RotaryEmbedding,
    parameters: NamedParameters,
    model_width: usize,
    heads: usize,
    head_width: usize,
}

impl MultiHeadAttention {
    /// Initializes Q, K, V, and 0 in order without partially advancing `rng`
    pub fn new(
        parameter_prefix: impl Into<String>,
        model_width: usize,
        heads: usize,
        max_positions: usize,
        rope_base: f64,
        rng: &mut SplitMix64,
    ) -> Result<Self, MultiHeadAttentionError> {
        let head_width = validate_configuration(model_width, heads)?;
        let rope = RotaryEmbedding::new(head_width, max_positions, rope_base)
            .map_err(MultiHeadAttentionError::RotaryConfiguration)?;
        let parameter_prefix = parameter_prefix.into();
        let mut trial = rng.clone();
        let qkv = QkvProjections::new(&parameter_prefix, model_width, model_width, &mut trial)
            .map_err(MultiHeadAttentionError::QkvProjection)?;
        let output = Linear::new(
            format!("{parameter_prefix}.output"),
            model_width,
            model_width,
            false,
            &mut trial,
        )
        .map_err(MultiHeadAttentionError::OutputProjection)?;
        let layer = Self::from_validated_parts(qkv, output, rope, heads, head_width)?;
        *rng = trial;
        Ok(layer)
    }

    pub fn from_parameters(
        query_weight: NamedParameter,
        key_weight: NamedParameter,
        value_weight: NamedParameter,
        output_weight: NamedParameter,
        heads: usize,
        max_positions: usize,
        rope_base: f64,
    ) -> Result<Self, MultiHeadAttentionError> {
        let qkv = QkvProjections::from_weights(query_weight, key_weight, value_weight)
            .map_err(MultiHeadAttentionError::QkvProjection)?;
        let output = Linear::from_parameters(output_weight, None)
            .map_err(MultiHeadAttentionError::OutputProjection)?;
        let model_width = qkv.model_width();
        let head_width = validate_configuration(model_width, heads)?;
        if qkv.head_width() != model_width {
            return Err(MultiHeadAttentionError::QkvOutputWidthMismatch {
                model_width,
                projected_width: qkv.head_width(),
            });
        }
        if output.input_width() != model_width {
            return Err(MultiHeadAttentionError::OutputInputWidthMismatch {
                expected: model_width,
                actual: output.input_width(),
            });
        }
        if output.output_width() != model_width {
            return Err(MultiHeadAttentionError::OutputWidthMismatch {
                expected: model_width,
                actual: output.output_width(),
            });
        }
        let rope = RotaryEmbedding::new(head_width, max_positions, rope_base)
            .map_err(MultiHeadAttentionError::RotaryConfiguration)?;
        Self::from_validated_parts(qkv, output, rope, heads, head_width)
    }

    fn from_validated_parts(
        qkv: QkvProjections,
        output: Linear,
        rope: RotaryEmbedding,
        heads: usize,
        head_width: usize,
    ) -> Result<Self, MultiHeadAttentionError> {
        let model_width = qkv.model_width();
        let mut listed = Vec::with_capacity(4);
        listed.extend(qkv.parameters().iter().cloned());
        listed.push(output.weight().clone());
        let parameters = NamedParameters::try_new(listed)?;
        Ok(Self {
            qkv,
            output,
            rope,
            parameters,
            model_width,
            heads,
            head_width,
        })
    }

    /// Runs position-aware causal attention independently in every head.
    pub fn forward(
        &self,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<MultiHeadAttentionForward, MultiHeadAttentionError> {
        self.forward_with_context(AutogradContext::recording(), input, position_offset)
    }

    /// Computes multi-head causal self-attention with an explicit autograd context.
    ///
    /// A single attention head produces one attention distribution per query,
    /// shared by all Value features in that head. Multi-head attention allows
    /// several independent distributions to be learned and applied simultaneously,
    /// so each token can gather different kinds of information from its context.
    ///
    /// Each head independently computes:
    ///
    /// ```text
    /// Qₕ = XW_Qₕ
    /// Kₕ = XW_Kₕ
    /// Vₕ = XW_Vₕ
    ///
    /// Q̃ₕ = RoPE(Qₕ)
    /// K̃ₕ = RoPE(Kₕ)
    ///
    /// Aₕ = causal_softmax((Q̃ₕ @ K̃ₕᵀ) / √dₕ)
    /// Hₕ = Aₕ @ Vₕ
    /// ```
    ///
    /// The final output combines all head representations:
    ///
    /// ```text
    /// output = Concat(H₀, H₁, ..., Hₕ₋₁) @ W_O
    /// ```
    ///
    /// Here, `h` identifies a head, `dₕ` is its feature width, and `W_O` is
    /// the learned output projection.
    ///
    /// # Computation stages
    ///
    /// 1. **Q/K/V projections**
    ///
    ///    Apply three independent, full-width Linear transformations to `input`.
    ///    Each projected feature may depend on every original input feature.
    ///
    /// 2. **Split into heads**
    ///
    ///    Reshape and transpose the projected tensors:
    ///
    ///    ```text
    ///    [B, T, D] → [B, T, H, Dh] → [B, H, T, Dh]
    ///    ```
    ///
    ///    This partitions the projected feature dimension into `H` heads,
    ///    each containing `Dh = D / H` features.
    ///
    ///    The split happens after projection, not before it. Therefore, each
    ///    head can learn combinations of all input features rather than being
    ///    restricted to a fixed subset of the original input.
    ///
    /// 3. **Rotary position embeddings (RoPE)**
    ///
    ///    Rotate adjacent feature pairs in Q and K according to their absolute
    ///    token positions, starting at `position_offset`.
    ///
    ///    The resulting query-key dot products can depend on relative token
    ///    positions because R(m)ᵀR(n) represents a relative rotation.
    ///
    ///    V is not rotated: positional information affects how token positions
    ///    are matched, while V supplies the content being aggregated.
    ///
    /// 4. **Independent causal attention**
    ///
    ///    Flatten batch and head axes:
    ///
    ///    ```text
    ///    [B, H, T, Dh] → [B × H, T, Dh]
    ///    ```
    ///
    ///    This allows the existing rank-3 causal attention operation to process
    ///    every `(batch, head)` pair independently.
    ///
    ///    Each head computes its own attention weights over key positions.
    ///    For query position `i`, only key positions `j ≤ i` are permitted.
    ///    Future positions receive exactly zero attention weight.
    ///
    ///    Softmax normalizes each allowed query row independently:
    ///
    ///    ```text
    ///    Σⱼ Aₕ[i,j] = 1,  for allowed j ≤ i
    ///    ```
    ///
    ///    Attention distributions are not normalized across different heads.
    ///
    /// 5. **Restore head dimensions**
    ///
    ///    Restore the separate batch and head axes:
    ///
    ///    ```text
    ///    attention weights: [B, H, T, T]
    ///    head outputs:      [B, H, T, Dh]
    ///    ```
    ///
    ///    Each head output is a weighted sum of its unrotated Value vectors.
    ///
    /// 6. **Merge head outputs**
    ///
    ///    Transpose and reshape the head outputs:
    ///
    ///    ```text
    ///    [B, H, T, Dh]
    ///          ↓ transpose
    ///    [B, T, H, Dh]
    ///          ↓ reshape
    ///    [B, T, D]
    ///    ```
    ///
    ///    The transpose is essential: it places all head representations
    ///    belonging to the same token next to each other.
    ///
    ///    Concatenation itself performs no learned mixing or averaging.
    ///
    /// 7. **Output projection**
    ///
    ///    Apply the learned matrix `W_O` of shape `[D, D]` to the merged
    ///    representations.
    ///
    ///    Unlike the independent attention computations, this projection can
    ///    combine features produced by different heads into the same output
    ///    coordinate.
    ///
    /// # Tensor shapes
    ///
    /// For `B = 1`, `T = 3`, `D = 4`, and `H = 2`:
    ///
    /// ```text
    /// Stage                  Shape
    /// ---------------------------------------
    /// Input                  [1, 3, 4]
    /// Projected Q/K/V        [1, 3, 4]
    /// Split Q/K/V            [1, 2, 3, 2]
    /// RoPE Q/K               [1, 2, 3, 2]
    /// Flattened Q/K/V        [2, 3, 2]
    /// Attention weights      [1, 2, 3, 3]
    /// Head outputs           [1, 2, 3, 2]
    /// Merged heads           [1, 3, 4]
    /// Output projection      [1, 3, 4]
    /// ```
    ///
    /// The complete layer preserves the input shape `[B, T, D]`, making its
    /// output suitable for subsequent residual addition.
    ///
    /// This method computes only the attention transformation. RMSNorm and
    /// residual addition are separate operations handled by the surrounding
    /// Transformer block.
    ///
    /// # Position offset
    ///
    /// `position_offset` specifies the absolute position of the first token
    /// in the supplied input.
    ///
    /// For example:
    ///
    /// ```text
    /// position_offset = 0, T = 3 → positions [0, 1, 2]
    /// position_offset = 5, T = 3 → positions [5, 6, 7]
    /// ```
    ///
    /// The complete position range must fit within the precomputed RoPE tables.
    /// Using an offset preserves the correct positional rotations when processing
    /// a sequence segment that does not start at position zero.
    ///
    /// # Autograd
    ///
    /// `context` controls whether the operations are recorded for backward.
    ///
    /// When recording is enabled, gradients propagate through the output
    /// projection, head merging, causal attention, RoPE, and Q/K/V projections.
    ///
    /// All four learned matrices receive gradients from the downstream loss.
    /// Contributions from all heads accumulate into the gradient of the shared
    /// input tensor.
    ///
    /// # Returns
    ///
    /// Returns [`MultiHeadAttentionForward`], exposing:
    ///
    /// - Projected Q/K/V tensors after head splitting.
    /// - Rotated Q/K tensors.
    /// - Per-head causal attention weights.
    /// - Per-head weighted Value representations.
    /// - Merged head representations before the output projection.
    /// - Final projected output.
    ///
    /// These intermediates are useful for inspecting attention distributions,
    /// verifying tensor layouts, and debugging gradient propagation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - `input` is not rank 3 `[B, T, D]`.
    /// - The input feature width differs from the configured model width.
    /// - The token axis is empty (`T = 0`).
    /// - `position_offset + T` overflows or exceeds the RoPE capacity.
    /// - The combined batch/head dimension overflows.
    /// - A Q/K/V projection, head-layout transformation, RoPE rotation,
    ///   causal attention computation, or output projection fails.
    ///
    /// An empty batch (`B = 0`) is allowed, provided the remaining shape
    /// and position constraints are satisfied.
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<MultiHeadAttentionForward, MultiHeadAttentionError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(MultiHeadAttentionError::InputRank { rank: shape.len() });
        }
        if shape[2] != self.model_width {
            return Err(MultiHeadAttentionError::InputWidthMismatch {
                expected: self.model_width,
                actual: shape[2],
            });
        }
        if shape[1] == 0 {
            return Err(MultiHeadAttentionError::EmptyTokens);
        }
        let position_end = position_offset.checked_add(shape[1]).ok_or(
            MultiHeadAttentionError::PositionOffsetOverflow {
                offset: position_offset,
                tokens: shape[1],
            },
        )?;
        if position_end > self.rope.max_positions() {
            return Err(MultiHeadAttentionError::PositionRangeExceeded {
                offset: position_offset,
                tokens: shape[1],
                max_positions: self.rope.max_positions(),
            });
        }

        let projected = self
            .qkv
            .forward_with_context(context, input)
            .map_err(MultiHeadAttentionError::QkvProjection)?;
        let projected_query_heads =
            split_heads_with_context(context, projected.query(), self.heads)
                .map_err(head_layout(MultiHeadInput::Query))?;
        let projected_key_heads = split_heads_with_context(context, projected.key(), self.heads)
            .map_err(head_layout(MultiHeadInput::Key))?;
        let projected_value_heads =
            split_heads_with_context(context, projected.value(), self.heads)
                .map_err(head_layout(MultiHeadInput::Value))?;
        let rotated_query_heads = self
            .rope
            .rotate_with_context(context, &projected_query_heads, position_offset)
            .map_err(rotary_error(MultiHeadInput::Query))?;
        let rotated_key_heads = self
            .rope
            .rotate_with_context(context, &projected_key_heads, position_offset)
            .map_err(rotary_error(MultiHeadInput::Key))?;

        let lanes =
            shape[0]
                .checked_mul(self.heads)
                .ok_or(MultiHeadAttentionError::BatchHeadOverflow {
                    batch: shape[0],
                    heads: self.heads,
                })?;
        let lane_shape = [lanes, shape[1], self.head_width];
        let query_lanes = rotated_query_heads
            .reshape_with_context(context, &lane_shape)
            .map_err(multi_autodiff(MultiHeadStage::QueryLanes))?;
        let key_lanes = rotated_key_heads
            .reshape_with_context(context, &lane_shape)
            .map_err(multi_autodiff(MultiHeadStage::KeyLanes))?;
        let value_lanes = projected_value_heads
            .reshape_with_context(context, &lane_shape)
            .map_err(multi_autodiff(MultiHeadStage::ValueLanes))?;
        let attended = causal_scaled_dot_product_self_attention_with_context(
            context,
            &query_lanes,
            &key_lanes,
            &value_lanes,
        )
        .map_err(MultiHeadAttentionError::CausalAttention)?;
        let attention_weights = attended
            .weights()
            .reshape_with_context(context, &[shape[0], self.heads, shape[1], shape[1]])
            .map_err(multi_autodiff(MultiHeadStage::RestoreWeights))?;
        let head_outputs = attended
            .output()
            .reshape_with_context(context, &[shape[0], self.heads, shape[1], self.head_width])
            .map_err(multi_autodiff(MultiHeadStage::RestoreHeadOutputs))?;
        let merged = merge_heads_with_context(context, &head_outputs)
            .map_err(MultiHeadAttentionError::MergeLayout)?;
        let output = self
            .output
            .forward_with_context(context, &merged)
            .map_err(MultiHeadAttentionError::OutputProjection)?;

        Ok(MultiHeadAttentionForward {
            projected_query_heads,
            projected_key_heads,
            projected_value_heads,
            rotated_query_heads,
            rotated_key_heads,
            attention_weights,
            head_outputs,
            merged,
            output,
        })
    }

    pub fn qkv(&self) -> &QkvProjections {
        &self.qkv
    }

    pub fn output_projection(&self) -> &Linear {
        &self.output
    }

    pub fn rope(&self) -> &RotaryEmbedding {
        &self.rope
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn model_width(&self) -> usize {
        self.model_width
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_width(&self) -> usize {
        self.head_width
    }

    pub fn parameter_count(&self) -> usize {
        4 * self.model_width * self.model_width
    }
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::autograd::tensor_core::GraphRetention;
    use crate::tensor::storage::Tensor;

    const MODEL_WIDTH: usize = 4;
    const HEADS: usize = 2;
    const TOKENS: usize = 3;
    const BASE: f64 = 100.0;

    fn tensor(shape: &[usize], values: &[f64]) -> Tensor {
        Tensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
    }

    fn constant(shape: &[usize], values: &[f64]) -> TensorValue {
        TensorValue::constant(tensor(shape, values)).unwrap()
    }

    fn parameter(name: &str, values: &[f64]) -> NamedParameter {
        NamedParameter::from_tensor(name, tensor(&[MODEL_WIDTH, MODEL_WIDTH], values)).unwrap()
    }

    fn identity() -> [f64; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn block_swap() -> [f64; 16] {
        [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ]
    }

    fn fixture_values() -> [f64; 12] {
        [
            1.0,
            0.0,
            1.0,
            0.0,
            1.0_f64.cos(),
            -1.0_f64.sin(),
            0.0,
            1.0,
            2.0_f64.cos(),
            -2.0_f64.sin(),
            1.0,
            1.0,
        ]
    }

    fn fixture_layer() -> MultiHeadAttention {
        let identity = identity();
        MultiHeadAttention::from_parameters(
            parameter("attention.query.weight", &identity),
            parameter("attention.key.weight", &identity),
            parameter("attention.value.weight", &identity),
            parameter("attention.output.weight", &block_swap()),
            HEADS,
            TOKENS + 3,
            BASE,
        )
        .unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected:.12}, got {actual:.12}"
            );
        }
    }

    fn sum_to_scalar(mut value: TensorValue) -> TensorValue {
        for axis in (0..value.shape().len()).rev() {
            value = value.sum_axis(axis, false).unwrap();
        }
        value
    }

    #[test]
    fn split_and_merge_are_exact_taped_inverses() {
        let values = (0..12).map(f64::from).collect::<Vec<_>>();
        let input = TensorValue::parameter(tensor(&[1, 2, 6], &values)).unwrap();
        let split = split_heads(&input, 3).unwrap();
        assert_eq!(split.shape(), [1, 3, 2, 2]);
        assert_eq!(
            split.value().as_slice(),
            &[0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]
        );
        let merged = merge_heads(&split).unwrap();
        assert_eq!(merged.shape(), [1, 2, 6]);
        assert_eq!(merged.value().as_slice(), values);

        let seed = tensor(
            &[1, 2, 6],
            &[
                0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 3.0, -2.0, 0.1, 0.2, 0.3, 0.4,
            ],
        );
        merged
            .backward_with_seed(&seed.view(), GraphRetention::Retain)
            .unwrap();
        assert_eq!(&*input.gradient().unwrap(), &seed);
    }

    #[test]
    fn frozen_heads_have_uniform_and_nonuniform_causal_tables() {
        let layer = fixture_layer();
        let pass = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &fixture_values()), 0)
            .unwrap();
        assert_eq!(pass.projected_query_heads().shape(), [1, 2, 3, 2]);
        assert_eq!(pass.attention_weights().shape(), [1, 2, 3, 3]);
        assert_eq!(pass.head_outputs().shape(), [1, 2, 3, 2]);
        assert_eq!(pass.merged().shape(), [1, 3, 4]);
        assert_eq!(pass.output().shape(), [1, 3, 4]);
        let weights = pass.attention_weights().value();
        assert_close(
            &weights.as_slice()[..9],
            &[
                1.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ],
            1e-12,
        );
        assert_ne!(&weights.as_slice()[..9], &weights.as_slice()[9..]);
        for head in 0..HEADS {
            for query in 0..TOKENS {
                let row = &weights.as_slice()
                    [(head * TOKENS + query) * TOKENS..(head * TOKENS + query + 1) * TOKENS];
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(row[query + 1..].iter().all(|value| *value == 0.0));
            }
        }
        let merged_values = pass.merged().value();
        let output_values = pass.output().value();
        for token in 0..TOKENS {
            let merged = &merged_values.as_slice()[token * MODEL_WIDTH..(token + 1) * MODEL_WIDTH];
            let output = &output_values.as_slice()[token * MODEL_WIDTH..(token + 1) * MODEL_WIDTH];
            assert_eq!(output, &[merged[2], merged[3], merged[0], merged[1]]);
        }
    }

    #[test]
    fn batch_and_head_lanes_match_independent_sequence_forwards() {
        let layer = fixture_layer();
        let first = fixture_values();
        let second = [
            -0.25, 0.75, 1.5, -1.0, 0.4, 0.2, -0.8, 1.25, 1.1, -0.6, 0.3, 0.9,
        ];
        let mut batched_values = first.to_vec();
        batched_values.extend(second);

        let batched = layer
            .forward(&constant(&[2, TOKENS, MODEL_WIDTH], &batched_values), 0)
            .unwrap();
        let first_pass = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &first), 0)
            .unwrap();
        let second_pass = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &second), 0)
            .unwrap();

        assert_eq!(batched.attention_weights().shape(), [2, 2, 3, 3]);
        assert_eq!(batched.head_outputs().shape(), [2, 2, 3, 2]);
        assert_eq!(batched.output().shape(), [2, 3, 4]);
        for (batched_tensor, first_tensor, second_tensor) in [
            (
                batched.attention_weights(),
                first_pass.attention_weights(),
                second_pass.attention_weights(),
            ),
            (
                batched.head_outputs(),
                first_pass.head_outputs(),
                second_pass.head_outputs(),
            ),
            (batched.output(), first_pass.output(), second_pass.output()),
        ] {
            let batch_stride = first_tensor.value().len();
            assert_eq!(
                &batched_tensor.value().as_slice()[..batch_stride],
                first_tensor.value().as_slice()
            );
            assert_eq!(
                &batched_tensor.value().as_slice()[batch_stride..],
                second_tensor.value().as_slice()
            );
        }
    }

    #[test]
    fn fixture_preserves_head_isolation_and_prefix_invariance() {
        let layer = fixture_layer();
        let input = fixture_values();
        let baseline = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &input), 0)
            .unwrap();

        let mut changed_block = input;
        changed_block[0] += 0.25;
        let changed = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &changed_block), 0)
            .unwrap();
        let baseline_heads = baseline.head_outputs().value();
        let changed_heads = changed.head_outputs().value();
        assert_ne!(
            &baseline_heads.as_slice()[..6],
            &changed_heads.as_slice()[..6]
        );
        assert_eq!(
            &baseline_heads.as_slice()[6..],
            &changed_heads.as_slice()[6..]
        );

        let mut changed_suffix = input;
        changed_suffix[8..12].copy_from_slice(&[3.0, -2.0, -1.0, 4.0]);
        let suffix = layer
            .forward(&constant(&[1, TOKENS, MODEL_WIDTH], &changed_suffix), 0)
            .unwrap();
        assert_eq!(
            &baseline.output().value().as_slice()[..8],
            &suffix.output().value().as_slice()[..8]
        );
        assert_ne!(
            &baseline.output().value().as_slice()[8..],
            &suffix.output().value().as_slice()[8..]
        );
    }

    #[test]
    fn parameters_are_bias_free_full_width_stable_and_distinct() {
        let layer = fixture_layer();
        assert_eq!(layer.model_width(), 4);
        assert_eq!(layer.heads(), 2);
        assert_eq!(layer.head_width(), 2);
        assert_eq!(layer.parameter_count(), 64);
        assert_eq!(
            layer
                .parameters()
                .iter()
                .map(NamedParameter::name)
                .collect::<Vec<_>>(),
            [
                "attention.query.weight",
                "attention.key.weight",
                "attention.value.weight",
                "attention.output.weight"
            ]
        );
        for parameter in layer.parameters() {
            assert_eq!(parameter.tensor().shape(), [4, 4]);
        }
        assert!(!layer.qkv().query().has_bias());
        assert!(!layer.qkv().key().has_bias());
        assert!(!layer.qkv().value().has_bias());
        assert!(!layer.output_projection().has_bias());
        for first in 0..4 {
            for second in first + 1..4 {
                assert!(
                    !layer.parameters()[first]
                        .tensor()
                        .is_same_node(layer.parameters()[second].tensor())
                );
            }
        }
    }

    #[test]
    fn configuration_and_forward_errors_have_stable_precedence() {
        let mut rng = SplitMix64::from_seed(9);
        assert_eq!(
            MultiHeadAttention::new("bad", 0, 0, 0, f64::NAN, &mut rng).unwrap_err(),
            MultiHeadAttentionError::ZeroModelWidth
        );
        assert_eq!(
            MultiHeadAttention::new("bad", 4, 0, 0, f64::NAN, &mut rng).unwrap_err(),
            MultiHeadAttentionError::ZeroHeadCount
        );
        assert_eq!(
            MultiHeadAttention::new("bad", 6, 4, 0, f64::NAN, &mut rng).unwrap_err(),
            MultiHeadAttentionError::ModelWidthNotDivisible {
                model_width: 6,
                heads: 4
            }
        );
        assert_eq!(
            MultiHeadAttention::new("bad", 4, 4, 0, f64::NAN, &mut rng).unwrap_err(),
            MultiHeadAttentionError::OddHeadWidth { head_width: 1 }
        );
        assert_eq!(
            MultiHeadAttention::new("bad", 4, 2, 0, 100.0, &mut rng).unwrap_err(),
            MultiHeadAttentionError::RotaryConfiguration(RopeError::ZeroPositionCapacity)
        );
        assert!(matches!(
            MultiHeadAttention::from_parameters(
                parameter("bad.query.weight", &identity()),
                parameter("bad.key.weight", &identity()),
                parameter("bad.value.weight", &identity()),
                parameter("bad.output.weight", &identity()),
                2,
                4,
                f64::NAN,
            ),
            Err(MultiHeadAttentionError::RotaryConfiguration(
                RopeError::InvalidBase { base }
            )) if base.is_nan()
        ));

        let layer = fixture_layer();
        assert_eq!(
            layer
                .forward(&constant(&[4], &[0.0; 4]), usize::MAX)
                .unwrap_err(),
            MultiHeadAttentionError::InputRank { rank: 1 }
        );
        assert_eq!(
            layer
                .forward(&constant(&[1, 2, 2], &[0.0; 4]), usize::MAX)
                .unwrap_err(),
            MultiHeadAttentionError::InputWidthMismatch {
                expected: 4,
                actual: 2
            }
        );
        assert_eq!(
            layer
                .forward(&constant(&[1, 0, 4], &[]), usize::MAX)
                .unwrap_err(),
            MultiHeadAttentionError::EmptyTokens
        );
        assert_eq!(
            layer
                .forward(&constant(&[1, 1, 4], &[0.0; 4]), usize::MAX)
                .unwrap_err(),
            MultiHeadAttentionError::PositionOffsetOverflow {
                offset: usize::MAX,
                tokens: 1
            }
        );
        assert_eq!(
            layer
                .forward(&constant(&[1, 2, 4], &[0.0; 8]), 5)
                .unwrap_err(),
            MultiHeadAttentionError::PositionRangeExceeded {
                offset: 5,
                tokens: 2,
                max_positions: 6
            }
        );
    }

    #[test]
    fn empty_batches_and_single_tokens_remain_differentiable() {
        let layer = fixture_layer();
        let empty = TensorValue::parameter(tensor(&[0, 3, 4], &[])).unwrap();
        let pass = layer.forward(&empty, 0).unwrap();
        assert_eq!(pass.attention_weights().shape(), [0, 2, 3, 3]);
        assert_eq!(pass.output().shape(), [0, 3, 4]);
        sum_to_scalar(pass.output().clone()).backward().unwrap();
        assert_eq!(empty.gradient().unwrap().shape(), [0, 3, 4]);
        for parameter in layer.parameters() {
            assert_eq!(
                parameter.tensor().gradient().unwrap().as_slice(),
                &[0.0; 16]
            );
        }

        let one = fixture_layer()
            .forward(&constant(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]), 2)
            .unwrap();
        assert_eq!(one.attention_weights().value().as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn initialization_is_transactional_and_reproducible() {
        let mut first_rng = SplitMix64::from_seed(77);
        let mut second_rng = SplitMix64::from_seed(77);
        let initial_rng = first_rng.clone();
        let first =
            MultiHeadAttention::new("decoder.attention", 4, 2, 8, 100.0, &mut first_rng).unwrap();
        let second =
            MultiHeadAttention::new("decoder.attention", 4, 2, 8, 100.0, &mut second_rng).unwrap();
        assert_ne!(first_rng, initial_rng);
        assert_eq!(first_rng, second_rng);
        assert_eq!(
            first
                .parameters()
                .iter()
                .map(NamedParameter::name)
                .collect::<Vec<_>>(),
            [
                "decoder.attention.query.weight",
                "decoder.attention.key.weight",
                "decoder.attention.value.weight",
                "decoder.attention.output.weight",
            ]
        );
        for (left, right) in first.parameters().iter().zip(second.parameters()) {
            assert_eq!(left.name(), right.name());
            assert_eq!(&*left.tensor().value(), &*right.tensor().value());
        }

        let mut rejected = SplitMix64::from_seed(77);
        let before = rejected.clone();
        assert!(
            MultiHeadAttention::new("decoder.attention", 4, 4, 8, 100.0, &mut rejected).is_err()
        );
        assert_eq!(rejected, before);

        let mut invalid_name = SplitMix64::from_seed(77);
        let before = invalid_name.clone();
        assert!(matches!(
            MultiHeadAttention::new("decoder attention", 4, 2, 8, 100.0, &mut invalid_name),
            Err(MultiHeadAttentionError::QkvProjection(
                QkvError::Projection {
                    projection: super::super::qkv::QkvProjection::Query,
                    source: LinearError::Initialization(
                        InitializationError::InvalidNameCharacter { .. }
                    ),
                }
            ))
        ));
        assert_eq!(invalid_name, before);
    }
}
