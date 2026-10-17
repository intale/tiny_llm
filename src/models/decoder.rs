//! A complete bias-free decoder language model with a tied vocabulary head.

use std::error::Error;
use std::fmt;

use crate::attention::multi_head::{MultiHeadAttention, MultiHeadAttentionError};
use crate::attention::qkv::{QkvError, QkvProjection};
use crate::attention::rope::RotaryEmbedding;
use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::models::decoder_block::{
    DecoderBlock, DecoderBlockComponent, DecoderBlockConfig, DecoderBlockError, DecoderBlockForward,
};
use crate::nn::embedding::{Embedding, EmbeddingError};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::nn::linear::LinearError;
use crate::nn::rms_norm::{RmsNorm, RmsNormError, RmsNormForward};
use crate::nn::swiglu::{SwiGlu, SwiGluError, SwiGluProjection};
use crate::tensor::storage::Tensor;

const BLOCK_PARAMETER_SUFFIXES: [&str; 9] = [
    "attention_norm.gain",
    "attention.query.weight",
    "attention.key.weight",
    "attention.value.weight",
    "attention.output.weight",
    "ffn_norm.gain",
    "ffn.gate.weight",
    "ffn.up.weight",
    "ffn.down.weight",
];

/// A model-owned taped operation that rejected a forward or loss calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderModelStage {
    TiedWeightTranspose,
    TiedVocabularyProjection,
    IndexedMeanLoss,
}

impl fmt::Display for DecoderModelStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TiedWeightTranspose => "transpose tied embedding weight",
            Self::TiedVocabularyProjection => "project tied vocabulary logits",
            Self::IndexedMeanLoss => "indexed mean negative log likelihood",
        })
    }
}

/// Which normalization inside a repeated block disagreed with model config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderModelNorm {
    Attention,
    FeedForward,
}

impl fmt::Display for DecoderModelNorm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Attention => "attention",
            Self::FeedForward => "feed-forward",
        })
    }
}

/// A rejected decoder configuration, component assembly, input, or tape stage.
#[derive(Clone, Debug, PartialEq)]
pub enum DecoderModelError {
    EmptyVocabulary,
    ZeroModelWidth,
    ZeroHeadCount,
    ModelWidthNotDivisible {
        model_width: usize,
        heads: usize,
    },
    OddHeadWidth {
        head_width: usize,
    },
    ZeroFeedForwardWidth,
    ZeroPositionCapacity,
    InvalidRopeBase {
        value: f64,
    },
    InvalidRmsEpsilon {
        value: f64,
    },
    LayerAllocationFailed {
        layers: usize,
    },
    ParameterAllocationFailed {
        tensors: usize,
    },
    ParameterCountMismatch {
        expected: usize,
        actual: usize,
    },
    TargetAllocationFailed {
        targets: usize,
    },
    Embedding(EmbeddingError),
    Block {
        layer: usize,
        source: DecoderBlockError,
    },
    FinalNorm(RmsNormError),
    Initialization(InitializationError),
    LayerCountMismatch {
        expected: usize,
        actual: usize,
    },
    EmbeddingVocabularyMismatch {
        expected: usize,
        actual: usize,
    },
    EmbeddingWidthMismatch {
        expected: usize,
        actual: usize,
    },
    BlockModelWidthMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    BlockHeadCountMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    BlockFeedForwardWidthMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    BlockPositionCapacityMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    BlockRopeBaseMismatch {
        layer: usize,
        expected: f64,
        actual: f64,
    },
    BlockRmsEpsilonMismatch {
        layer: usize,
        norm: DecoderModelNorm,
        expected: f64,
        actual: f64,
    },
    FinalNormWidthMismatch {
        expected: usize,
        actual: usize,
    },
    FinalNormEpsilonMismatch {
        expected: f64,
        actual: f64,
    },
    ParameterNameMismatch {
        index: usize,
        expected: String,
        actual: String,
    },
    TokenRank {
        rank: usize,
    },
    EmptyBatch,
    EmptyTokens,
    TokenCountOverflow {
        batch: usize,
        tokens: usize,
    },
    TokenCountMismatch {
        expected: usize,
        actual: usize,
    },
    ContextLengthExceeded {
        tokens: usize,
        max_positions: usize,
    },
    TargetCountMismatch {
        expected: usize,
        actual: usize,
    },
    TargetIdOutOfBounds {
        position: usize,
        id: u32,
        vocabulary_size: usize,
    },
    Autodiff {
        stage: DecoderModelStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for DecoderModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVocabulary => {
                formatter.write_str("decoder vocabulary must contain at least one token")
            }
            Self::ZeroModelWidth => formatter.write_str("decoder model width must be nonzero"),
            Self::ZeroHeadCount => formatter.write_str("decoder head count must be nonzero"),
            Self::ModelWidthNotDivisible { model_width, heads } => write!(
                formatter,
                "decoder model width {model_width} must be divisible by head count {heads}"
            ),
            Self::OddHeadWidth { head_width } => write!(
                formatter,
                "decoder per-head width must be even for RoPE, got {head_width}"
            ),
            Self::ZeroFeedForwardWidth => {
                formatter.write_str("decoder feed-forward width must be nonzero")
            }
            Self::ZeroPositionCapacity => {
                formatter.write_str("decoder context capacity must be nonzero")
            }
            Self::InvalidRopeBase { value } => {
                write!(
                    formatter,
                    "decoder RoPE base must be finite and positive, got {value:?}"
                )
            }
            Self::InvalidRmsEpsilon { value } => write!(
                formatter,
                "decoder RMSNorm epsilon must be finite and nonnegative, got {value:?}"
            ),
            Self::LayerAllocationFailed { layers } => {
                write!(formatter, "could not reserve {layers} decoder blocks")
            }
            Self::ParameterAllocationFailed { tensors } => {
                write!(
                    formatter,
                    "could not reserve {tensors} decoder parameter handles"
                )
            }
            Self::ParameterCountMismatch { expected, actual } => write!(
                formatter,
                "decoder parameter count must be {expected}, got {actual}"
            ),
            Self::TargetAllocationFailed { targets } => {
                write!(
                    formatter,
                    "could not reserve {targets} decoder target indices"
                )
            }
            Self::Embedding(source) => write!(formatter, "token embedding: {source}"),
            Self::Block { layer, source } => {
                write!(formatter, "decoder block {layer}: {source}")
            }
            Self::FinalNorm(source) => write!(formatter, "final RMSNorm: {source}"),
            Self::Initialization(source) => source.fmt(formatter),
            Self::LayerCountMismatch { expected, actual } => write!(
                formatter,
                "decoder config needs {expected} blocks, but received {actual}"
            ),
            Self::EmbeddingVocabularyMismatch { expected, actual } => write!(
                formatter,
                "token embedding vocabulary must be {expected}, got {actual}"
            ),
            Self::EmbeddingWidthMismatch { expected, actual } => write!(
                formatter,
                "token embedding width must be {expected}, got {actual}"
            ),
            Self::BlockModelWidthMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} model width must be {expected}, got {actual}"
            ),
            Self::BlockHeadCountMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} head count must be {expected}, got {actual}"
            ),
            Self::BlockFeedForwardWidthMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} feed-forward width must be {expected}, got {actual}"
            ),
            Self::BlockPositionCapacityMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} position capacity must be {expected}, got {actual}"
            ),
            Self::BlockRopeBaseMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} RoPE base must be {expected:?}, got {actual:?}"
            ),
            Self::BlockRmsEpsilonMismatch {
                layer,
                norm,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder block {layer} {norm} RMSNorm epsilon must be {expected:?}, got {actual:?}"
            ),
            Self::FinalNormWidthMismatch { expected, actual } => write!(
                formatter,
                "final RMSNorm width must be {expected}, got {actual}"
            ),
            Self::FinalNormEpsilonMismatch { expected, actual } => write!(
                formatter,
                "final RMSNorm epsilon must be {expected:?}, got {actual:?}"
            ),
            Self::ParameterNameMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder parameter {index} must be named {expected:?}, got {actual:?}"
            ),
            Self::TokenRank { rank } => write!(
                formatter,
                "decoder token shape must have rank two [batch, tokens], got rank {rank}"
            ),
            Self::EmptyBatch => formatter.write_str("decoder batch must be nonempty"),
            Self::EmptyTokens => formatter.write_str("decoder token sequence must be nonempty"),
            Self::TokenCountOverflow { batch, tokens } => write!(
                formatter,
                "decoder token count overflows for batch {batch} and length {tokens}"
            ),
            Self::TokenCountMismatch { expected, actual } => write!(
                formatter,
                "decoder token shape needs {expected} IDs, but received {actual}"
            ),
            Self::ContextLengthExceeded {
                tokens,
                max_positions,
            } => write!(
                formatter,
                "decoder sequence length {tokens} exceeds context capacity {max_positions}"
            ),
            Self::TargetCountMismatch { expected, actual } => write!(
                formatter,
                "decoder loss needs {expected} targets, but received {actual}"
            ),
            Self::TargetIdOutOfBounds {
                position,
                id,
                vocabulary_size,
            } => write!(
                formatter,
                "target token ID {id} at flat position {position} is out of bounds for vocabulary size {vocabulary_size}"
            ),
            Self::Autodiff { stage, source } => write!(formatter, "decoder {stage}: {source}"),
        }
    }
}

impl Error for DecoderModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Embedding(source) => Some(source),
            Self::Block { source, .. } => Some(source),
            Self::FinalNorm(source) => Some(source),
            Self::Initialization(source) => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<InitializationError> for DecoderModelError {
    fn from(source: InitializationError) -> Self {
        Self::Initialization(source)
    }
}

fn autodiff_error(
    stage: DecoderModelStage,
) -> impl FnOnce(TensorAutodiffError) -> DecoderModelError {
    move |source| DecoderModelError::Autodiff { stage, source }
}

/// Every dimension and numerical constant owned by one decoder model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderModelConfig {
    vocabulary_size: usize,
    model_width: usize,
    heads: usize,
    feed_forward_width: usize,
    layers: usize,
    max_positions: usize,
    rope_base: f64,
    rms_epsilon: f64,
}

impl DecoderModelConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocabulary_size: usize,
        model_width: usize,
        heads: usize,
        feed_forward_width: usize,
        layers: usize,
        max_positions: usize,
        rope_base: f64,
        rms_epsilon: f64,
    ) -> Self {
        Self {
            vocabulary_size,
            model_width,
            heads,
            feed_forward_width,
            layers,
            max_positions,
            rope_base,
            rms_epsilon,
        }
    }

    pub fn vocabulary_size(self) -> usize {
        self.vocabulary_size
    }

    pub fn model_width(self) -> usize {
        self.model_width
    }

    pub fn heads(self) -> usize {
        self.heads
    }

    pub fn head_width(self) -> Option<usize> {
        if self.heads == 0 || !self.model_width.is_multiple_of(self.heads) {
            None
        } else {
            Some(self.model_width / self.heads)
        }
    }

    pub fn feed_forward_width(self) -> usize {
        self.feed_forward_width
    }

    pub fn layers(self) -> usize {
        self.layers
    }

    pub fn max_positions(self) -> usize {
        self.max_positions
    }

    pub fn rope_base(self) -> f64 {
        self.rope_base
    }

    pub fn rms_epsilon(self) -> f64 {
        self.rms_epsilon
    }

    pub fn block_config(self) -> DecoderBlockConfig {
        DecoderBlockConfig::new(
            self.model_width,
            self.heads,
            self.feed_forward_width,
            self.max_positions,
            self.rope_base,
            self.rms_epsilon,
        )
    }
}

fn validate_config(config: DecoderModelConfig) -> Result<(), DecoderModelError> {
    if config.vocabulary_size == 0 {
        return Err(DecoderModelError::EmptyVocabulary);
    }
    if config.model_width == 0 {
        return Err(DecoderModelError::ZeroModelWidth);
    }
    if config.heads == 0 {
        return Err(DecoderModelError::ZeroHeadCount);
    }
    if !config.model_width.is_multiple_of(config.heads) {
        return Err(DecoderModelError::ModelWidthNotDivisible {
            model_width: config.model_width,
            heads: config.heads,
        });
    }
    let head_width = config.model_width / config.heads;
    if !head_width.is_multiple_of(2) {
        return Err(DecoderModelError::OddHeadWidth { head_width });
    }
    if config.feed_forward_width == 0 {
        return Err(DecoderModelError::ZeroFeedForwardWidth);
    }
    if config.max_positions == 0 {
        return Err(DecoderModelError::ZeroPositionCapacity);
    }
    if !config.rope_base.is_finite() || config.rope_base <= 0.0 {
        return Err(DecoderModelError::InvalidRopeBase {
            value: config.rope_base,
        });
    }
    if !config.rms_epsilon.is_finite() || config.rms_epsilon < 0.0 {
        return Err(DecoderModelError::InvalidRmsEpsilon {
            value: config.rms_epsilon,
        });
    }
    Ok(())
}

fn expected_parameter_tensors(layers: usize) -> Result<usize, DecoderModelError> {
    layers
        .checked_mul(BLOCK_PARAMETER_SUFFIXES.len())
        .and_then(|count| count.checked_add(2))
        .ok_or(DecoderModelError::ParameterAllocationFailed {
            tensors: usize::MAX,
        })
}

/// Inspectable values from lookup through the tied vocabulary projection.
#[derive(Clone, Debug)]
pub struct DecoderModelForward {
    embedding: TensorValue,
    blocks: Vec<DecoderBlockForward>,
    final_norm: RmsNormForward,
    logits: TensorValue,
}

impl DecoderModelForward {
    pub fn embedding(&self) -> &TensorValue {
        &self.embedding
    }

    pub fn blocks(&self) -> &[DecoderBlockForward] {
        &self.blocks
    }

    pub fn final_norm(&self) -> &RmsNormForward {
        &self.final_norm
    }

    pub fn logits(&self) -> &TensorValue {
        &self.logits
    }

    pub fn into_logits(self) -> TensorValue {
        self.logits
    }
}

/// A crate-private stable parameter list that exposes each tensor for one scoped read.
///
/// The count, names, and tensors must describe the same immutable layout for the
/// complete `validate_parameter_layout` call.
pub(crate) trait DecoderParameterSource {
    fn len(&self) -> usize;
    fn name(&self, index: usize) -> &str;
    fn with_tensor<R>(&self, index: usize, inspect: impl FnOnce(&Tensor) -> R) -> R;
}

impl DecoderParameterSource for [NamedParameter] {
    fn len(&self) -> usize {
        <[NamedParameter]>::len(self)
    }

    fn name(&self, index: usize) -> &str {
        self[index].name()
    }

    fn with_tensor<R>(&self, index: usize, inspect: impl FnOnce(&Tensor) -> R) -> R {
        let tensor = self[index].tensor().value();
        inspect(&tensor)
    }
}

/// Validates decoder-wide relationships without moving or copying tensor values.
pub(crate) fn validate_parameter_layout<S: DecoderParameterSource + ?Sized>(
    config: DecoderModelConfig,
    parameters: &S,
) -> Result<(), DecoderModelError> {
    validate_config(config)?;
    let expected = expected_parameter_tensors(config.layers)?;
    if parameters.len() != expected {
        return Err(DecoderModelError::ParameterCountMismatch {
            expected,
            actual: parameters.len(),
        });
    }
    validate_parameter_names(parameters, config.layers)?;

    let (vocabulary_size, embedding_width) = embedding_dimensions(parameters, 0)?;
    let mut model_mismatch = if vocabulary_size != config.vocabulary_size {
        Some(DecoderModelError::EmbeddingVocabularyMismatch {
            expected: config.vocabulary_size,
            actual: vocabulary_size,
        })
    } else if embedding_width != config.model_width {
        Some(DecoderModelError::EmbeddingWidthMismatch {
            expected: config.model_width,
            actual: embedding_width,
        })
    } else {
        None
    };
    for layer in 0..config.layers {
        let dimensions = validate_block_parameter_shapes(parameters, layer, config)?;
        if model_mismatch.is_none() {
            model_mismatch = if dimensions.model_width != config.model_width {
                Some(DecoderModelError::BlockModelWidthMismatch {
                    layer,
                    expected: config.model_width,
                    actual: dimensions.model_width,
                })
            } else if dimensions.feed_forward_width != config.feed_forward_width {
                Some(DecoderModelError::BlockFeedForwardWidthMismatch {
                    layer,
                    expected: config.feed_forward_width,
                    actual: dimensions.feed_forward_width,
                })
            } else {
                None
            };
        }
    }
    let final_width = norm_width(parameters, expected - 1).map_err(DecoderModelError::FinalNorm)?;
    if model_mismatch.is_none() && final_width != config.model_width {
        model_mismatch = Some(DecoderModelError::FinalNormWidthMismatch {
            expected: config.model_width,
            actual: final_width,
        });
    }
    match model_mismatch {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
// endregion:decoder-parameter-layout

fn validate_parameter_names<S: DecoderParameterSource + ?Sized>(
    parameters: &S,
    layers: usize,
) -> Result<(), DecoderModelError> {
    let final_index = parameters.len() - 1;
    for index in 0..parameters.len() {
        let expected = if index == 0 {
            "token_embedding.weight".to_owned()
        } else if index == final_index {
            "final_norm.gain".to_owned()
        } else {
            let block_offset = index - 1;
            let layer = block_offset / BLOCK_PARAMETER_SUFFIXES.len();
            debug_assert!(layer < layers);
            let suffix = BLOCK_PARAMETER_SUFFIXES[block_offset % BLOCK_PARAMETER_SUFFIXES.len()];
            format!("blocks.{layer}.{suffix}")
        };
        if parameters.name(index) != expected {
            return Err(DecoderModelError::ParameterNameMismatch {
                index,
                expected,
                actual: parameters.name(index).to_owned(),
            });
        }
    }
    Ok(())
}

fn block_parameter_start(layer: usize) -> usize {
    1 + layer * BLOCK_PARAMETER_SUFFIXES.len()
}

fn embedding_dimensions<S: DecoderParameterSource + ?Sized>(
    parameters: &S,
    index: usize,
) -> Result<(usize, usize), DecoderModelError> {
    parameters.with_tensor(index, |tensor| {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(DecoderModelError::Embedding(EmbeddingError::TableRank {
                rank: shape.len(),
            }));
        }
        if shape[0] == 0 {
            return Err(DecoderModelError::Embedding(
                EmbeddingError::EmptyVocabulary,
            ));
        }
        if shape[1] == 0 {
            return Err(DecoderModelError::Embedding(
                EmbeddingError::ZeroEmbeddingWidth,
            ));
        }
        Ok((shape[0], shape[1]))
    })
}

fn norm_width<S: DecoderParameterSource + ?Sized>(
    parameters: &S,
    index: usize,
) -> Result<usize, RmsNormError> {
    parameters.with_tensor(index, |tensor| {
        let shape = tensor.shape();
        if shape.len() != 1 {
            return Err(RmsNormError::GainRank {
                shape: shape.to_vec(),
            });
        }
        if shape[0] == 0 {
            return Err(RmsNormError::EmptyFeatureWidth);
        }
        Ok(shape[0])
    })
}

fn linear_dimensions<S: DecoderParameterSource + ?Sized>(
    parameters: &S,
    index: usize,
) -> Result<(usize, usize), LinearError> {
    parameters.with_tensor(index, |tensor| {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(LinearError::WeightRank { rank: shape.len() });
        }
        if shape[0] == 0 {
            return Err(LinearError::ZeroInputWidth);
        }
        if shape[1] == 0 {
            return Err(LinearError::ZeroOutputWidth);
        }
        Ok((shape[0], shape[1]))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockParameterDimensions {
    model_width: usize,
    feed_forward_width: usize,
}

fn validate_block_parameter_shapes<S: DecoderParameterSource + ?Sized>(
    parameters: &S,
    layer: usize,
    config: DecoderModelConfig,
) -> Result<BlockParameterDimensions, DecoderModelError> {
    let start = block_parameter_start(layer);
    let attention_norm =
        norm_width(parameters, start).map_err(|source| DecoderModelError::Block {
            layer,
            source: DecoderBlockError::AttentionNorm(source),
        })?;
    let projections = [
        (QkvProjection::Query, start + 1),
        (QkvProjection::Key, start + 2),
        (QkvProjection::Value, start + 3),
    ];
    let mut qkv_dimensions = [(0, 0); 3];
    for (slot, (projection, index)) in projections.into_iter().enumerate() {
        qkv_dimensions[slot] =
            linear_dimensions(parameters, index).map_err(|source| DecoderModelError::Block {
                layer,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                    QkvError::Projection { projection, source },
                )),
            })?;
    }
    let [query, key, value] = qkv_dimensions;
    if query.0 != key.0 || query.0 != value.0 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                QkvError::BranchInputWidthMismatch {
                    query: query.0,
                    key: key.0,
                    value: value.0,
                },
            )),
        });
    }
    if query.1 != key.1 || query.1 != value.1 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                QkvError::BranchOutputWidthMismatch {
                    query: query.1,
                    key: key.1,
                    value: value.1,
                },
            )),
        });
    }
    let output =
        linear_dimensions(parameters, start + 4).map_err(|source| DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::OutputProjection(source)),
        })?;
    if !query.0.is_multiple_of(config.heads) {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::ModelWidthNotDivisible {
                model_width: query.0,
                heads: config.heads,
            }),
        });
    }
    let head_width = query.0 / config.heads;
    if !head_width.is_multiple_of(2) {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::OddHeadWidth {
                head_width,
            }),
        });
    }
    if query.1 != query.0 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvOutputWidthMismatch {
                model_width: query.0,
                projected_width: query.1,
            }),
        });
    }
    if output.0 != query.0 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(
                MultiHeadAttentionError::OutputInputWidthMismatch {
                    expected: query.0,
                    actual: output.0,
                },
            ),
        });
    }
    if output.1 != query.0 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::OutputWidthMismatch {
                expected: query.0,
                actual: output.1,
            }),
        });
    }
    RotaryEmbedding::validate_table_specification(
        head_width,
        config.max_positions,
        config.rope_base,
    )
    .map_err(|source| DecoderModelError::Block {
        layer,
        source: DecoderBlockError::Attention(MultiHeadAttentionError::RotaryConfiguration(source)),
    })?;

    let feed_forward_norm =
        norm_width(parameters, start + 5).map_err(|source| DecoderModelError::Block {
            layer,
            source: DecoderBlockError::FeedForwardNorm(source),
        })?;
    let feed_forward_projections = [
        (SwiGluProjection::Gate, start + 6),
        (SwiGluProjection::Up, start + 7),
        (SwiGluProjection::Down, start + 8),
    ];
    let mut feed_forward_dimensions = [(0, 0); 3];
    for (slot, (projection, index)) in feed_forward_projections.into_iter().enumerate() {
        feed_forward_dimensions[slot] =
            linear_dimensions(parameters, index).map_err(|source| DecoderModelError::Block {
                layer,
                source: DecoderBlockError::FeedForward(SwiGluError::Projection {
                    projection,
                    source,
                }),
            })?;
    }
    let [gate, up, down] = feed_forward_dimensions;
    if gate.0 != up.0 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::FeedForward(SwiGluError::BranchInputWidthMismatch {
                gate: gate.0,
                up: up.0,
            }),
        });
    }
    if gate.1 != up.1 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::FeedForward(SwiGluError::BranchHiddenWidthMismatch {
                gate: gate.1,
                up: up.1,
            }),
        });
    }
    if down.0 != gate.1 {
        return Err(DecoderModelError::Block {
            layer,
            source: DecoderBlockError::FeedForward(SwiGluError::DownInputWidthMismatch {
                hidden: gate.1,
                down: down.0,
            }),
        });
    }

    for (component, actual) in [
        (DecoderBlockComponent::AttentionNorm, attention_norm),
        (DecoderBlockComponent::FeedForwardNorm, feed_forward_norm),
        (DecoderBlockComponent::FeedForwardInput, gate.0),
        (DecoderBlockComponent::FeedForwardOutput, down.1),
    ] {
        if actual != query.0 {
            return Err(DecoderModelError::Block {
                layer,
                source: DecoderBlockError::ComponentWidthMismatch {
                    component,
                    expected: query.0,
                    actual,
                },
            });
        }
    }
    Ok(BlockParameterDimensions {
        model_width: query.0,
        feed_forward_width: gate.1,
    })
}

/// Token lookup, repeated decoder blocks, final RMSNorm, and one tied head
#[derive(Clone, Debug)]
pub struct DecoderModel {
    config: DecoderModelConfig,
    embedding: Embedding,
    blocks: Vec<DecoderBlock>,
    final_norm: RmsNorm,
    parameters: NamedParameters,
}

impl DecoderModel {
    pub fn new(
        config: DecoderModelConfig,
        rng: &mut SplitMix64,
    ) -> Result<Self, DecoderModelError> {
        validate_config(config)?;
        let mut trial = rng.clone();
        let embedding = Embedding::new(
            "token_embedding.weight",
            config.vocabulary_size,
            config.model_width,
            &mut trial,
        )
        .map_err(DecoderModelError::Embedding)?;
        let mut blocks = Vec::new();
        blocks.try_reserve_exact(config.layers).map_err(|_| {
            DecoderModelError::LayerAllocationFailed {
                layers: config.layers,
            }
        })?;
        for layer in 0..config.layers {
            blocks.push(
                DecoderBlock::new(format!("blocks.{layer}"), config.block_config(), &mut trial)
                    .map_err(|source| DecoderModelError::Block { layer, source })?,
            );
        }
        let final_norm = RmsNorm::new("final_norm.gain", config.model_width, config.rms_epsilon)
            .map_err(DecoderModelError::FinalNorm)?;
        let model = Self::from_parts(config, embedding, blocks, final_norm)?;
        *rng = trial;
        Ok(model)
    }

    /// Rebuilds every component handle from one exact stable-order parameter set.
    ///
    /// State restoration uses this construction boundary to create an isolated
    /// decoder. Ordinary optimizer steps instead update the existing leaves, so
    /// the registry, components, and tied embedding keep their live aliases.
    pub fn from_parameters(
        config: DecoderModelConfig,
        parameters: Vec<NamedParameter>,
    ) -> Result<Self, DecoderModelError> {
        validate_parameter_layout(config, parameters.as_slice())?;
        let expected = parameters.len();

        let embedding = Embedding::from_parameter(parameters[0].clone())
            .map_err(DecoderModelError::Embedding)?;
        let mut blocks = Vec::new();
        blocks.try_reserve_exact(config.layers).map_err(|_| {
            DecoderModelError::LayerAllocationFailed {
                layers: config.layers,
            }
        })?;
        for layer in 0..config.layers {
            let start = 1 + layer * BLOCK_PARAMETER_SUFFIXES.len();
            let attention_norm = RmsNorm::from_gain(parameters[start].clone(), config.rms_epsilon)
                .map_err(|source| DecoderModelError::Block {
                    layer,
                    source: DecoderBlockError::AttentionNorm(source),
                })?;
            let attention = MultiHeadAttention::from_parameters(
                parameters[start + 1].clone(),
                parameters[start + 2].clone(),
                parameters[start + 3].clone(),
                parameters[start + 4].clone(),
                config.heads,
                config.max_positions,
                config.rope_base,
            )
            .map_err(|source| DecoderModelError::Block {
                layer,
                source: DecoderBlockError::Attention(source),
            })?;
            let feed_forward_norm =
                RmsNorm::from_gain(parameters[start + 5].clone(), config.rms_epsilon).map_err(
                    |source| DecoderModelError::Block {
                        layer,
                        source: DecoderBlockError::FeedForwardNorm(source),
                    },
                )?;
            let feed_forward = SwiGlu::from_parameters(
                parameters[start + 6].clone(),
                parameters[start + 7].clone(),
                parameters[start + 8].clone(),
            )
            .map_err(|source| DecoderModelError::Block {
                layer,
                source: DecoderBlockError::FeedForward(source),
            })?;
            blocks.push(
                DecoderBlock::from_parts(
                    attention_norm,
                    attention,
                    feed_forward_norm,
                    feed_forward,
                )
                .map_err(|source| DecoderModelError::Block { layer, source })?,
            );
        }
        let final_norm = RmsNorm::from_gain(parameters[expected - 1].clone(), config.rms_epsilon)
            .map_err(DecoderModelError::FinalNorm)?;
        Self::from_parts(config, embedding, blocks, final_norm)
    }

    /// Assembles exact named components and rejects any configuration drift.
    pub fn from_parts(
        config: DecoderModelConfig,
        embedding: Embedding,
        blocks: Vec<DecoderBlock>,
        final_norm: RmsNorm,
    ) -> Result<Self, DecoderModelError> {
        validate_config(config)?;
        if embedding.vocabulary_size() != config.vocabulary_size {
            return Err(DecoderModelError::EmbeddingVocabularyMismatch {
                expected: config.vocabulary_size,
                actual: embedding.vocabulary_size(),
            });
        }
        if embedding.embedding_width() != config.model_width {
            return Err(DecoderModelError::EmbeddingWidthMismatch {
                expected: config.model_width,
                actual: embedding.embedding_width(),
            });
        }
        if blocks.len() != config.layers {
            return Err(DecoderModelError::LayerCountMismatch {
                expected: config.layers,
                actual: blocks.len(),
            });
        }
        for (layer, block) in blocks.iter().enumerate() {
            if block.model_width() != config.model_width {
                return Err(DecoderModelError::BlockModelWidthMismatch {
                    layer,
                    expected: config.model_width,
                    actual: block.model_width(),
                });
            }
            if block.attention().heads() != config.heads {
                return Err(DecoderModelError::BlockHeadCountMismatch {
                    layer,
                    expected: config.heads,
                    actual: block.attention().heads(),
                });
            }
            if block.feed_forward().hidden_width() != config.feed_forward_width {
                return Err(DecoderModelError::BlockFeedForwardWidthMismatch {
                    layer,
                    expected: config.feed_forward_width,
                    actual: block.feed_forward().hidden_width(),
                });
            }
            if block.attention().rope().max_positions() != config.max_positions {
                return Err(DecoderModelError::BlockPositionCapacityMismatch {
                    layer,
                    expected: config.max_positions,
                    actual: block.attention().rope().max_positions(),
                });
            }
            if block.attention().rope().base().to_bits() != config.rope_base.to_bits() {
                return Err(DecoderModelError::BlockRopeBaseMismatch {
                    layer,
                    expected: config.rope_base,
                    actual: block.attention().rope().base(),
                });
            }
            for (norm, epsilon) in [
                (
                    DecoderModelNorm::Attention,
                    block.attention_norm().epsilon(),
                ),
                (
                    DecoderModelNorm::FeedForward,
                    block.feed_forward_norm().epsilon(),
                ),
            ] {
                if epsilon.to_bits() != config.rms_epsilon.to_bits() {
                    return Err(DecoderModelError::BlockRmsEpsilonMismatch {
                        layer,
                        norm,
                        expected: config.rms_epsilon,
                        actual: epsilon,
                    });
                }
            }
        }
        if final_norm.feature_width() != config.model_width {
            return Err(DecoderModelError::FinalNormWidthMismatch {
                expected: config.model_width,
                actual: final_norm.feature_width(),
            });
        }
        if final_norm.epsilon().to_bits() != config.rms_epsilon.to_bits() {
            return Err(DecoderModelError::FinalNormEpsilonMismatch {
                expected: config.rms_epsilon,
                actual: final_norm.epsilon(),
            });
        }

        let parameter_tensors = expected_parameter_tensors(config.layers)?;
        let mut listed = Vec::new();
        listed.try_reserve_exact(parameter_tensors).map_err(|_| {
            DecoderModelError::ParameterAllocationFailed {
                tensors: parameter_tensors,
            }
        })?;
        listed.push(embedding.table().clone());
        for block in &blocks {
            listed.extend(block.parameters().iter().cloned());
        }
        listed.push(final_norm.gain().clone());
        validate_parameter_names(listed.as_slice(), config.layers)?;
        let parameters = NamedParameters::try_new(listed)?;

        Ok(Self {
            config,
            embedding,
            blocks,
            final_norm,
            parameters,
        })
    }

    /// Runs the complete model while retaining one evidence record per layer.
    pub fn forward(
        &self,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<DecoderModelForward, DecoderModelError> {
        self.forward_with_context(AutogradContext::recording(), token_ids, token_shape)
    }

    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<DecoderModelForward, DecoderModelError> {
        self.validate_tokens(token_ids, token_shape)?;
        let embedding = self
            .embedding
            .forward_with_context(context, token_ids, token_shape)
            .map_err(DecoderModelError::Embedding)?;
        let mut current = embedding.clone();
        let mut block_forwards = Vec::new();
        block_forwards
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| DecoderModelError::LayerAllocationFailed {
                layers: self.blocks.len(),
            })?;
        for (layer, block) in self.blocks.iter().enumerate() {
            let forward = block
                .forward_with_context(context, &current, 0)
                .map_err(|source| DecoderModelError::Block { layer, source })?;
            current = forward.output().clone();
            block_forwards.push(forward);
        }
        let final_norm = self
            .final_norm
            .forward_with_intermediates_and_context(context, &current)
            .map_err(DecoderModelError::FinalNorm)?;
        let tied_weight = self
            .embedding
            .table()
            .tensor()
            .transpose_with_context(context, 0, 1)
            .map_err(autodiff_error(DecoderModelStage::TiedWeightTranspose))?;
        let logits = final_norm
            .output()
            .matmul_with_context(context, &tied_weight)
            .map_err(autodiff_error(DecoderModelStage::TiedVocabularyProjection))?;

        Ok(DecoderModelForward {
            embedding,
            blocks: block_forwards,
            final_norm,
            logits,
        })
    }

    /// Computes one mean next-token loss over the vocabulary axis.
    pub fn loss(
        &self,
        token_ids: &[u32],
        token_shape: &[usize],
        targets: &[u32],
    ) -> Result<TensorValue, DecoderModelError> {
        self.loss_with_context(
            AutogradContext::recording(),
            token_ids,
            token_shape,
            targets,
        )
    }

    pub fn loss_with_context(
        &self,
        context: AutogradContext,
        token_ids: &[u32],
        token_shape: &[usize],
        targets: &[u32],
    ) -> Result<TensorValue, DecoderModelError> {
        let expected = self.validate_tokens(token_ids, token_shape)?;
        if targets.len() != expected {
            return Err(DecoderModelError::TargetCountMismatch {
                expected,
                actual: targets.len(),
            });
        }
        let mut target_indices = Vec::new();
        target_indices
            .try_reserve_exact(expected)
            .map_err(|_| DecoderModelError::TargetAllocationFailed { targets: expected })?;
        for (position, &id) in targets.iter().enumerate() {
            let valid = usize::try_from(id)
                .ok()
                .is_some_and(|index| index < self.config.vocabulary_size);
            if !valid {
                return Err(DecoderModelError::TargetIdOutOfBounds {
                    position,
                    id,
                    vocabulary_size: self.config.vocabulary_size,
                });
            }
            target_indices
                .push(usize::try_from(id).expect("validated target token ID must fit usize"))
        }
        self.forward_with_context(context, token_ids, token_shape)?
            .into_logits()
            .indexed_mean_nll_with_context(context, 2, &target_indices)
            .map_err(autodiff_error(DecoderModelStage::IndexedMeanLoss))
    }

    fn validate_tokens(
        &self,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<usize, DecoderModelError> {
        if token_shape.len() != 2 {
            return Err(DecoderModelError::TokenRank {
                rank: token_shape.len(),
            });
        }
        let batch = token_shape[0];
        let tokens = token_shape[1];
        if batch == 0 {
            return Err(DecoderModelError::EmptyBatch);
        }
        if tokens == 0 {
            return Err(DecoderModelError::EmptyTokens);
        }
        if tokens > self.config.max_positions {
            return Err(DecoderModelError::ContextLengthExceeded {
                tokens,
                max_positions: self.config.max_positions,
            });
        }
        let expected = batch
            .checked_mul(tokens)
            .ok_or(DecoderModelError::TokenCountOverflow { batch, tokens })?;
        if token_ids.len() != expected {
            return Err(DecoderModelError::TokenCountMismatch {
                expected,
                actual: token_ids.len(),
            });
        }
        Ok(expected)
    }

    pub fn config(&self) -> DecoderModelConfig {
        self.config
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    pub fn blocks(&self) -> &[DecoderBlock] {
        &self.blocks
    }

    pub fn final_norm(&self) -> &RmsNorm {
        &self.final_norm
    }

    /// The sole table used for both token lookup and vocabulary projection.
    pub fn tied_embedding(&self) -> &NamedParameter {
        self.embedding.table()
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters
            .iter()
            .map(|parameter| parameter.tensor().value().len())
            .sum()
    }
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::support::gradcheck::sampled_tensor_gradient_check;
    use crate::autograd::tensor_core::GraphRetention;
    use crate::tensor::storage::Tensor;
    use crate::training::adam_w::{AdamW, AdamWConfig};

    const STEP: f64 = 1e-6;
    const TOLERANCE: f64 = 2e-5;

    fn config(layers: usize) -> DecoderModelConfig {
        DecoderModelConfig::new(5, 4, 2, 4, layers, 4, 10_000.0, 1e-6)
    }

    fn tensor(shape: &[usize], values: &[f64]) -> Tensor {
        Tensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
    }

    fn model(layers: usize, seed: u64) -> DecoderModel {
        DecoderModel::new(config(layers), &mut SplitMix64::from_seed(seed)).unwrap()
    }

    fn copied_parameters(model: &DecoderModel) -> Vec<NamedParameter> {
        model
            .parameters()
            .iter()
            .map(|parameter| {
                NamedParameter::from_tensor(parameter.name(), parameter.tensor().value_snapshot())
                    .unwrap()
            })
            .collect()
    }

    #[derive(Clone, Debug)]
    struct BorrowedParameterFixture {
        names: Vec<String>,
        tensors: Vec<Tensor>,
    }

    impl DecoderParameterSource for BorrowedParameterFixture {
        fn len(&self) -> usize {
            self.tensors.len()
        }

        fn name(&self, index: usize) -> &str {
            &self.names[index]
        }

        fn with_tensor<R>(&self, index: usize, inspect: impl FnOnce(&Tensor) -> R) -> R {
            inspect(&self.tensors[index])
        }
    }

    impl BorrowedParameterFixture {
        fn from_model(model: &DecoderModel) -> Self {
            Self {
                names: model
                    .parameters()
                    .iter()
                    .map(|parameter| parameter.name().to_owned())
                    .collect(),
                tensors: model
                    .parameters()
                    .iter()
                    .map(|parameter| parameter.tensor().value_snapshot())
                    .collect(),
            }
        }

        fn replace_shape(&mut self, index: usize, shape: &[usize]) {
            let elements = shape.iter().copied().product();
            self.tensors[index] = Tensor::from_vec(shape.to_vec(), vec![0.0; elements]).unwrap();
        }

        fn storage_addresses(&self) -> Vec<usize> {
            self.tensors
                .iter()
                .map(|tensor| tensor.as_slice().as_ptr() as usize)
                .collect()
        }
    }

    fn borrowed_layout_error(changes: &[(usize, &[usize])]) -> DecoderModelError {
        let mut parameters = BorrowedParameterFixture::from_model(&model(1, 33));
        for &(index, shape) in changes {
            parameters.replace_shape(index, shape);
        }
        validate_parameter_layout(config(1), &parameters).unwrap_err()
    }

    fn zero_layer_model(table: Tensor, gain: Tensor) -> DecoderModel {
        let embedding = Embedding::from_parameter(
            NamedParameter::from_tensor("token_embedding.weight", table).unwrap(),
        )
            .unwrap();
        let final_norm = RmsNorm::from_gain(
            NamedParameter::from_tensor("final_norm.gain", gain).unwrap(),
            1e-6,
        )
            .unwrap();
        DecoderModel::from_parts(config(0), embedding, Vec::new(), final_norm).unwrap()
    }

    #[test]
    fn initialization_is_transactional_reproducible_and_stably_named() {
        let mut rejected = SplitMix64::from_seed(32);
        let state = rejected.state();
        assert_eq!(
            DecoderModel::new(
                DecoderModelConfig::new(0, 4, 2, 4, 2, 4, 10_000.0, 1e-6),
                &mut rejected,
            )
                .unwrap_err(),
            DecoderModelError::EmptyVocabulary
        );
        assert_eq!(rejected.state(), state);

        let left = model(2, 32);
        let right = model(2, 32);
        assert_eq!(left.parameters().len(), 20);
        assert_eq!(left.parameter_count(), 264);
        assert_eq!(
            left.parameters()
                .iter()
                .map(NamedParameter::name)
                .collect::<Vec<_>>(),
            [
                "token_embedding.weight",
                "blocks.0.attention_norm.gain",
                "blocks.0.attention.query.weight",
                "blocks.0.attention.key.weight",
                "blocks.0.attention.value.weight",
                "blocks.0.attention.output.weight",
                "blocks.0.ffn_norm.gain",
                "blocks.0.ffn.gate.weight",
                "blocks.0.ffn.up.weight",
                "blocks.0.ffn.down.weight",
                "blocks.1.attention_norm.gain",
                "blocks.1.attention.query.weight",
                "blocks.1.attention.key.weight",
                "blocks.1.attention.value.weight",
                "blocks.1.attention.output.weight",
                "blocks.1.ffn_norm.gain",
                "blocks.1.ffn.gate.weight",
                "blocks.1.ffn.up.weight",
                "blocks.1.ffn.down.weight",
                "final_norm.gain",
            ]
        );
        for (left, right) in left.parameters().iter().zip(right.parameters()) {
            assert_eq!(&*left.tensor().value(), &*right.tensor().value());
            assert!(!left.tensor().is_same_node(right.tensor()));
            assert!(!left.name().contains("bias"));
        }
        assert!(
            left.embedding()
                .table()
                .tensor()
                .is_same_node(left.tied_embedding().tensor())
        );
    }

    #[test]
    fn parameter_reconstruction_rebinds_every_component_and_rejects_drift() {
        let original = model(1, 33);
        let ids = [0, 1, 2];
        let expected_logits = original
            .forward(&ids, &[1, 3])
            .unwrap()
            .into_logits()
            .value_snapshot();
        let rebuilt =
            DecoderModel::from_parameters(config(1), copied_parameters(&original)).unwrap();

        let rebuilt_logits = rebuilt.forward(&ids, &[1, 3]).unwrap().into_logits();
        assert_eq!(&*rebuilt_logits.value(), &expected_logits);
        assert!(
            rebuilt
                .embedding()
                .table()
                .tensor()
                .is_same_node(rebuilt.parameters()[0].tensor())
        );
        for (component, listed) in rebuilt.blocks()[0]
            .parameters()
            .iter()
            .zip(&rebuilt.parameters()[1..10])
        {
            assert!(component.tensor().is_same_node(listed.tensor()));
        }
        assert!(
            rebuilt
                .final_norm()
                .gain()
                .tensor()
                .is_same_node(rebuilt.parameters().last().unwrap().tensor())
        );
        assert!(rebuilt.parameters().iter().all(|parameter| {
            parameter
                .tensor()
                .gradient()
                .is_some_and(|gradient| gradient.as_slice().iter().all(|value| *value == 0.0))
        }));

        let mut missing = copied_parameters(&original);
        missing.pop();
        assert_eq!(
            DecoderModel::from_parameters(config(1), missing).unwrap_err(),
            DecoderModelError::ParameterCountMismatch {
                expected: 11,
                actual: 10,
            }
        );

        let mut renamed = copied_parameters(&original);
        renamed[1] = NamedParameter::from_tensor(
            "blocks.0.wrong.gain",
            renamed[1].tensor().value_snapshot(),
        )
            .unwrap();
        assert_eq!(
            DecoderModel::from_parameters(config(1), renamed).unwrap_err(),
            DecoderModelError::ParameterNameMismatch {
                index: 1,
                expected: "blocks.0.attention_norm.gain".to_owned(),
                actual: "blocks.0.wrong.gain".to_owned(),
            }
        );

        let mut wrong_shape = copied_parameters(&original);
        wrong_shape[2] =
            NamedParameter::from_tensor("blocks.0.attention.query.weight", tensor(&[1], &[0.0]))
                .unwrap();
        assert!(matches!(
            DecoderModel::from_parameters(config(1), wrong_shape),
            Err(DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(_),
            })
        ));
    }

    #[test]
    fn borrowed_layout_validation_preserves_storage_and_boundary_precedence() {
        let valid = BorrowedParameterFixture::from_model(&model(1, 33));
        let addresses = valid.storage_addresses();
        let bits = valid
            .tensors
            .iter()
            .flat_map(|tensor| tensor.as_slice().iter().map(|value| value.to_bits()))
            .collect::<Vec<_>>();
        validate_parameter_layout(config(1), &valid).unwrap();
        assert_eq!(valid.storage_addresses(), addresses);
        assert_eq!(
            valid
                .tensors
                .iter()
                .flat_map(|tensor| tensor.as_slice().iter().map(|value| value.to_bits()))
                .collect::<Vec<_>>(),
            bits
        );

        let mut invalid_config = valid.clone();
        invalid_config.names[1] = "blocks.0.wrong.gain".to_owned();
        invalid_config.replace_shape(0, &[4]);
        assert_eq!(
            validate_parameter_layout(
                DecoderModelConfig::new(0, 4, 2, 4, 1, 4, 10_000.0, 1e-6),
                &invalid_config,
            ),
            Err(DecoderModelError::EmptyVocabulary)
        );

        let mut wrong_count = invalid_config.clone();
        wrong_count.tensors.pop();
        assert_eq!(
            validate_parameter_layout(config(1), &wrong_count),
            Err(DecoderModelError::ParameterCountMismatch {
                expected: 11,
                actual: 10,
            })
        );

        assert_eq!(
            validate_parameter_layout(config(1), &invalid_config),
            Err(DecoderModelError::ParameterNameMismatch {
                index: 1,
                expected: "blocks.0.attention_norm.gain".to_owned(),
                actual: "blocks.0.wrong.gain".to_owned(),
            })
        );

        let mut local_before_outer = valid.clone();
        local_before_outer.replace_shape(0, &[6, 4]);
        local_before_outer.replace_shape(10, &[4, 1]);
        assert_eq!(
            validate_parameter_layout(config(1), &local_before_outer),
            Err(DecoderModelError::FinalNorm(RmsNormError::GainRank {
                shape: vec![4, 1],
            }))
        );

        let zero_layer = BorrowedParameterFixture::from_model(&model(0, 33));
        validate_parameter_layout(config(0), &zero_layer).unwrap();
        let mut extra_output_head = zero_layer;
        extra_output_head
            .names
            .insert(1, "lm_head.weight".to_owned());
        extra_output_head
            .tensors
            .insert(1, tensor(&[4, 5], &[0.0; 20]));
        assert_eq!(
            validate_parameter_layout(config(0), &extra_output_head),
            Err(DecoderModelError::ParameterCountMismatch {
                expected: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn borrowed_layout_validation_preserves_component_error_taxonomy() {
        assert_eq!(
            borrowed_layout_error(&[(0, &[5])]),
            DecoderModelError::Embedding(EmbeddingError::TableRank { rank: 1 })
        );
        assert_eq!(
            borrowed_layout_error(&[(0, &[0, 4])]),
            DecoderModelError::Embedding(EmbeddingError::EmptyVocabulary)
        );
        assert_eq!(
            borrowed_layout_error(&[(0, &[5, 0])]),
            DecoderModelError::Embedding(EmbeddingError::ZeroEmbeddingWidth)
        );
        assert_eq!(
            borrowed_layout_error(&[(1, &[2, 2])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::AttentionNorm(RmsNormError::GainRank {
                    shape: vec![2, 2],
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(2, &[4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                    QkvError::Projection {
                        projection: QkvProjection::Query,
                        source: LinearError::WeightRank { rank: 1 },
                    },
                )),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(3, &[6, 4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                    QkvError::BranchInputWidthMismatch {
                        query: 4,
                        key: 6,
                        value: 4,
                    },
                )),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(3, &[4, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                    QkvError::BranchOutputWidthMismatch {
                        query: 4,
                        key: 6,
                        value: 4,
                    },
                )),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(5, &[4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::OutputProjection(
                    LinearError::WeightRank { rank: 1 }
                ),),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(2, &[5, 5]), (3, &[5, 5]), (4, &[5, 5]), (5, &[5, 5])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(
                    MultiHeadAttentionError::ModelWidthNotDivisible {
                        model_width: 5,
                        heads: 2,
                    },
                ),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(2, &[6, 6]), (3, &[6, 6]), (4, &[6, 6]), (5, &[6, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::OddHeadWidth {
                    head_width: 3,
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(2, &[4, 6]), (3, &[4, 6]), (4, &[4, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(
                    MultiHeadAttentionError::QkvOutputWidthMismatch {
                        model_width: 4,
                        projected_width: 6,
                    },
                ),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(5, &[6, 4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(
                    MultiHeadAttentionError::OutputInputWidthMismatch {
                        expected: 4,
                        actual: 6,
                    },
                ),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(5, &[4, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::Attention(
                    MultiHeadAttentionError::OutputWidthMismatch {
                        expected: 4,
                        actual: 6,
                    },
                ),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(6, &[2, 2])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::FeedForwardNorm(RmsNormError::GainRank {
                    shape: vec![2, 2],
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(7, &[4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::FeedForward(SwiGluError::Projection {
                    projection: SwiGluProjection::Gate,
                    source: LinearError::WeightRank { rank: 1 },
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(8, &[6, 4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::FeedForward(SwiGluError::BranchInputWidthMismatch {
                    gate: 4,
                    up: 6,
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(8, &[4, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::FeedForward(SwiGluError::BranchHiddenWidthMismatch {
                    gate: 4,
                    up: 6,
                }),
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(9, &[6, 4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::FeedForward(SwiGluError::DownInputWidthMismatch {
                    hidden: 4,
                    down: 6,
                }),
            }
        );
        for (index, shape, component, actual) in [
            (1, &[6][..], DecoderBlockComponent::AttentionNorm, 6),
            (6, &[6][..], DecoderBlockComponent::FeedForwardNorm, 6),
        ] {
            assert_eq!(
                borrowed_layout_error(&[(index, shape)]),
                DecoderModelError::Block {
                    layer: 0,
                    source: DecoderBlockError::ComponentWidthMismatch {
                        component,
                        expected: 4,
                        actual,
                    },
                }
            );
        }
        assert_eq!(
            borrowed_layout_error(&[(7, &[6, 4]), (8, &[6, 4])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::ComponentWidthMismatch {
                    component: DecoderBlockComponent::FeedForwardInput,
                    expected: 4,
                    actual: 6,
                },
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(9, &[4, 6])]),
            DecoderModelError::Block {
                layer: 0,
                source: DecoderBlockError::ComponentWidthMismatch {
                    component: DecoderBlockComponent::FeedForwardOutput,
                    expected: 4,
                    actual: 6,
                },
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(10, &[2, 2])]),
            DecoderModelError::FinalNorm(RmsNormError::GainRank { shape: vec![2, 2] })
        );

        assert_eq!(
            borrowed_layout_error(&[(0, &[6, 4])]),
            DecoderModelError::EmbeddingVocabularyMismatch {
                expected: 5,
                actual: 6,
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(0, &[5, 6])]),
            DecoderModelError::EmbeddingWidthMismatch {
                expected: 4,
                actual: 6,
            }
        );
        assert_eq!(
            borrowed_layout_error(&[
                (1, &[8]),
                (2, &[8, 8]),
                (3, &[8, 8]),
                (4, &[8, 8]),
                (5, &[8, 8]),
                (6, &[8]),
                (7, &[8, 4]),
                (8, &[8, 4]),
                (9, &[4, 8]),
            ]),
            DecoderModelError::BlockModelWidthMismatch {
                layer: 0,
                expected: 4,
                actual: 8,
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(7, &[4, 6]), (8, &[4, 6]), (9, &[6, 4])]),
            DecoderModelError::BlockFeedForwardWidthMismatch {
                layer: 0,
                expected: 4,
                actual: 6,
            }
        );
        assert_eq!(
            borrowed_layout_error(&[(10, &[6])]),
            DecoderModelError::FinalNormWidthMismatch {
                expected: 4,
                actual: 6,
            }
        );
    }

    #[test]
    fn every_parameter_slot_reports_its_exact_rank_owner() {
        let qkv_rank = |projection| DecoderModelError::Block {
            layer: 0,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                QkvError::Projection {
                    projection,
                    source: LinearError::WeightRank { rank: 1 },
                },
            )),
        };
        let swiglu_rank = |projection| DecoderModelError::Block {
            layer: 0,
            source: DecoderBlockError::FeedForward(SwiGluError::Projection {
                projection,
                source: LinearError::WeightRank { rank: 1 },
            }),
        };
        let cases = vec![
            (
                0,
                &[4][..],
                DecoderModelError::Embedding(EmbeddingError::TableRank { rank: 1 }),
            ),
            (
                1,
                &[2, 2][..],
                DecoderModelError::Block {
                    layer: 0,
                    source: DecoderBlockError::AttentionNorm(RmsNormError::GainRank {
                        shape: vec![2, 2],
                    }),
                },
            ),
            (2, &[4][..], qkv_rank(QkvProjection::Query)),
            (3, &[4][..], qkv_rank(QkvProjection::Key)),
            (4, &[4][..], qkv_rank(QkvProjection::Value)),
            (
                5,
                &[4][..],
                DecoderModelError::Block {
                    layer: 0,
                    source: DecoderBlockError::Attention(
                        MultiHeadAttentionError::OutputProjection(LinearError::WeightRank {
                            rank: 1,
                        }),
                    ),
                },
            ),
            (
                6,
                &[2, 2][..],
                DecoderModelError::Block {
                    layer: 0,
                    source: DecoderBlockError::FeedForwardNorm(RmsNormError::GainRank {
                        shape: vec![2, 2],
                    }),
                },
            ),
            (7, &[4][..], swiglu_rank(SwiGluProjection::Gate)),
            (8, &[4][..], swiglu_rank(SwiGluProjection::Up)),
            (9, &[4][..], swiglu_rank(SwiGluProjection::Down)),
            (
                10,
                &[2, 2][..],
                DecoderModelError::FinalNorm(RmsNormError::GainRank { shape: vec![2, 2] }),
            ),
        ];
        for (index, shape, expected) in cases {
            assert_eq!(borrowed_layout_error(&[(index, shape)]), expected);
        }
    }

    #[test]
    fn rope_specification_errors_precede_later_block_and_model_mismatches() {
        let overflowing_config = DecoderModelConfig::new(5, 4, 2, 4, 1, usize::MAX, 10_000.0, 1e-6);
        let expected = DecoderModelError::Block {
            layer: 0,
            source: DecoderBlockError::Attention(MultiHeadAttentionError::RotaryConfiguration(
                crate::attention::rope::RopeError::TableSizeOverflow {
                    positions: usize::MAX,
                    pairs: 2,
                },
            )),
        };
        let mut outer_mismatch = BorrowedParameterFixture::from_model(&model(1, 33));
        for index in 2..=5 {
            outer_mismatch.replace_shape(index, &[8, 8]);
        }
        assert_eq!(
            validate_parameter_layout(overflowing_config, &outer_mismatch),
            Err(expected.clone())
        );

        outer_mismatch.replace_shape(7, &[4]);
        assert_eq!(
            validate_parameter_layout(overflowing_config, &outer_mismatch),
            Err(expected)
        );
    }

    #[test]
    fn later_layer_local_error_precedes_an_earlier_decoder_wide_mismatch() {
        let mut parameters = BorrowedParameterFixture::from_model(&model(2, 33));
        for (index, shape) in [
            (1, &[8][..]),
            (2, &[8, 8][..]),
            (3, &[8, 8][..]),
            (4, &[8, 8][..]),
            (5, &[8, 8][..]),
            (6, &[8][..]),
            (7, &[8, 4][..]),
            (8, &[8, 4][..]),
            (9, &[4, 8][..]),
            (11, &[4][..]),
        ] {
            parameters.replace_shape(index, shape);
        }
        assert_eq!(
            validate_parameter_layout(config(2), &parameters),
            Err(DecoderModelError::Block {
                layer: 1,
                source: DecoderBlockError::Attention(MultiHeadAttentionError::QkvProjection(
                    QkvError::Projection {
                        projection: QkvProjection::Query,
                        source: LinearError::WeightRank { rank: 1 },
                    }
                ),),
            })
        );
    }

    #[test]
    fn adamw_updates_live_registry_component_and_tied_handles_together() {
        let model = model(1, 33);
        let nodes = model
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor().clone())
            .collect::<Vec<_>>();
        let embedding_before = model.tied_embedding().tensor().value_snapshot();
        let embedding = model.tied_embedding();
        let seed = Tensor::from_vec(
            embedding.tensor().shape(),
            vec![0.25; embedding.tensor().value().len()],
        )
            .unwrap();
        embedding
            .tensor()
            .backward_with_seed(&seed.view(), GraphRetention::Retain)
            .unwrap();

        let mut optimizer = AdamW::new(AdamWConfig::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap());
        optimizer.step(model.parameters()).unwrap();

        assert!(
            model
                .parameters()
                .iter()
                .zip(&nodes)
                .all(|(parameter, node)| parameter.tensor().is_same_node(node))
        );
        assert!(
            model
                .embedding()
                .table()
                .tensor()
                .is_same_node(model.tied_embedding().tensor())
        );
        assert_ne!(&*model.tied_embedding().tensor().value(), &embedding_before);
        assert_eq!(
            &*model.embedding().table().tensor().value(),
            &*model.parameters()[0].tensor().value()
        );
    }

    #[test]
    fn zero_one_and_two_block_models_preserve_shapes_and_depth_evidence() {
        let ids = [0, 1, 2, 3, 4, 0];
        for layers in 0..=2 {
            let pass = model(layers, 7).forward(&ids, &[2, 3]).unwrap();
            assert_eq!(pass.embedding().shape(), [2, 3, 4]);
            assert_eq!(pass.blocks().len(), layers);
            assert_eq!(pass.final_norm().output().shape(), [2, 3, 4]);
            assert_eq!(pass.logits().shape(), [2, 3, 5]);
            assert!(
                pass.logits()
                    .value()
                    .as_slice()
                    .iter()
                    .all(|x| x.is_finite())
            );
        }
    }

    #[test]
    fn future_token_changes_cannot_change_earlier_logits() {
        let model = model(2, 19);
        let original = model.forward(&[0, 1, 2], &[1, 3]).unwrap();
        let changed = model.forward(&[0, 1, 4], &[1, 3]).unwrap();
        let original = original.logits().value();
        let changed = changed.logits().value();
        assert_eq!(&original.as_slice()[..10], &changed.as_slice()[..10]);
        assert_ne!(&original.as_slice()[10..], &changed.as_slice()[10..]);
    }

    fn role_gradient(include_lookup: bool, include_head: bool) -> Vec<f64> {
        let values = [
            0.8, -0.2, 0.1, 0.4, -0.3, 0.7, 0.2, -0.1, 0.5, 0.1, -0.6, 0.3, -0.4, -0.2, 0.9, 0.1,
            0.2, 0.3, -0.1, 0.8,
        ];
        let table = NamedParameter::from_tensor("token_embedding.weight", tensor(&[5, 4], &values))
            .unwrap();
        let embedding = Embedding::from_parameter(table.clone()).unwrap();
        let lookup = embedding.forward(&[0, 1], &[1, 2]).unwrap();
        let lookup = if include_lookup {
            lookup
        } else {
            lookup.detach()
        };
        let norm = RmsNorm::new("final_norm.gain", 4, 1e-6)
            .unwrap()
            .forward(&lookup)
            .unwrap();
        let head_source = if include_head {
            table.tensor().clone()
        } else {
            table.tensor().detach()
        };
        let logits = norm.matmul(&head_source.transpose(0, 1).unwrap()).unwrap();
        logits
            .indexed_mean_nll(2, &[1, 2])
            .unwrap()
            .backward_with_seed(&tensor(&[], &[1.0]).view(), GraphRetention::Retain)
            .unwrap();
        table.tensor().gradient().map_or_else(
            || vec![0.0; values.len()],
            |gradient| gradient.as_slice().to_vec(),
        )
    }

    #[test]
    fn tied_table_accumulates_lookup_and_vocabulary_projection_gradients() {
        let full = role_gradient(true, true);
        let lookup = role_gradient(true, false);
        let head = role_gradient(false, true);
        for index in 0..full.len() {
            assert!((full[index] - lookup[index] - head[index]).abs() < 1e-12);
        }
        assert!(lookup[8..].iter().all(|value| value.to_bits() == 0));
        assert!(
            head.chunks_exact(4)
                .all(|row| row.iter().any(|value| value.abs() > 1e-12))
        );
    }

    #[test]
    fn tied_table_and_final_gain_match_all_coordinate_central_differences() {
        let table_values = [
            0.8, -0.2, 0.1, 0.4, -0.3, 0.7, 0.2, -0.1, 0.5, 0.1, -0.6, 0.3, -0.4, -0.2, 0.9, 0.1,
            0.2, 0.3, -0.1, 0.8,
        ];
        let gain_values = [1.0, 0.9, 1.1, 0.8];
        let model = zero_layer_model(tensor(&[5, 4], &table_values), tensor(&[4], &gain_values));
        model
            .loss(&[0, 1], &[1, 2], &[1, 2])
            .unwrap()
            .backward()
            .unwrap();
        let table_gradient = model.tied_embedding().tensor().gradient().unwrap();
        let gain_gradient = model.final_norm().gain().tensor().gradient().unwrap();

        let table_report = sampled_tensor_gradient_check(
            &mut tensor(&[5, 4], &table_values),
            &table_gradient.view(),
            STEP,
            TOLERANCE,
            table_values.len(),
            |candidate| {
                zero_layer_model(candidate.clone(), tensor(&[4], &gain_values))
                    .loss(&[0, 1], &[1, 2], &[1, 2])
                    .unwrap()
                    .value()
                    .as_slice()[0]
            },
        )
            .unwrap();
        assert!(table_report.passed);

        let gain_report = sampled_tensor_gradient_check(
            &mut tensor(&[4], &gain_values),
            &gain_gradient.view(),
            STEP,
            TOLERANCE,
            gain_values.len(),
            |candidate| {
                zero_layer_model(tensor(&[5, 4], &table_values), candidate.clone())
                    .loss(&[0, 1], &[1, 2], &[1, 2])
                    .unwrap()
                    .value()
                    .as_slice()[0]
            },
        )
            .unwrap();
        assert!(gain_report.passed);
    }

    #[test]
    fn multi_block_loss_reaches_every_parameter_with_finite_gradients() {
        let model = model(2, 23);
        model
            .loss(&[0, 1, 2], &[1, 3], &[1, 2, 3])
            .unwrap()
            .backward()
            .unwrap();
        assert!(model.parameters().iter().all(|parameter| {
            parameter
                .tensor()
                .gradient()
                .is_some_and(|gradient| gradient.as_slice().iter().all(|value| value.is_finite()))
        }));
    }

    #[test]
    fn configuration_component_input_and_target_errors_are_specific() {
        let mut rng = SplitMix64::from_seed(1);
        assert_eq!(
            DecoderModel::new(
                DecoderModelConfig::new(5, 4, 0, 4, 0, 4, 10_000.0, 1e-6),
                &mut rng,
            )
                .unwrap_err(),
            DecoderModelError::ZeroHeadCount
        );
        assert_eq!(
            DecoderModel::new(
                DecoderModelConfig::new(5, 6, 2, 4, 0, 4, 10_000.0, 1e-6),
                &mut rng,
            )
                .unwrap_err(),
            DecoderModelError::OddHeadWidth { head_width: 3 }
        );
        let model = model(0, 1);
        assert_eq!(
            model.forward(&[0], &[1]).unwrap_err(),
            DecoderModelError::TokenRank { rank: 1 }
        );
        assert_eq!(
            model.forward(&[], &[0, 1]).unwrap_err(),
            DecoderModelError::EmptyBatch
        );
        assert_eq!(
            model.forward(&[], &[1, 0]).unwrap_err(),
            DecoderModelError::EmptyTokens
        );
        assert_eq!(
            model.forward(&[0; 5], &[1, 5]).unwrap_err(),
            DecoderModelError::ContextLengthExceeded {
                tokens: 5,
                max_positions: 4,
            }
        );
        assert_eq!(
            model.forward(&[0], &[1, 2]).unwrap_err(),
            DecoderModelError::TokenCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            model.forward(&[0, 5], &[1, 2]).unwrap_err(),
            DecoderModelError::Embedding(EmbeddingError::TokenIdOutOfBounds {
                position: 1,
                id: 5,
                vocabulary_size: 5,
            })
        );
        assert_eq!(
            model.loss(&[0, 1], &[1, 2], &[1]).unwrap_err(),
            DecoderModelError::TargetCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            model.loss(&[0, 1], &[1, 2], &[1, 5]).unwrap_err(),
            DecoderModelError::TargetIdOutOfBounds {
                position: 1,
                id: 5,
                vocabulary_size: 5,
            }
        );
    }
}
