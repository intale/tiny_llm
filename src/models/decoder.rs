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
