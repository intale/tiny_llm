//! Model-wide KV state, prompt prefill, one-token decode, and cached generation.

use std::cell::Ref;
use std::error::Error;
use std::fmt;

use crate::attention::incremental::{IncrementalAttentionError, LayerKvCache, LayerKvCacheError};
use crate::autograd::tensor_core::{
    AutogradContext, TensorAutodiffError, TensorValue, TensorValueBinding,
};
use crate::models::decoder::{DecoderModel, DecoderModelConfig};
use crate::nn::embedding::EmbeddingError;
use crate::nn::init::SplitMix64;
use crate::nn::probability::log_softmax_forward;
use crate::nn::residual::{ResidualError, residual_add_with_context};
use crate::nn::rms_norm::RmsNormError;
use crate::nn::swiglu::SwiGluError;
use crate::tensor::storage::Tensor;

use super::sampling::{
    GenerationConfig, GenerationStop, SamplingError, SamplingMode, sample_next_token,
};

fn valid_token(token_id: u32, vocabulary_size: usize) -> bool {
    usize::try_from(token_id)
        .ok()
        .is_some_and(|token| token < vocabulary_size)
}

fn same_config(left: DecoderModelConfig, right: DecoderModelConfig) -> bool {
    left.vocabulary_size() == right.vocabulary_size()
        && left.model_width() == right.model_width()
        && left.heads() == right.heads()
        && left.feed_forward_width() == right.feed_forward_width()
        && left.layers() == right.layers()
        && left.max_positions() == right.max_positions()
        && left.rope_base().to_bits() == right.rope_base().to_bits()
        && left.rms_epsilon().to_bits() == right.rms_epsilon().to_bits()
}

fn checked_add(
    left: usize,
    right: usize,
    counter: DecoderKvCacheCounter,
) -> Result<usize, DecoderKvCacheError> {
    left.checked_add(right)
        .ok_or(DecoderKvCacheError::WorkOverflow { counter })
}

fn checked_mul(
    left: usize,
    right: usize,
    counter: DecoderKvCacheCounter,
) -> Result<usize, DecoderKvCacheError> {
    left.checked_mul(right)
        .ok_or(DecoderKvCacheError::WorkOverflow { counter })
}

/// A checked model-wide work counter that could not be represented as `usize`
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DecoderKvCacheCounter {
    TokenForwards,
    PrefillTokens,
    DecodeTokens,
    CacheAppends,
    QkvProjectionRows,
    AttentionScoreValues,
    CompletePrefixAttentionScoreValues,
}

impl fmt::Display for DecoderKvCacheCounter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TokenForwards => "token forwards",
            Self::PrefillTokens => "prefill tokens",
            Self::DecodeTokens => "decode tokens",
            Self::CacheAppends => "cache appends",
            Self::QkvProjectionRows => "Q/K/V projection rows",
            Self::AttentionScoreValues => "cached attention score values",
            Self::CompletePrefixAttentionScoreValues => "complete-prefix attention score values",
        })
    }
}

/// A graph-free cached-decoder stage that rejected a request or computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachedDecoderStage {
    TiedWeightTranspose,
    TiedVocabularyProjection,
}

impl fmt::Display for CachedDecoderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TiedWeightTranspose => "transpose tied embedding weight",
            Self::TiedVocabularyProjection => "project tied vocabulary logits",
        })
    }
}

/// A rejected model/cache pairing, phase transition, token, or decoder stage.
#[derive(Clone, Debug, PartialEq)]
pub enum DecoderKvCacheError {
    LayerAllocationFailed {
        layers: usize,
    },
    ParameterAllocationFailed {
        parameters: usize,
    },
    LayerCache {
        layer: usize,
        source: LayerKvCacheError,
    },
    ModelConfigMismatch,
    ModelParameterCountMismatch {
        cache: usize,
        model: usize,
    },
    ModelParameterMismatch {
        index: usize,
    },
    ModelParameterRevisionMismatch {
        index: usize,
        cache: u64,
        model: u64,
    },
    LayerCountMismatch {
        cache: usize,
        model: usize,
    },
    LayerBatchSizeMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    LayerCapacityMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    EmptyPrompt,
    PromptTooLong {
        tokens: usize,
        capacity: usize,
    },
    PromptTokenOutOfBounds {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    PrefillRequiresEmpty {
        len: usize,
    },
    DecodeRequiresPrefill,
    DecodeTokenOutOfBounds {
        token_id: u32,
        vocabulary_size: usize,
    },
    Full {
        capacity: usize,
    },
    LayerLengthInvariant {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    PreparedCacheChanged {
        layer: usize,
    },
    WorkOverflow {
        counter: DecoderKvCacheCounter,
    },
    Embedding(EmbeddingError),
    AttentionNorm {
        layer: usize,
        source: RmsNormError,
    },
    IncrementalAttention {
        layer: usize,
        source: IncrementalAttentionError,
    },
    AttentionResidual {
        layer: usize,
        source: ResidualError,
    },
    FeedForwardNorm {
        layer: usize,
        source: RmsNormError,
    },
    FeedForward {
        layer: usize,
        source: SwiGluError,
    },
    FeedForwardResidual {
        layer: usize,
        source: ResidualError,
    },
    FinalNorm(RmsNormError),
    Autodiff {
        stage: CachedDecoderStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for DecoderKvCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerAllocationFailed { layers } => {
                write!(formatter, "cannot allocate {layers} decoder layer caches")
            }
            Self::ParameterAllocationFailed { parameters } => write!(
                formatter,
                "cannot retain {parameters} decoder parameter bindings"
            ),
            Self::LayerCache { layer, source } => {
                write!(formatter, "decoder layer {layer} cache: {source}")
            }
            Self::ModelConfigMismatch => formatter
                .write_str("decoder cache configuration does not match this decoder model exactly"),
            Self::ModelParameterCountMismatch { cache, model } => write!(
                formatter,
                "decoder cache binds {cache} parameter nodes, but the model exposes {model}"
            ),
            Self::ModelParameterMismatch { index } => write!(
                formatter,
                "decoder cache parameter identity differs at stable index {index}"
            ),
            Self::ModelParameterRevisionMismatch {
                index,
                cache,
                model,
            } => write!(
                formatter,
                "decoder cache parameter revision {cache} at stable index {index} differs from model revision {model}"
            ),
            Self::LayerCountMismatch { cache, model } => write!(
                formatter,
                "decoder cache owns {cache} layer caches, but the model exposes {model} blocks"
            ),
            Self::LayerBatchSizeMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder layer {layer} cache batch size must be {expected}, got {actual}"
            ),
            Self::LayerCapacityMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder layer {layer} cache capacity must be {expected}, got {actual}"
            ),
            Self::EmptyPrompt => formatter.write_str("cached prefill needs a nonempty prompt"),
            Self::PromptTooLong { tokens, capacity } => write!(
                formatter,
                "cached prefill has {tokens} prompt tokens, exceeding capacity {capacity}"
            ),
            Self::PromptTokenOutOfBounds {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "cached prefill token {token_id} at position {position} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::PrefillRequiresEmpty { len } => write!(
                formatter,
                "cached prefill requires empty state, but the cache length is {len}"
            ),
            Self::DecodeRequiresPrefill => {
                formatter.write_str("cached decode requires one completed nonempty prompt prefill")
            }
            Self::DecodeTokenOutOfBounds {
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "cached decode token {token_id} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::Full { capacity } => {
                write!(formatter, "decoder KV cache is full at capacity {capacity}")
            }
            Self::LayerLengthInvariant {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder layer {layer} cache length must be {expected}, got {actual}"
            ),
            Self::PreparedCacheChanged { layer } => write!(
                formatter,
                "decoder layer {layer} cache changed after its row was prepared"
            ),
            Self::WorkOverflow { counter } => {
                write!(formatter, "decoder cache {counter} counter overflows")
            }
            Self::Embedding(source) => write!(formatter, "cached token embedding: {source}"),
            Self::AttentionNorm { layer, source } => {
                write!(
                    formatter,
                    "decoder layer {layer} attention RMSNorm: {source}"
                )
            }
            Self::IncrementalAttention { layer, source } => write!(
                formatter,
                "decoder layer {layer} incremental attention: {source}"
            ),
            Self::AttentionResidual { layer, source } => write!(
                formatter,
                "decoder layer {layer} attention residual merge: {source}"
            ),
            Self::FeedForwardNorm { layer, source } => write!(
                formatter,
                "decoder layer {layer} feed-forward RMSNorm: {source}"
            ),
            Self::FeedForward { layer, source } => {
                write!(formatter, "decoder layer {layer} SwiGLU: {source}")
            }
            Self::FeedForwardResidual { layer, source } => write!(
                formatter,
                "decoder layer {layer} feed-forward residual merge: {source}"
            ),
            Self::FinalNorm(source) => write!(formatter, "cached final RMSNorm: {source}"),
            Self::Autodiff { stage, source } => write!(formatter, "cached {stage}: {source}"),
        }
    }
}

impl Error for DecoderKvCacheError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LayerCache { source, .. } => Some(source),
            Self::Embedding(source) => Some(source),
            Self::AttentionNorm { source, .. } | Self::FeedForwardNorm { source, .. } => {
                Some(source)
            }
            Self::IncrementalAttention { source, .. } => Some(source),
            Self::AttentionResidual { source, .. } | Self::FeedForwardResidual { source, .. } => {
                Some(source)
            }
            Self::FeedForward { source, .. } => Some(source),
            Self::FinalNorm(source) => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Exact model work committed to one decoder KV state
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct DecoderKvCacheWork {
    token_forwards: usize,
    prefill_tokens: usize,
    decode_tokens: usize,
    cache_appends: usize,
    qkv_projection_rows: usize,
    attention_score_values: usize,
}

impl DecoderKvCacheWork {
    pub fn token_forwards(self) -> usize {
        self.token_forwards
    }

    pub fn prefill_tokens(self) -> usize {
        self.prefill_tokens
    }

    pub fn decode_tokens(self) -> usize {
        self.decode_tokens
    }

    pub fn cache_appends(self) -> usize {
        self.cache_appends
    }

    pub fn qkv_projection_rows(self) -> usize {
        self.qkv_projection_rows
    }

    pub fn attention_score_values(self) -> usize {
        self.attention_score_values
    }
}

/// The newest graph-free vocabulary logits after one committed cached row
#[derive(Clone, Debug)]
pub struct CachedDecoderOutput {
    logits: TensorValue,
    position: usize,
    cache_len: usize,
    attention_score_values: usize,
}

impl CachedDecoderOutput {
    /// Returns logits shaped `[1, 1, vocabulary_size]`.
    pub fn logits(&self) -> &TensorValue {
        &self.logits
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn attention_score_values(&self) -> usize {
        self.attention_score_values
    }
}

#[derive(Clone, Copy, Debug)]
enum CachedPhase {
    Prefill,
    Decode,
}

/// Reusable per-block cache storage plus coherent model-wide sequence state
#[derive(Clone, Debug)]
pub struct DecoderKvCache {
    config: DecoderModelConfig,
    parameter_bindings: Vec<TensorValueBinding>,
    layers: Vec<LayerKvCache>,
    len: usize,
    prefill_complete: bool,
    work: DecoderKvCacheWork,
}

impl PartialEq for DecoderKvCache {
    fn eq(&self, other: &Self) -> bool {
        same_config(self.config, other.config)
            && self.layers == other.layers
            && self.len == other.len
            && self.prefill_complete == other.prefill_complete
            && self.work == other.work
            && self.parameter_bindings.len() == other.parameter_bindings.len()
            && self
                .parameter_bindings
                .iter()
                .zip(&other.parameter_bindings)
                .all(|(left, right)| left.same_binding(right))
    }
}

impl DecoderKvCache {
    /// Allocates one fixed-capacity cache per block and captures compatibility evidence
    pub fn new(model: &DecoderModel) -> Result<Self, DecoderKvCacheError> {
        let config = model.config();
        let mut parameter_bindings = Vec::new();
        parameter_bindings
            .try_reserve_exact(model.parameters().len())
            .map_err(|_| DecoderKvCacheError::ParameterAllocationFailed {
                parameters: model.parameters().len(),
            })?;
        parameter_bindings.extend(
            model
                .parameters()
                .iter()
                .map(|parameter| TensorValueBinding::capture(parameter.tensor())),
        );

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(model.blocks().len())
            .map_err(|_| DecoderKvCacheError::LayerAllocationFailed {
                layers: model.blocks().len(),
            })?;
        for (layer, block) in model.blocks().iter().enumerate() {
            layers.push(
                LayerKvCache::new(block.attention(), 1, config.max_positions())
                    .map_err(|source| DecoderKvCacheError::LayerCache { layer, source })?,
            )
        }
        Ok(Self {
            config,
            parameter_bindings,
            layers,
            len: 0,
            prefill_complete: false,
            work: DecoderKvCacheWork::default(),
        })
    }

    /// Validates one exact pairing and retains read borrows of the model's parameter values
    pub fn bind<'model, 'cache>(
        &'cache mut self,
        model: &'model DecoderModel,
    ) -> Result<DecoderKvSession<'model, 'cache>, DecoderKvCacheError> {
        self.validate_binding(model)?;
        let mut parameter_value_guards: Vec<Ref<'model, Tensor>> = Vec::new();
        parameter_value_guards
            .try_reserve_exact(model.parameters().len())
            .map_err(|_| DecoderKvCacheError::ParameterAllocationFailed {
                parameters: model.parameters().len(),
            })?;
        parameter_value_guards.extend(
            model
                .parameters()
                .iter()
                .map(|parameter| parameter.tensor().value()),
        );

        Ok(DecoderKvSession {
            model,
            cache: self,
            _parameter_value_guards: parameter_value_guards,
        })
    }

    fn next_work(
        &self,
        phase: CachedPhase,
        score_values: usize,
    ) -> Result<DecoderKvCacheWork, DecoderKvCacheError> {
        let layers = self.layers.len();
        let qkv_rows = checked_mul(3, layers, DecoderKvCacheCounter::QkvProjectionRows)?;

        Ok(DecoderKvCacheWork {
            token_forwards: checked_add(
                self.work.token_forwards,
                1,
                DecoderKvCacheCounter::TokenForwards,
            )?,
            prefill_tokens: checked_add(
                self.work.prefill_tokens,
                usize::from(matches!(phase, CachedPhase::Prefill)),
                DecoderKvCacheCounter::PrefillTokens,
            )?,
            decode_tokens: checked_add(
                self.work.decode_tokens,
                usize::from(matches!(phase, CachedPhase::Decode)),
                DecoderKvCacheCounter::DecodeTokens,
            )?,
            cache_appends: checked_add(
                self.work.cache_appends,
                layers,
                DecoderKvCacheCounter::CacheAppends,
            )?,
            qkv_projection_rows: checked_add(
                self.work.qkv_projection_rows,
                qkv_rows,
                DecoderKvCacheCounter::QkvProjectionRows,
            )?,
            attention_score_values: checked_add(
                self.work.attention_score_values,
                score_values,
                DecoderKvCacheCounter::AttentionScoreValues,
            )?,
        })
    }

    fn validate_binding(&self, model: &DecoderModel) -> Result<(), DecoderKvCacheError> {
        if !same_config(self.config, model.config()) {
            return Err(DecoderKvCacheError::ModelConfigMismatch);
        }
        if self.parameter_bindings.len() != model.parameters().len() {
            return Err(DecoderKvCacheError::ModelParameterCountMismatch {
                cache: self.parameter_bindings.len(),
                model: model.parameters().len(),
            });
        }
        if let Some(index) = self
            .parameter_bindings
            .iter()
            .zip(model.parameters())
            .position(|(cached, parameter)| !cached.node_matches(parameter.tensor()))
        {
            return Err(DecoderKvCacheError::ModelParameterMismatch { index });
        }
        if let Some((index, (cached, parameter))) = self
            .parameter_bindings
            .iter()
            .zip(model.parameters())
            .enumerate()
            .find(|(_, (cached, parameter))| !cached.revision_matches(parameter.tensor()))
        {
            return Err(DecoderKvCacheError::ModelParameterRevisionMismatch {
                index,
                cache: cached.revision(),
                model: parameter.tensor().value_revision(),
            });
        }
        if self.layers.len() != model.blocks().len() {
            return Err(DecoderKvCacheError::LayerCountMismatch {
                cache: self.layers.len(),
                model: model.blocks().len(),
            });
        }
        self.validate_layer_lengths()?;
        for (layer, (block, cache)) in model.blocks().iter().zip(&self.layers).enumerate() {
            if cache.batch_size() != 1 {
                return Err(DecoderKvCacheError::LayerBatchSizeMismatch {
                    layer,
                    expected: 1,
                    actual: cache.batch_size(),
                });
            }
            if cache.capacity() != self.capacity() {
                return Err(DecoderKvCacheError::LayerCapacityMismatch {
                    layer,
                    expected: self.capacity(),
                    actual: cache.capacity(),
                });
            }
            block
                .attention()
                .validate_incremental_cache_binding(cache)
                .map_err(|source| DecoderKvCacheError::IncrementalAttention { layer, source })?;
        }
        Ok(())
    }

    fn validate_layer_lengths(&self) -> Result<(), DecoderKvCacheError> {
        for (layer, cache) in self.layers.iter().enumerate() {
            if cache.len() != self.len {
                return Err(DecoderKvCacheError::LayerLengthInvariant {
                    layer,
                    expected: self.len,
                    actual: cache.len(),
                });
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        for cache in &mut self.layers {
            cache.reset()
        }
        self.len = 0;
        self.prefill_complete = false;
        self.work = DecoderKvCacheWork::default();
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.config.max_positions()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_len(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(LayerKvCache::len)
    }

    pub fn layer_cache(&self, layer: usize) -> Option<&LayerKvCache> {
        self.layers.get(layer)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    pub fn work(&self) -> DecoderKvCacheWork {
        self.work
    }
}

/// One decoder and one mutable model-wide cache bound for one checked session
///
/// The retained parameter-value borrows prevent an optimizer from changing the decoder while any
/// K/V rows from that decoder may still be used
pub struct DecoderKvSession<'model, 'cache> {
    model: &'model DecoderModel,
    cache: &'cache mut DecoderKvCache,
    _parameter_value_guards: Vec<Ref<'model, Tensor>>,
}

impl DecoderKvSession<'_, '_> {
    /// Fills every block cache from one validated, nonempty prompt
    ///
    /// Prompt rows advance serially through the shared one-row path. A later internal failure
    /// restores empty logical state while retaining allocation and the session's exact-model
    /// binding
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<CachedDecoderOutput, DecoderKvCacheError> {
        if prompt.is_empty() {
            return Err(DecoderKvCacheError::EmptyPrompt);
        }
        if self.cache.len != 0 || self.cache.prefill_complete {
            return Err(DecoderKvCacheError::PrefillRequiresEmpty {
                len: self.cache.len,
            });
        }
        if prompt.len() > self.cache.capacity() {
            return Err(DecoderKvCacheError::PromptTooLong {
                tokens: prompt.len(),
                capacity: self.cache.capacity(),
            });
        }
        for (position, &token_id) in prompt.iter().enumerate() {
            if !valid_token(token_id, self.cache.config.vocabulary_size()) {
                return Err(DecoderKvCacheError::PromptTokenOutOfBounds {
                    position,
                    token_id,
                    vocabulary_size: self.cache.config.vocabulary_size(),
                });
            }
        }

        let mut final_output = None;
        for &token_id in prompt {
            match self.forward_token(token_id, CachedPhase::Prefill) {
                Ok(output) => final_output = Some(output),
                Err(error) => {
                    self.cache.reset();
                    return Err(error);
                }
            }
        }
        self.cache.prefill_complete = true;
        Ok(final_output.expect("a validated prompt has at least one token"))
    }

    /// Appends one selected token and returns logits for the following choice
    pub fn decode(&mut self, token_id: u32) -> Result<CachedDecoderOutput, DecoderKvCacheError> {
        if !self.cache.prefill_complete || self.cache.len == 0 {
            return Err(DecoderKvCacheError::DecodeRequiresPrefill);
        }
        if !valid_token(token_id, self.cache.config.vocabulary_size()) {
            return Err(DecoderKvCacheError::DecodeTokenOutOfBounds {
                token_id,
                vocabulary_size: self.cache.config.vocabulary_size(),
            });
        }
        if self.cache.is_full() {
            return Err(DecoderKvCacheError::Full {
                capacity: self.cache.capacity(),
            });
        }
        self.forward_token(token_id, CachedPhase::Decode)
    }

    fn forward_token(
        &mut self,
        token_id: u32,
        phase: CachedPhase,
    ) -> Result<CachedDecoderOutput, DecoderKvCacheError> {
        let position = self.cache.len;
        let layer_count = self.cache.layers.len();
        let context = AutogradContext::no_grad();
        let (logits, prepared, score_values) = {
            let embedding = self.model.embedding().forward_with_context(context, &[token_id], &[1, 1])
                .map_err(DecoderKvCacheError::Embedding)?;
            let mut current = embedding;
            let mut prepared = Vec::new();
            prepared.try_reserve_exact(layer_count).map_err(|_| {
                DecoderKvCacheError::LayerAllocationFailed {
                    layers: layer_count,
                }
            })?;
            let mut score_values = 0_usize;
            for (layer, (block, cache)) in self
                .model
                .blocks()
                .iter()
                .zip(&self.cache.layers)
                .enumerate() {
                let attention_norm = block
                    .attention_norm()
                    .forward_with_context(context, &current)
                    .map_err(|source| DecoderKvCacheError::AttentionNorm { layer, source })?;
                let ticket = block
                    .attention()
                    .prepare_incremental_bound_with_context(context, &attention_norm, cache)
                    .map_err(|source| DecoderKvCacheError::IncrementalAttention {
                        layer,
                        source
                    })?;
                score_values = checked_add(score_values, ticket.attention_score_values(), DecoderKvCacheCounter::AttentionScoreValues)?;
                let after_attention = residual_add_with_context(context, &current, ticket.output())
                    .map_err(|source| DecoderKvCacheError::AttentionResidual { layer, source })?;
                let feed_forward_norm = block.feed_forward_norm().forward_with_context(context, &after_attention)
                    .map_err(|source| DecoderKvCacheError::FeedForwardNorm { layer, source })?;
                let feed_forward = block.feed_forward().forward_with_context(context, &feed_forward_norm)
                    .map_err(|source| DecoderKvCacheError::FeedForward { layer, source })?;
                current = residual_add_with_context(context, &after_attention, &feed_forward)
                    .map_err(|source| DecoderKvCacheError::FeedForwardResidual { layer, source })?;
                prepared.push(ticket);
            }
            let final_norm = self.model.final_norm().forward_with_context(context, &current)
                .map_err(DecoderKvCacheError::FinalNorm)?;
            let tied_weight = self.model.tied_embedding().tensor().transpose_with_context(context, 0, 1).map_err(|source| DecoderKvCacheError::Autodiff {
                stage: CachedDecoderStage::TiedWeightTranspose,
                source,
            })?;
            let logits = final_norm.matmul_with_context(context, &tied_weight).map_err(|source| DecoderKvCacheError::Autodiff {
                stage: CachedDecoderStage::TiedVocabularyProjection,
                source,
            })?;
            Ok::<_, DecoderKvCacheError>((logits, prepared, score_values))
        }?;

        for (layer, (ticket, cache)) in prepared.iter().zip(&self.cache.layers).enumerate() {
            if ticket.cache_len() != position + 1 || !ticket.matches_cache(cache) {
                return Err(DecoderKvCacheError::PreparedCacheChanged { layer })
            }
        }
        let next_len = checked_add(position, 1, DecoderKvCacheCounter::TokenForwards)?;
        let next_work = self.cache.next_work(phase, score_values)?;

        for (ticket, cache) in prepared.into_iter().zip(&mut self.cache.layers) {
            ticket.commit(cache);
        }
        self.cache.len = next_len;
        self.cache.work = next_work;

        Ok(CachedDecoderOutput {
            logits,
            position,
            cache_len: next_len,
            attention_score_values: score_values
        })
    }

    /// Clears sequence state while retaining this session's exact model binding.
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// Exposes the bound cache for deterministic state and work inspection.
    pub fn cache(&self) -> &DecoderKvCache {
        self.cache
    }
}

/// One selected token and the categorical evidence used by cached generation
#[derive(Clone, Debug, PartialEq)]
pub struct CachedGenerationStep {
    prefix_length: usize,
    token_id: u32,
    unit_draw: Option<f64>,
    interval_start: f64,
    interval_end: f64
}

impl CachedGenerationStep {
    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    pub fn unit_draw(&self) -> Option<f64> {
        self.unit_draw
    }

    pub fn interval_start(&self) -> f64 {
        self.interval_start
    }

    pub fn interval_end(&self) -> f64 {
        self.interval_end
    }
}

/// Exact cached work and its dense complete-prefix attention-score baseline
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CachedGenerationWork {
    prefill_tokens: usize,
    decode_tokens: usize,
    layer_cache_count: usize,
    cache_appends: usize,
    qkv_projection_rows: usize,
    cached_attention_score_values: usize,
    complete_prefix_attention_score_values: usize
}

impl CachedGenerationWork {
    pub fn prefill_tokens(self) -> usize {
        self.prefill_tokens
    }

    pub fn decode_tokens(self) -> usize {
        self.decode_tokens
    }

    pub fn layer_cache_count(self) -> usize {
        self.layer_cache_count
    }

    pub fn cache_appends(self) -> usize {
        self.cache_appends
    }

    pub fn qkv_projection_rows(self) -> usize {
        self.qkv_projection_rows
    }

    pub fn cached_attention_score_values(self) -> usize {
        self.cached_attention_score_values
    }

    pub fn complete_prefix_attention_score_values(self) -> usize {
        self.complete_prefix_attention_score_values
    }
}

/// Cached tokens, stops, final logical state, and exact work evidence
#[derive(Clone, Debug, PartialEq)]
pub struct CachedGenerationResult {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    steps: Vec<CachedGenerationStep>,
    stop: GenerationStop,
    final_cache_len: usize,
    work: CachedGenerationWork
}

impl CachedGenerationResult {
    pub fn prompt(&self) -> &[u32] {
        &self.prompt
    }

    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub fn steps(&self) -> &[CachedGenerationStep] {
        &self.steps
    }

    pub fn stop(&self) -> GenerationStop {
        self.stop
    }

    pub fn final_cache_len(&self) -> usize {
        self.final_cache_len
    }

    pub fn work(&self) -> CachedGenerationWork {
        self.work
    }
}

/// A cached-generation request, model step, sampling step, or work count failed.
#[derive(Debug, PartialEq)]
pub enum CachedGenerationError {
    Cache(DecoderKvCacheError),
    Sampling(SamplingError),
    EmptyPrompt,
    PromptTooLong {
        tokens: usize,
        max_positions: usize,
    },
    PromptTokenOutOfBounds {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    EosTokenOutOfBounds {
        token_id: u32,
        vocabulary_size: usize,
    },
    LogitCountMismatch {
        expected: usize,
        actual: usize,
    },
    AllocationFailed {
        values: usize,
    },
    WorkOverflow {
        counter: DecoderKvCacheCounter,
    },
}

impl fmt::Display for CachedGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(source) => source.fmt(formatter),
            Self::Sampling(source) => source.fmt(formatter),
            Self::EmptyPrompt => formatter.write_str("cached generation needs a nonempty prompt"),
            Self::PromptTooLong {
                tokens,
                max_positions,
            } => write!(
                formatter,
                "cached generation prompt has {tokens} tokens, exceeding context capacity {max_positions}"
            ),
            Self::PromptTokenOutOfBounds {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "cached generation prompt token {token_id} at position {position} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::EosTokenOutOfBounds {
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "cached generation EOS token {token_id} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::LogitCountMismatch { expected, actual } => write!(
                formatter,
                "cached last-position logits need {expected} values, received {actual}"
            ),
            Self::AllocationFailed { values } => write!(
                formatter,
                "cannot allocate cached generation evidence for {values} values"
            ),
            Self::WorkOverflow { counter } => {
                write!(formatter, "cached generation {counter} counter overflows")
            }
        }
    }
}

impl Error for CachedGenerationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cache(source) => Some(source),
            Self::Sampling(source) => Some(source),
            _ => None,
        }
    }
}

impl From<DecoderKvCacheError> for CachedGenerationError {
    fn from(source: DecoderKvCacheError) -> Self {
        Self::Cache(source)
    }
}

impl From<SamplingError> for CachedGenerationError {
    fn from(source: SamplingError) -> Self {
        Self::Sampling(source)
    }
}

fn validate_generation_request(
    vocabulary_size: usize,
    max_positions: usize,
    prompt: &[u32],
    config: GenerationConfig,
) -> Result<(), CachedGenerationError> {
    if prompt.is_empty() {
        return Err(CachedGenerationError::EmptyPrompt);
    }
    if prompt.len() > max_positions {
        return Err(CachedGenerationError::PromptTooLong {
            tokens: prompt.len(),
            max_positions,
        });
    }
    for (position, &token_id) in prompt.iter().enumerate() {
        if !valid_token(token_id, vocabulary_size) {
            return Err(CachedGenerationError::PromptTokenOutOfBounds {
                position,
                token_id,
                vocabulary_size,
            });
        }
    }
    if let Some(token_id) = config.eos_token()
        && !valid_token(token_id, vocabulary_size)
    {
        return Err(CachedGenerationError::EosTokenOutOfBounds {
            token_id,
            vocabulary_size,
        });
    }
    if let SamplingMode::TemperatureTopK { temperature, top_k } = config.mode() {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature { value: temperature }.into());
        }
        if top_k == 0 || top_k > vocabulary_size {
            return Err(SamplingError::InvalidTopK {
                top_k,
                vocabulary_size,
            }
                .into());
        }
    }
    Ok(())
}

fn checked_complete_prefix_scores(
    current: usize,
    layers: usize,
    heads: usize,
    prefix_length: usize,
) -> Result<usize, CachedGenerationError> {
    let counter = DecoderKvCacheCounter::CompletePrefixAttentionScoreValues;
    let square = prefix_length
        .checked_mul(prefix_length)
        .ok_or(CachedGenerationError::WorkOverflow { counter })?;
    let values = layers
        .checked_mul(heads)
        .and_then(|factor| factor.checked_mul(square))
        .ok_or(CachedGenerationError::WorkOverflow { counter })?;
    current
        .checked_add(values)
        .ok_or(CachedGenerationError::WorkOverflow { counter })
}

pub fn generate_cached(model: &DecoderModel, prompt: &[u32], config: GenerationConfig, rng: &mut SplitMix64) -> Result<CachedGenerationResult, CachedGenerationError> {
    let model_config = model.config();
    let vocabulary_size = model_config.vocabulary_size();
    let max_positions = model_config.max_positions();
    validate_generation_request(vocabulary_size, max_positions, prompt, config)?;

    let planned_steps = config.max_new_tokens().min(
        max_positions.checked_sub(prompt.len())
            .and_then(|remaining| remaining.checked_add(1))
            .ok_or(CachedGenerationError::AllocationFailed { values: usize::MAX })?
    );
    let mut prompt_copy = Vec::new();
    prompt_copy.try_reserve_exact(prompt.len()).map_err(|_| {
        CachedGenerationError::AllocationFailed {
            values: prompt.len(),
        }
    })?;
    prompt_copy.extend_from_slice(prompt);
    let mut generated = Vec::new();
    generated.try_reserve_exact(planned_steps).map_err(|_| {
        CachedGenerationError::AllocationFailed {
            values: planned_steps,
        }
    })?;
    let mut steps = Vec::new();
    steps.try_reserve_exact(planned_steps).map_err(|_| {
        CachedGenerationError::AllocationFailed {
            values: planned_steps,
        }
    })?;

    if config.max_new_tokens() == 0 {
        return Ok(CachedGenerationResult {
            prompt: prompt_copy,
            generated,
            steps,
            stop: GenerationStop::TokenLimit,
            final_cache_len: 0,
            work: CachedGenerationWork {
                layer_cache_count: model_config.layers(),
                ..CachedGenerationWork::default()
            },
        });
    }

    let mut cache = DecoderKvCache::new(model)?;
    let mut session = cache.bind(model)?;
    let mut current = session.prefill(prompt)?;
    let mut prefix_length = prompt.len();
    let mut complete_prefix_attention_score_values = 0_usize;

    let stop = loop {
        complete_prefix_attention_score_values = checked_complete_prefix_scores(
            complete_prefix_attention_score_values,
            model_config.layers(),
            model_config.heads(),
            prefix_length
        )?;
        let decision = {
            let logits = current.logits().value();
            if logits.len() != vocabulary_size {
                return Err(CachedGenerationError::LogitCountMismatch {
                    expected: vocabulary_size,
                    actual: logits.len(),
                });
            }
            sample_next_token(logits.as_slice(), config.mode(), rng)?
        };
        let token_id = decision.token_id();
        generated.push(token_id);
        steps.push(CachedGenerationStep {
            prefix_length,
            token_id,
            unit_draw: decision.unit_draw(),
            interval_start: decision.interval_start(),
            interval_end: decision.interval_end()
        });
        prefix_length = prefix_length.checked_add(1).ok_or(CachedGenerationError::WorkOverflow {
            counter: DecoderKvCacheCounter::TokenForwards,
        })?;

        if config.eos_token() == Some(token_id) {
            break GenerationStop::Eos;
        }
        if generated.len() == config.max_new_tokens() {
            break GenerationStop::TokenLimit;
        }
        if prefix_length > max_positions {
            break GenerationStop::ContextLimit;
        }
        current = session.decode(token_id)?;
    };

    let cache_work = session.cache().work();
    Ok(CachedGenerationResult {
        prompt: prompt_copy,
        generated,
        steps,
        stop,
        final_cache_len: session.cache().len(),
        work: CachedGenerationWork {
            prefill_tokens: cache_work.prefill_tokens(),
            decode_tokens: cache_work.decode_tokens(),
            layer_cache_count: session.cache().layer_count(),
            cache_appends: cache_work.cache_appends(),
            qkv_projection_rows: cache_work.qkv_projection_rows(),
            cached_attention_score_values: cache_work.attention_score_values(),
            complete_prefix_attention_score_values,
        },
    })
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::autograd::tensor_core::GraphRetention;
    use crate::generation::sampling::generate_uncached;
    use crate::models::decoder::DecoderModel;
    use crate::nn::init::NamedParameter;
    use crate::tensor::storage::Tensor;
    use crate::training::adam_w::{AdamW, AdamWConfig};

    const TOLERANCE: f64 = 2e-12;

    fn config(layers: usize, max_positions: usize, rms_epsilon: f64) -> DecoderModelConfig {
        DecoderModelConfig::new(5, 4, 2, 4, layers, max_positions, 10_000.0, rms_epsilon)
    }

    fn model(layers: usize, max_positions: usize, seed: u64) -> DecoderModel {
        DecoderModel::new(
            config(layers, max_positions, 1e-6),
            &mut SplitMix64::from_seed(seed),
        )
            .unwrap()
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

    fn final_logits(model: &DecoderModel, prefix: &[u32]) -> Vec<f64> {
        let logits = model
            .forward_with_context(AutogradContext::no_grad(), prefix, &[1, prefix.len()])
            .unwrap()
            .logits()
            .value_snapshot();
        logits.as_slice()[logits.len() - model.config().vocabulary_size()..].to_vec()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= TOLERANCE,
                "value {index}: expected {expected:.16}, got {actual:.16}"
            );
        }
    }

    #[test]
    fn two_layer_prefill_and_decode_match_complete_prefix_logits() {
        let model = model(2, 4, 38);
        let mut cache = DecoderKvCache::new(&model).unwrap();
        let mut session = cache.bind(&model).unwrap();
        assert_eq!(
            session._parameter_value_guards.len(),
            model.parameters().len()
        );
        let prefill = session.prefill(&[0, 1]).unwrap();
        assert_close(
            prefill.logits().value().as_slice(),
            &final_logits(&model, &[0, 1]),
        );
        assert_eq!(prefill.position(), 1);
        assert_eq!(prefill.cache_len(), 2);
        let decoded = session.decode(2).unwrap();
        assert_close(
            decoded.logits().value().as_slice(),
            &final_logits(&model, &[0, 1, 2]),
        );
        assert_eq!(decoded.position(), 2);
        assert_eq!(session.cache().len(), 3);
        assert_eq!(session.cache().layer_len(0), Some(3));
        assert_eq!(session.cache().layer_len(1), Some(3));
        assert_ne!(
            session
                .cache()
                .layer_cache(0)
                .unwrap()
                .key_storage()
                .as_ptr(),
            session
                .cache()
                .layer_cache(1)
                .unwrap()
                .key_storage()
                .as_ptr()
        );
        assert_eq!(session.cache().work().token_forwards(), 3);
        assert_eq!(session.cache().work().prefill_tokens(), 2);
        assert_eq!(session.cache().work().decode_tokens(), 1);
        assert_eq!(session.cache().work().cache_appends(), 6);
        assert_eq!(session.cache().work().qkv_projection_rows(), 18);
        assert_eq!(session.cache().work().attention_score_values(), 24);
        assert!(!decoded.logits().tracks_gradient());
    }

    #[test]
    fn reset_reuses_allocations_and_model_identity_is_exact() {
        let model = model(2, 4, 39);
        let shared = model.clone();
        let rebuilt =
            DecoderModel::from_parameters(model.config(), copied_parameters(&model)).unwrap();
        let changed_epsilon =
            DecoderModel::from_parameters(config(2, 4, 2e-6), model.parameters().to_vec()).unwrap();
        let mut cache = DecoderKvCache::new(&model).unwrap();
        cache.bind(&shared).unwrap().prefill(&[0, 1]).unwrap();
        let pointers = cache
            .layers
            .iter()
            .map(|layer| (layer.key_storage().as_ptr(), layer.value_storage().as_ptr()))
            .collect::<Vec<_>>();
        let before = cache.clone();
        assert!(matches!(
            cache.bind(&rebuilt),
            Err(DecoderKvCacheError::ModelParameterMismatch { .. })
        ));
        assert_eq!(cache, before);
        assert!(matches!(
            cache.bind(&changed_epsilon),
            Err(DecoderKvCacheError::ModelConfigMismatch)
        ));
        assert_eq!(cache, before);
        cache.bind(&model).unwrap().reset();
        assert!(cache.is_empty());
        assert_eq!(cache.work(), DecoderKvCacheWork::default());
        assert!(cache.layers.iter().zip(pointers).all(|(layer, pointers)| {
            layer.key_storage().as_ptr() == pointers.0
                && layer.value_storage().as_ptr() == pointers.1
        }));
        let replay = cache.bind(&model).unwrap().prefill(&[0, 1]).unwrap();
        assert_close(
            replay.logits().value().as_slice(),
            &final_logits(&model, &[0, 1]),
        );
    }

    #[test]
    fn reset_does_not_rebind_a_cache_after_an_in_place_model_update() {
        let model = model(2, 4, 239);
        let mut cache = DecoderKvCache::new(&model).unwrap();
        {
            let mut session = cache.bind(&model).unwrap();
            session.prefill(&[0, 1]).unwrap();
            session.reset();
        }
        let before = cache.clone();
        let mut optimizer = AdamW::new(AdamWConfig::new(0.01, 0.9, 0.999, 1e-8, 0.1).unwrap());

        optimizer.step(model.parameters()).unwrap();

        assert!(matches!(
            cache.bind(&model),
            Err(DecoderKvCacheError::ModelParameterRevisionMismatch {
                index: 0,
                cache: 0,
                model: 1,
            })
        ));
        assert_eq!(cache, before);

        let mut fresh_cache = DecoderKvCache::new(&model).unwrap();
        fresh_cache.bind(&model).unwrap().prefill(&[0, 1]).unwrap();
        assert_eq!(fresh_cache.len(), 2);
    }

    #[test]
    fn live_session_prevents_parameter_updates_until_drop() {
        let model = model(2, 4, 240);
        let mut cache = DecoderKvCache::new(&model).unwrap();
        let mut session = cache.bind(&model).unwrap();
        let mut optimizer = AdamW::new(AdamWConfig::new(0.01, 0.9, 0.999, 1e-8, 0.1).unwrap());
        session.prefill(&[0, 1]).unwrap();
        let values_before = model
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor().value_snapshot())
            .collect::<Vec<_>>();
        let revisions_before = model
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor().value_revision())
            .collect::<Vec<_>>();
        let optimizer_before = optimizer.clone();
        let cache_before = session.cache().clone();

        for parameter in model.parameters() {
            assert!(
                parameter.tensor().try_value_write().is_err(),
                "session did not retain the value borrow for {}",
                parameter.name()
            );
        }
        assert!(matches!(
            optimizer.step(model.parameters()),
            Err(crate::training::adam_w::AdamWError::ParameterValueBorrowed { ref name })
                if name == model.parameters()[0].name()
        ));
        assert_eq!(
            session._parameter_value_guards.len(),
            model.parameters().len()
        );
        assert_eq!(optimizer, optimizer_before);
        assert_eq!(session.cache(), &cache_before);
        assert_eq!(
            model
                .parameters()
                .iter()
                .map(|parameter| parameter.tensor().value_snapshot())
                .collect::<Vec<_>>(),
            values_before
        );
        assert_eq!(
            model
                .parameters()
                .iter()
                .map(|parameter| parameter.tensor().value_revision())
                .collect::<Vec<_>>(),
            revisions_before
        );
        session.decode(2).unwrap();

        drop(session);
        optimizer.step(model.parameters()).unwrap();
        assert!(matches!(
            cache.bind(&model),
            Err(DecoderKvCacheError::ModelParameterRevisionMismatch {
                index: 0,
                cache: 0,
                model: 1,
            })
        ));
        let mut fresh_cache = DecoderKvCache::new(&model).unwrap();
        fresh_cache.bind(&model).unwrap().prefill(&[0]).unwrap();
    }

    #[test]
    fn embedding_feed_forward_and_final_norm_identity_are_independently_bound() {
        let model = model(2, 4, 139);
        let mut cache = DecoderKvCache::new(&model).unwrap();
        cache.bind(&model).unwrap().prefill(&[0]).unwrap();
        let before = cache.clone();
        let feed_forward = model
            .parameters()
            .iter()
            .position(|parameter| parameter.name() == "blocks.0.ffn.gate.weight")
            .unwrap();
        let final_norm = model.parameters().len() - 1;

        for index in [0, feed_forward, final_norm] {
            let mut parameters = model.parameters().to_vec();
            let name = parameters[index].name().to_owned();
            let value = parameters[index].tensor().value_snapshot();
            parameters[index] = NamedParameter::from_tensor(name, value).unwrap();
            let rebuilt = DecoderModel::from_parameters(model.config(), parameters).unwrap();
            assert!(matches!(
                cache.bind(&rebuilt),
                Err(DecoderKvCacheError::ModelParameterMismatch {
                    index: actual
                }) if actual == index
            ));
            assert_eq!(cache, before);
        }
    }

    #[test]
    fn request_errors_and_late_prompt_error_preserve_logical_state() {
        let model = model(2, 3, 40);
        let mut cache = DecoderKvCache::new(&model).unwrap();
        let mut session = cache.bind(&model).unwrap();
        assert!(matches!(
            session.decode(5),
            Err(DecoderKvCacheError::DecodeRequiresPrefill)
        ));
        assert!(matches!(
            session.prefill(&[]),
            Err(DecoderKvCacheError::EmptyPrompt)
        ));
        assert!(matches!(
            session.prefill(&[5, 0, 0, 0]),
            Err(DecoderKvCacheError::PromptTooLong {
                tokens: 4,
                capacity: 3,
            })
        ));
        assert!(matches!(
            session.prefill(&[0, 5]),
            Err(DecoderKvCacheError::PromptTokenOutOfBounds { position: 1, .. })
        ));
        assert!(session.cache().is_empty());
        assert_eq!(session.cache().work(), DecoderKvCacheWork::default());
        session.prefill(&[0, 1, 2]).unwrap();
        let before = session.cache().clone();
        assert!(matches!(
            session.decode(5),
            Err(DecoderKvCacheError::DecodeTokenOutOfBounds { token_id: 5, .. })
        ));
        assert_eq!(session.cache(), &before);
        assert!(matches!(
            session.decode(3),
            Err(DecoderKvCacheError::Full { capacity: 3 })
        ));
        assert_eq!(session.cache(), &before);
        assert!(matches!(
            session.prefill(&[]),
            Err(DecoderKvCacheError::EmptyPrompt)
        ));
        assert_eq!(session.cache(), &before);
        assert!(matches!(
            session.prefill(&[5, 0, 0, 0]),
            Err(DecoderKvCacheError::PrefillRequiresEmpty { len: 3 })
        ));
        assert_eq!(session.cache(), &before);
        assert!(matches!(
            session.prefill(&[0]),
            Err(DecoderKvCacheError::PrefillRequiresEmpty { len: 3 })
        ));
        assert_eq!(session.cache(), &before);
    }

    #[test]
    fn bind_rejects_an_incompatible_layer_cache_before_state_operations() {
        let decoder = model(2, 3, 140);
        let other = model(2, 3, 141);
        let mut cache = DecoderKvCache::new(&decoder).unwrap();
        cache.layers[1] = LayerKvCache::new(other.blocks()[1].attention(), 1, 3).unwrap();
        let pointers = cache
            .layers
            .iter()
            .map(|layer| (layer.key_storage().as_ptr(), layer.value_storage().as_ptr()))
            .collect::<Vec<_>>();
        let before = cache.clone();

        assert!(matches!(
            cache.bind(&decoder),
            Err(DecoderKvCacheError::IncrementalAttention {
                layer: 1,
                source: IncrementalAttentionError::CacheLayerMismatch,
            })
        ));
        assert_eq!(cache, before);
        assert!(cache.is_empty());
        assert_eq!(cache.work(), DecoderKvCacheWork::default());
        assert!(cache.layers.iter().all(LayerKvCache::is_empty));
        assert!(cache.layers.iter().zip(pointers).all(|(layer, pointers)| {
            layer.key_storage().as_ptr() == pointers.0
                && layer.value_storage().as_ptr() == pointers.1
        }));
    }

    #[test]
    fn bind_checks_model_wide_state_and_each_layer_before_retaining_values() {
        let decoder = model(2, 4, 241);

        let mut parameter_count = DecoderKvCache::new(&decoder).unwrap();
        parameter_count.parameter_bindings.pop();
        parameter_count.layers.pop();
        let before = parameter_count.clone();
        assert!(matches!(
            parameter_count.bind(&decoder),
            Err(DecoderKvCacheError::ModelParameterCountMismatch { .. })
        ));
        assert_eq!(parameter_count, before);

        let mut layer_count = DecoderKvCache::new(&decoder).unwrap();
        layer_count.layers.pop();
        let before = layer_count.clone();
        assert!(matches!(
            layer_count.bind(&decoder),
            Err(DecoderKvCacheError::LayerCountMismatch { cache: 1, model: 2 })
        ));
        assert_eq!(layer_count, before);

        let mut layer_length = DecoderKvCache::new(&decoder).unwrap();
        layer_length.bind(&decoder).unwrap().prefill(&[0]).unwrap();
        layer_length.layers[0].reset();
        let before = layer_length.clone();
        assert!(matches!(
            layer_length.bind(&decoder),
            Err(DecoderKvCacheError::LayerLengthInvariant {
                layer: 0,
                expected: 1,
                actual: 0,
            })
        ));
        assert_eq!(layer_length, before);

        let mut batch = DecoderKvCache::new(&decoder).unwrap();
        batch.layers[0] = LayerKvCache::new(decoder.blocks()[0].attention(), 2, 4).unwrap();
        let before = batch.clone();
        assert!(matches!(
            batch.bind(&decoder),
            Err(DecoderKvCacheError::LayerBatchSizeMismatch {
                layer: 0,
                expected: 1,
                actual: 2,
            })
        ));
        assert_eq!(batch, before);

        let mut capacity = DecoderKvCache::new(&decoder).unwrap();
        capacity.layers[0] = LayerKvCache::new(decoder.blocks()[0].attention(), 1, 3).unwrap();
        let before = capacity.clone();
        assert!(matches!(
            capacity.bind(&decoder),
            Err(DecoderKvCacheError::LayerCapacityMismatch {
                layer: 0,
                expected: 4,
                actual: 3,
            })
        ));
        assert_eq!(capacity, before);

        let one_head_config = DecoderModelConfig::new(5, 4, 1, 4, 2, 4, 10_000.0, 1e-6);
        let one_head =
            DecoderModel::from_parameters(one_head_config, decoder.parameters().to_vec()).unwrap();
        let mut heads = DecoderKvCache::new(&decoder).unwrap();
        heads.layers[0] = LayerKvCache::new(one_head.blocks()[0].attention(), 1, 4).unwrap();
        let before = heads.clone();
        assert!(matches!(
            heads.bind(&decoder),
            Err(DecoderKvCacheError::IncrementalAttention {
                layer: 0,
                source: IncrementalAttentionError::CacheHeadCountMismatch { layer: 2, cache: 1 },
            })
        ));
        assert_eq!(heads, before);

        let other_rope_config = DecoderModelConfig::new(5, 4, 2, 4, 2, 4, 20_000.0, 1e-6);
        let other_rope =
            DecoderModel::from_parameters(other_rope_config, decoder.parameters().to_vec())
                .unwrap();
        let mut rope = DecoderKvCache::new(&decoder).unwrap();
        rope.layers[0] = LayerKvCache::new(other_rope.blocks()[0].attention(), 1, 4).unwrap();
        let before = rope.clone();
        assert!(matches!(
            rope.bind(&decoder),
            Err(DecoderKvCacheError::IncrementalAttention {
                layer: 0,
                source: IncrementalAttentionError::CacheRopeMismatch { .. },
            })
        ));
        assert_eq!(rope, before);
    }

    #[test]
    fn a_second_prefill_row_failure_resets_every_logical_layer() {
        let config = config(2, 3, 0.0);
        let base = DecoderModel::new(config, &mut SplitMix64::from_seed(142)).unwrap();
        let mut parameters = base.parameters().to_vec();
        let mut embedding = parameters[0].tensor().value_snapshot();
        embedding.as_mut_slice()[4..8].fill(0.0);
        let embedding_name = parameters[0].name().to_owned();
        parameters[0] = NamedParameter::from_tensor(embedding_name, embedding).unwrap();
        let decoder = DecoderModel::from_parameters(config, parameters).unwrap();
        let mut cache = DecoderKvCache::new(&decoder).unwrap();
        let pointers = cache
            .layers
            .iter()
            .map(|layer| (layer.key_storage().as_ptr(), layer.value_storage().as_ptr()))
            .collect::<Vec<_>>();
        let mut session = cache.bind(&decoder).unwrap();

        assert!(matches!(
            session.prefill(&[0, 1]),
            Err(DecoderKvCacheError::AttentionNorm { layer: 0, .. })
        ));
        assert!(session.cache().is_empty());
        assert!(!session.cache().prefill_complete);
        assert_eq!(session.cache().work(), DecoderKvCacheWork::default());
        assert!(session.cache().layers.iter().all(LayerKvCache::is_empty));
        assert!(
            session
                .cache()
                .layers
                .iter()
                .zip(pointers)
                .all(|(layer, pointers)| {
                    layer.key_storage().as_ptr() == pointers.0
                        && layer.value_storage().as_ptr() == pointers.1
                })
        );
        assert_close(
            session.prefill(&[0]).unwrap().logits().value().as_slice(),
            &final_logits(&decoder, &[0]),
        );
    }

    #[test]
    fn final_normalization_failure_after_layer_preparation_commits_nothing() {
        let base = model(2, 3, 143);
        let mut parameters = base.parameters().to_vec();
        let final_norm = parameters.len() - 1;
        let final_norm_name = parameters[final_norm].name().to_owned();
        parameters[final_norm] = NamedParameter::from_tensor(
            final_norm_name,
            Tensor::from_vec(vec![4], vec![f64::MAX; 4]).unwrap(),
        )
            .unwrap();
        let decoder = DecoderModel::from_parameters(base.config(), parameters).unwrap();
        let mut cache = DecoderKvCache::new(&decoder).unwrap();
        let before = cache.clone();
        let mut session = cache.bind(&decoder).unwrap();

        assert!(matches!(
            session.prefill(&[0]),
            Err(DecoderKvCacheError::FinalNorm(_))
        ));
        assert_eq!(session.cache(), &before);
        assert!(session.cache().layers.iter().all(LayerKvCache::is_empty));
    }

    #[test]
    fn cached_generation_matches_tokens_draws_stops_and_exact_score_counts() {
        let model = model(2, 4, 41);
        let config = GenerationConfig::new(
            SamplingMode::TemperatureTopK {
                temperature: 1.0,
                top_k: 3,
            },
            None,
            4,
        );
        let mut cached_rng = SplitMix64::from_seed(42);
        let mut uncached_rng = cached_rng.clone();
        let cached = generate_cached(&model, &[0], config, &mut cached_rng).unwrap();
        let uncached = generate_uncached(&model, &[0], config, &mut uncached_rng).unwrap();
        assert_eq!(cached.prompt(), uncached.prompt());
        assert_eq!(cached.generated(), uncached.generated());
        assert_eq!(cached.stop(), uncached.stop());
        assert_eq!(cached_rng.state(), uncached_rng.state());
        assert!(
            cached
                .steps()
                .iter()
                .zip(uncached.steps())
                .all(|(left, right)| {
                    left.prefix_length() == right.prefix_length()
                        && left.token_id() == right.token_id()
                        && left.unit_draw() == right.unit_draw()
                        && left.interval_start() == right.interval_start()
                        && left.interval_end() == right.interval_end()
                })
        );
        assert_eq!(cached.work().prefill_tokens(), 1);
        assert_eq!(cached.work().decode_tokens(), 3);
        assert_eq!(cached.work().cache_appends(), 8);
        assert_eq!(cached.work().qkv_projection_rows(), 24);
        assert_eq!(cached.work().cached_attention_score_values(), 40);
        assert_eq!(cached.work().complete_prefix_attention_score_values(), 120);
        assert_eq!(cached.final_cache_len(), 4);
    }

    #[test]
    fn cached_generation_matches_a_small_seed_prefix_and_mode_matrix() {
        for model_seed in [47, 48] {
            let decoder = model(2, 4, model_seed);
            for prompt in [&[0][..], &[0, 1][..]] {
                for mode in [
                    SamplingMode::Greedy,
                    SamplingMode::TemperatureTopK {
                        temperature: 0.75,
                        top_k: 3,
                    },
                ] {
                    for rng_seed in [49, 50] {
                        let config = GenerationConfig::new(mode, None, 2);
                        let mut cached_rng = SplitMix64::from_seed(rng_seed);
                        let mut uncached_rng = cached_rng.clone();
                        let cached =
                            generate_cached(&decoder, prompt, config, &mut cached_rng).unwrap();
                        let uncached =
                            generate_uncached(&decoder, prompt, config, &mut uncached_rng).unwrap();
                        assert_eq!(cached.prompt(), uncached.prompt());
                        assert_eq!(cached.generated(), uncached.generated());
                        assert_eq!(cached.stop(), uncached.stop());
                        assert_eq!(cached_rng.state(), uncached_rng.state());
                        assert!(cached.steps().iter().zip(uncached.steps()).all(
                            |(left, right)| {
                                left.prefix_length() == right.prefix_length()
                                    && left.token_id() == right.token_id()
                                    && left.unit_draw() == right.unit_draw()
                                    && left.interval_start() == right.interval_start()
                                    && left.interval_end() == right.interval_end()
                            }
                        ));
                    }
                }
            }
        }
    }

    #[test]
    fn eos_context_token_limit_zero_budget_and_zero_layers_do_no_extra_work() {
        let decoder = model(2, 2, 43);
        let mut reference_rng = SplitMix64::from_seed(44);
        let one_token = generate_cached(
            &decoder,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 1),
            &mut reference_rng,
        )
            .unwrap();
        let first = one_token.generated()[0];
        assert_eq!(one_token.stop(), GenerationStop::TokenLimit);
        assert_eq!(one_token.work().decode_tokens(), 0);
        let mut eos_rng = SplitMix64::from_seed(44);
        let eos = generate_cached(
            &decoder,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, Some(first), 4),
            &mut eos_rng,
        )
            .unwrap();
        assert_eq!(eos.stop(), GenerationStop::Eos);
        assert_eq!(eos.generated(), &[first]);
        assert_eq!(eos.work().decode_tokens(), 0);

        let one_position = model(2, 1, 46);
        let mut first_rng = SplitMix64::from_seed(44);
        let first = generate_cached(
            &one_position,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 1),
            &mut first_rng,
        )
            .unwrap()
            .generated()[0];
        let mut eos_collision_rng = SplitMix64::from_seed(44);
        let eos_collision = generate_cached(
            &one_position,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, Some(first), 1),
            &mut eos_collision_rng,
        )
            .unwrap();
        assert_eq!(eos_collision.stop(), GenerationStop::Eos);
        assert_eq!(eos_collision.generated(), &[first]);
        assert_eq!(eos_collision.work().decode_tokens(), 0);
        let mut token_collision_rng = SplitMix64::from_seed(44);
        let token_collision = generate_cached(
            &one_position,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 1),
            &mut token_collision_rng,
        )
            .unwrap();
        assert_eq!(token_collision.stop(), GenerationStop::TokenLimit);
        assert_eq!(token_collision.work().decode_tokens(), 0);

        let mut context_rng = SplitMix64::from_seed(44);
        let context = generate_cached(
            &decoder,
            &[0, 1],
            GenerationConfig::new(SamplingMode::Greedy, None, 4),
            &mut context_rng,
        )
            .unwrap();
        assert_eq!(context.stop(), GenerationStop::ContextLimit);
        assert_eq!(context.generated().len(), 1);
        assert_eq!(context.final_cache_len(), 2);
        assert_eq!(context.work().decode_tokens(), 0);

        let mut zero_rng = SplitMix64::from_seed(44);
        let state = zero_rng.state();
        let zero = generate_cached(
            &decoder,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 0),
            &mut zero_rng,
        )
            .unwrap();
        assert_eq!(zero.stop(), GenerationStop::TokenLimit);
        assert_eq!(zero.final_cache_len(), 0);
        assert_eq!(zero.work().prefill_tokens(), 0);
        assert_eq!(zero_rng.state(), state);

        let zero_layer = model(0, 2, 45);
        let mut cache = DecoderKvCache::new(&zero_layer).unwrap();
        let mut session = cache.bind(&zero_layer).unwrap();
        let output = session.prefill(&[0]).unwrap();
        assert_eq!(output.logits().shape(), [1, 1, 5]);
        assert_eq!(session.cache().layer_count(), 0);
        assert_eq!(session.cache().work().attention_score_values(), 0);
    }

    #[test]
    fn invalid_cached_generation_requests_never_advance_rng() {
        let decoder = model(1, 2, 145);
        let mut rng = SplitMix64::from_seed(146);
        let state = rng.state();
        assert!(matches!(
            generate_cached(
                &decoder,
                &[],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
            ),
            Err(CachedGenerationError::EmptyPrompt)
        ));
        assert!(matches!(
            generate_cached(
                &decoder,
                &[0, 1, 2],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
            ),
            Err(CachedGenerationError::PromptTooLong { .. })
        ));
        assert!(matches!(
            generate_cached(
                &decoder,
                &[5],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
            ),
            Err(CachedGenerationError::PromptTokenOutOfBounds { .. })
        ));
        assert!(matches!(
            generate_cached(
                &decoder,
                &[0],
                GenerationConfig::new(SamplingMode::Greedy, Some(5), 1),
                &mut rng,
            ),
            Err(CachedGenerationError::EosTokenOutOfBounds { .. })
        ));
        assert!(matches!(
            generate_cached(
                &decoder,
                &[0],
                GenerationConfig::new(
                    SamplingMode::TemperatureTopK {
                        temperature: 0.0,
                        top_k: 1,
                    },
                    None,
                    1,
                ),
                &mut rng,
            ),
            Err(CachedGenerationError::Sampling(
                SamplingError::InvalidTemperature { .. }
            ))
        ));
        assert!(matches!(
            generate_cached(
                &decoder,
                &[0],
                GenerationConfig::new(
                    SamplingMode::TemperatureTopK {
                        temperature: 1.0,
                        top_k: 0,
                    },
                    None,
                    1,
                ),
                &mut rng,
            ),
            Err(CachedGenerationError::Sampling(
                SamplingError::InvalidTopK { .. }
            ))
        ));
        assert_eq!(rng.state(), state);
    }

    #[test]
    fn cached_inference_survives_a_released_training_graph_without_grad_changes() {
        let model = model(1, 3, 46);
        let loss = model.loss(&[0, 1], &[1, 2], &[1, 2]).unwrap();
        loss.backward_with_seed(
            &Tensor::from_vec(Vec::new(), vec![1.0]).unwrap().view(),
            GraphRetention::Release,
        )
            .unwrap();
        drop(loss);
        let before = model
            .parameters()
            .iter()
            .map(|parameter| {
                parameter.tensor().gradient().map(|gradient| {
                    gradient
                        .as_slice()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let mut cache = DecoderKvCache::new(&model).unwrap();
        let mut session = cache.bind(&model).unwrap();
        let output = session.prefill(&[0, 1]).unwrap();
        assert!(!output.logits().tracks_gradient());
        let decoded = session.decode(2).unwrap();
        assert!(!decoded.logits().tracks_gradient());
        let mut rng = SplitMix64::from_seed(47);
        let generated = generate_cached(
            &model,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 2),
            &mut rng,
        )
            .unwrap();
        assert_eq!(generated.generated().len(), 2);
        let after = model
            .parameters()
            .iter()
            .map(|parameter| {
                parameter.tensor().gradient().map(|gradient| {
                    gradient
                        .as_slice()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(after, before);
    }

    #[test]
    fn checked_counter_helpers_reject_overflow() {
        assert_eq!(
            checked_add(usize::MAX, 1, DecoderKvCacheCounter::AttentionScoreValues),
            Err(DecoderKvCacheError::WorkOverflow {
                counter: DecoderKvCacheCounter::AttentionScoreValues
            })
        );
        assert_eq!(
            checked_complete_prefix_scores(0, usize::MAX, 2, 2),
            Err(CachedGenerationError::WorkOverflow {
                counter: DecoderKvCacheCounter::CompletePrefixAttentionScoreValues
            })
        );
    }
}
