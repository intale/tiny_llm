//! One differentiable pre-normalized causal Transformer decoder block.

use std::error::Error;
use std::fmt;

use crate::attention::multi_head::{
    MultiHeadAttention, MultiHeadAttentionError, MultiHeadAttentionForward,
};
use crate::autograd::tensor_core::{AutogradContext, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::nn::residual::{ResidualError};
use crate::nn::rms_norm::{RmsNorm, RmsNormError, RmsNormForward};
use crate::nn::swiglu::{SwiGlu, SwiGluError, SwiGluForward};

/// A component whose feature width is incompatible with the residual stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderBlockComponent {
    AttentionNorm,
    FeedForwardNorm,
    FeedForwardInput,
    FeedForwardOutput,
}

impl fmt::Display for DecoderBlockComponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AttentionNorm => "attention RMSNorm",
            Self::FeedForwardNorm => "feed-forward RMSNorm",
            Self::FeedForwardInput => "feed-forward input",
            Self::FeedForwardOutput => "feed-forward output",
        })
    }
}

/// A rejected component assembly or stage of one decoder-block forward pass.
#[derive(Clone, Debug, PartialEq)]
pub enum DecoderBlockError {
    AttentionNorm(RmsNormError),
    Attention(MultiHeadAttentionError),
    AttentionResidual(ResidualError),
    FeedForwardNorm(RmsNormError),
    FeedForward(SwiGluError),
    FeedForwardResidual(ResidualError),
    ComponentWidthMismatch {
        component: DecoderBlockComponent,
        expected: usize,
        actual: usize,
    },
    Initialization(InitializationError),
}

impl fmt::Display for DecoderBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AttentionNorm(source) => write!(formatter, "attention RMSNorm: {source}"),
            Self::Attention(source) => write!(formatter, "causal multi-head attention: {source}"),
            Self::AttentionResidual(source) => {
                write!(formatter, "attention residual merge: {source}")
            }
            Self::FeedForwardNorm(source) => {
                write!(formatter, "feed-forward RMSNorm: {source}")
            }
            Self::FeedForward(source) => write!(formatter, "SwiGLU feed-forward: {source}"),
            Self::FeedForwardResidual(source) => {
                write!(formatter, "feed-forward residual merge: {source}")
            }
            Self::ComponentWidthMismatch {
                component,
                expected,
                actual,
            } => write!(
                formatter,
                "decoder-block {component} width must be {expected}, got {actual}"
            ),
            Self::Initialization(source) => source.fmt(formatter),
        }
    }
}

impl Error for DecoderBlockError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::AttentionNorm(source) | Self::FeedForwardNorm(source) => Some(source),
            Self::Attention(source) => Some(source),
            Self::AttentionResidual(source) | Self::FeedForwardResidual(source) => Some(source),
            Self::FeedForward(source) => Some(source),
            Self::Initialization(source) => Some(source),
            Self::ComponentWidthMismatch { .. } => None,
        }
    }
}

impl From<InitializationError> for DecoderBlockError {
    fn from(source: InitializationError) -> Self {
        Self::Initialization(source)
    }
}

/// Dimensions and numerical constants needed to initialize one decoder block
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DecoderBlockConfig {
    model_width: usize,
    heads: usize,
    feed_forward_width: usize,
    max_positions: usize,
    rope_base: f64,
    rms_epsilon: f64,
}

impl DecoderBlockConfig {
    pub fn new(
        model_width: usize,
        heads: usize,
        feed_forward_width: usize,
        max_positions: usize,
        rope_base: f64,
        rms_epsilon: f64,
    ) -> Self {
        Self {
            model_width,
            heads,
            feed_forward_width,
            max_positions,
            rope_base,
            rms_epsilon,
        }
    }

    pub fn model_width(self) -> usize {
        self.model_width
    }

    pub fn heads(self) -> usize {
        self.heads
    }

    pub fn feed_forward_width(self) -> usize {
        self.feed_forward_width
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
}

/// Every inspectable value produced by the two pre-normalized residual paths
#[derive(Clone, Debug)]
pub struct DecoderBlockForward {
    attention_norm: RmsNormForward,
    attention: MultiHeadAttentionForward,
    after_attention: TensorValue,
    feed_forward_norm: RmsNormForward,
    feed_forward: SwiGluForward,
    output: TensorValue,
}

impl DecoderBlockForward {
    pub fn attention_norm(&self) -> &RmsNormForward {
        &self.attention_norm
    }

    pub fn attention(&self) -> &MultiHeadAttentionForward {
        &self.attention
    }

    pub fn attention_weights(&self) -> &TensorValue {
        self.attention.attention_weights()
    }

    pub fn after_attention(&self) -> &TensorValue {
        &self.after_attention
    }

    pub fn feed_forward_norm(&self) -> &RmsNormForward {
        &self.feed_forward_norm
    }

    pub fn feed_forward(&self) -> &SwiGluForward {
        &self.feed_forward
    }

    pub fn output(&self) -> &TensorValue {
        &self.output
    }

    pub fn into_output(self) -> TensorValue {
        self.output
    }
}

/// A decoder block with two pre-normalized residual sublayers.
///
/// ```text
/// input ──┬──────────────────────────► (+) ─► after_attention
///         │                              ▲
///         └─► RMSNorm ─► causal MHA ─────┘
///
/// after_attention ──┬─────────────────► (+) ─► output
///                   │                    ▲
///                   └─► RMSNorm ─► SwiGLU┘
/// ```
///
/// In other words:
///
/// ```text
/// after_attention = input + causal_mha(rms_norm(input))
/// output = after_attention + swiglu(rms_norm(after_attention))
/// ```
///
/// The first residual result therefore has a direct path around the feed-forward transformation
/// and is added to the [`SwiGlu`] result at the block output.
#[derive(Clone, Debug)]
pub struct DecoderBlock {
    attention_norm: RmsNorm,
    attention: MultiHeadAttention,
    feed_forward_norm: RmsNorm,
    feed_forward: SwiGlu,
    parameters: NamedParameters,
    model_width: usize
}

impl DecoderBlock {
    /// Initializes every matrix transactionally from one deterministic stream.
    pub fn new(
        parameter_prefix: impl Into<String>,
        config: DecoderBlockConfig,
        rng: &mut SplitMix64,
    ) -> Result<Self, DecoderBlockError> {
        let parameter_prefix = parameter_prefix.into();
        let attention_norm = RmsNorm::new(
            format!("{parameter_prefix}.attention_norm.gain"),
            config.model_width,
            config.rms_epsilon,
        )
            .map_err(DecoderBlockError::AttentionNorm)?;
        let feed_forward_norm = RmsNorm::new(
            format!("{parameter_prefix}.ffn_norm.gain"),
            config.model_width,
            config.rms_epsilon,
        )
            .map_err(DecoderBlockError::FeedForwardNorm)?;

        let mut trial = rng.clone();
        let attention = MultiHeadAttention::new(
            format!("{parameter_prefix}.attention"),
            config.model_width,
            config.heads,
            config.max_positions,
            config.rope_base,
            &mut trial,
        )
            .map_err(DecoderBlockError::Attention)?;
        let feed_forward = SwiGlu::new(
            format!("{parameter_prefix}.ffn"),
            config.model_width,
            config.feed_forward_width,
            config.model_width,
            &mut trial,
        )
            .map_err(DecoderBlockError::FeedForward)?;
        let block = Self::from_parts(attention_norm, attention, feed_forward_norm, feed_forward)?;
        *rng = trial;
        Ok(block)
    }

    /// Assembles already named deterministic components after cross-width checks.
    pub fn from_parts(
        attention_norm: RmsNorm,
        attention: MultiHeadAttention,
        feed_forward_norm: RmsNorm,
        feed_forward: SwiGlu,
    ) -> Result<Self, DecoderBlockError> {
        let model_width = attention.model_width();
        for (component, actual) in [
            (
                DecoderBlockComponent::AttentionNorm,
                attention_norm.feature_width(),
            ),
            (
                DecoderBlockComponent::FeedForwardNorm,
                feed_forward_norm.feature_width(),
            ),
            (
                DecoderBlockComponent::FeedForwardInput,
                feed_forward.input_width(),
            ),
            (
                DecoderBlockComponent::FeedForwardOutput,
                feed_forward.output_width(),
            ),
        ] {
            if actual != model_width {
                return Err(DecoderBlockError::ComponentWidthMismatch {
                    component,
                    expected: model_width,
                    actual,
                });
            }
        }

        let mut listed = Vec::with_capacity(9);
        listed.extend(attention_norm.parameters().iter().cloned());
        listed.extend(attention.parameters().iter().cloned());
        listed.extend(feed_forward_norm.parameters().iter().cloned());
        listed.extend(feed_forward.parameters().into_iter().cloned());
        let parameters = NamedParameters::try_new(listed)?;

        Ok(Self {
            attention_norm,
            attention,
            feed_forward_norm,
            feed_forward,
            parameters,
            model_width,
        })
    }

    /// Runs the two transformation branches in exact pre-normalized order.
    pub fn forward(
        &self,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<DecoderBlockForward, DecoderBlockError> {
        self.forward_with_context(AutogradContext::recording(), input, position_offset)
    }

    /// Runs both pre-normalized branches under an explicit recording policy.
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<DecoderBlockForward, DecoderBlockError> {
        let attention_norm = self.attention_norm.forward_with_intermediates_and_context(context, input).map_err(DecoderBlockError::AttentionNorm)?;
        let attention = self.attention.forward_with_context(context, attention_norm.output(), position_offset).map_err(DecoderBlockError::Attention)?;
        let after_attention = attention.output().residual_add_with_context(context, input).map_err(DecoderBlockError::AttentionResidual)?;
        let feed_forward_norm = self.feed_forward_norm.forward_with_intermediates_and_context(context, &after_attention).map_err(DecoderBlockError::FeedForwardNorm)?;
        let feed_forward = self.feed_forward.forward_with_intermediates_and_context(context, feed_forward_norm.output()).map_err(DecoderBlockError::FeedForward)?;
        let output = after_attention.residual_add_with_context(context, feed_forward.output()).map_err(DecoderBlockError::FeedForwardResidual)?;

        Ok(DecoderBlockForward {
            attention_norm,
            attention,
            after_attention,
            feed_forward_norm,
            feed_forward,
            output,
        })
    }

    pub fn attention_norm(&self) -> &RmsNorm {
        &self.attention_norm
    }

    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }

    pub fn feed_forward_norm(&self) -> &RmsNorm {
        &self.feed_forward_norm
    }

    pub fn feed_forward(&self) -> &SwiGlu {
        &self.feed_forward
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn model_width(&self) -> usize {
        self.model_width
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters
            .as_slice()
            .iter()
            .map(|parameter| parameter.tensor().value().len())
            .sum()
    }
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::autograd::tensor_core::{GraphRetention, TensorOperation};
    use crate::nn::init::NamedParameter;
    use crate::tensor::storage::Tensor;

    const MODEL_WIDTH: usize = 4;
    const TOKENS: usize = 3;
    const HEADS: usize = 2;
    const MAX_POSITIONS: usize = 6;
    const ROPE_BASE: f64 = 100.0;
    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    const INPUT: [f64; 12] = [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];

    fn tensor(shape: &[usize], values: &[f64]) -> Tensor {
        Tensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
    }

    fn parameter(name: &str, shape: &[usize], values: &[f64]) -> NamedParameter {
        NamedParameter::from_tensor(name, tensor(shape, values)).unwrap()
    }

    fn norm(name: &str, width: usize) -> RmsNorm {
        RmsNorm::from_gain(parameter(name, &[width], &vec![1.0; width]), 0.0).unwrap()
    }

    fn attention() -> MultiHeadAttention {
        MultiHeadAttention::from_parameters(
            parameter(
                "decoder.block.0.attention.query.weight",
                &[MODEL_WIDTH, MODEL_WIDTH],
                &IDENTITY,
            ),
            parameter(
                "decoder.block.0.attention.key.weight",
                &[MODEL_WIDTH, MODEL_WIDTH],
                &IDENTITY,
            ),
            parameter(
                "decoder.block.0.attention.value.weight",
                &[MODEL_WIDTH, MODEL_WIDTH],
                &IDENTITY,
            ),
            parameter(
                "decoder.block.0.attention.output.weight",
                &[MODEL_WIDTH, MODEL_WIDTH],
                &IDENTITY,
            ),
            HEADS,
            MAX_POSITIONS,
            ROPE_BASE,
        )
            .unwrap()
    }

    fn feed_forward(input_width: usize, output_width: usize) -> SwiGlu {
        let gate = (0..input_width * MODEL_WIDTH)
            .map(|index| usize::from(index / MODEL_WIDTH == index % MODEL_WIDTH) as f64)
            .collect::<Vec<_>>();
        let down = (0..MODEL_WIDTH * output_width)
            .map(|index| usize::from(index / output_width == index % output_width) as f64)
            .collect::<Vec<_>>();
        SwiGlu::from_parameters(
            parameter(
                "decoder.block.0.ffn.gate.weight",
                &[input_width, MODEL_WIDTH],
                &gate,
            ),
            parameter(
                "decoder.block.0.ffn.up.weight",
                &[input_width, MODEL_WIDTH],
                &gate,
            ),
            parameter(
                "decoder.block.0.ffn.down.weight",
                &[MODEL_WIDTH, output_width],
                &down,
            ),
        )
            .unwrap()
    }

    fn fixture() -> DecoderBlock {
        DecoderBlock::from_parts(
            norm("decoder.block.0.attention_norm.gain", MODEL_WIDTH),
            attention(),
            norm("decoder.block.0.ffn_norm.gain", MODEL_WIDTH),
            feed_forward(MODEL_WIDTH, MODEL_WIDTH),
        )
            .unwrap()
    }

    #[test]
    fn exact_pre_norm_order_preserves_two_residual_identities() {
        let block = fixture();
        let input = TensorValue::constant(tensor(&[1, TOKENS, MODEL_WIDTH], &INPUT)).unwrap();
        let pass = block.forward(&input, 0).unwrap();

        assert_eq!(&*pass.attention_norm().output().value(), &*input.value());
        let expected_after_attention = input
            .value()
            .as_slice()
            .iter()
            .zip(pass.attention().output().value().as_slice())
            .map(|(identity, branch)| identity + branch)
            .collect::<Vec<_>>();
        assert_eq!(
            pass.after_attention().value().as_slice(),
            expected_after_attention
        );
        let expected_output = pass
            .after_attention()
            .value()
            .as_slice()
            .iter()
            .zip(pass.feed_forward().output().value().as_slice())
            .map(|(identity, branch)| identity + branch)
            .collect::<Vec<_>>();
        assert_eq!(pass.output().value().as_slice(), expected_output);
        assert_ne!(
            &*pass.after_attention().value(),
            &*pass.feed_forward_norm().output().value()
        );
    }

    #[test]
    fn shape_causality_and_suffix_invariance_hold_for_the_complete_block() {
        let block = fixture();
        let input = TensorValue::constant(tensor(&[1, TOKENS, MODEL_WIDTH], &INPUT)).unwrap();
        let pass = block.forward(&input, 0).unwrap();
        assert_eq!(pass.attention_weights().shape(), [1, HEADS, TOKENS, TOKENS]);
        assert_eq!(pass.output().shape(), [1, TOKENS, MODEL_WIDTH]);
        for head in 0..HEADS {
            for query in 0..TOKENS {
                let start = (head * TOKENS + query) * TOKENS;
                assert!(
                    pass.attention_weights().value().as_slice()[start + query + 1..start + TOKENS]
                        .iter()
                        .all(|value| *value == 0.0)
                );
            }
        }

        let mut perturbed = INPUT;
        perturbed[8..].copy_from_slice(&[1.0, -1.0, 1.0, 1.0]);
        let changed = block
            .forward(
                &TensorValue::constant(tensor(&[1, TOKENS, MODEL_WIDTH], &perturbed)).unwrap(),
                0,
            )
            .unwrap()
            .output()
            .value_snapshot();
        assert_eq!(
            pass.output().value().as_slice()[..8],
            changed.as_slice()[..8]
        );
        assert_ne!(
            pass.output().value().as_slice()[8..],
            changed.as_slice()[8..]
        );

        let empty_batch = block
            .forward(
                &TensorValue::constant(tensor(&[0, TOKENS, MODEL_WIDTH], &[])).unwrap(),
                0,
            )
            .unwrap();
        assert_eq!(empty_batch.output().shape(), [0, TOKENS, MODEL_WIDTH]);
    }

    #[test]
    fn stable_parameter_order_is_bias_free_and_has_the_formula_count() {
        let block = fixture();
        assert_eq!(
            block
                .parameters()
                .iter()
                .map(|parameter| parameter.name())
                .collect::<Vec<_>>(),
            [
                "decoder.block.0.attention_norm.gain",
                "decoder.block.0.attention.query.weight",
                "decoder.block.0.attention.key.weight",
                "decoder.block.0.attention.value.weight",
                "decoder.block.0.attention.output.weight",
                "decoder.block.0.ffn_norm.gain",
                "decoder.block.0.ffn.gate.weight",
                "decoder.block.0.ffn.up.weight",
                "decoder.block.0.ffn.down.weight",
            ]
        );
        assert_eq!(block.parameters().len(), 9);
        assert_eq!(block.parameter_count(), 120);
        assert!(!block.attention().qkv().query().has_bias());
        assert!(!block.attention().qkv().key().has_bias());
        assert!(!block.attention().qkv().value().has_bias());
        assert!(!block.attention().output_projection().has_bias());
        assert!(!block.feed_forward().gate().has_bias());
        assert!(!block.feed_forward().up().has_bias());
        assert!(!block.feed_forward().down().has_bias());
    }

    #[test]
    fn initialization_is_deterministic_and_transactional() {
        let config = DecoderBlockConfig::new(4, 2, 4, 6, 100.0, 1e-5);
        let mut first_rng = SplitMix64::from_seed(31);
        let mut second_rng = SplitMix64::from_seed(31);
        let first = DecoderBlock::new("decoder.block.0", config, &mut first_rng).unwrap();
        let second = DecoderBlock::new("decoder.block.0", config, &mut second_rng).unwrap();
        assert_eq!(first_rng.state(), second_rng.state());
        assert!(
            first
                .parameters()
                .iter()
                .zip(second.parameters())
                .all(|(left, right)| *left.tensor().value() == *right.tensor().value())
        );

        let mut rejected_rng = SplitMix64::from_seed(31);
        let state_before = rejected_rng.state();
        assert!(matches!(
            DecoderBlock::new(
                "decoder.block.0",
                DecoderBlockConfig::new(4, 2, 0, 6, 100.0, 1e-5),
                &mut rejected_rng,
            ),
            Err(DecoderBlockError::FeedForward(_))
        ));
        assert_eq!(rejected_rng.state(), state_before);
    }

    #[test]
    fn cross_component_widths_and_duplicate_names_are_rejected() {
        assert_eq!(
            DecoderBlock::from_parts(
                norm("decoder.block.0.attention_norm.gain", MODEL_WIDTH),
                attention(),
                norm("decoder.block.0.ffn_norm.gain", MODEL_WIDTH),
                feed_forward(MODEL_WIDTH, 3),
            )
                .unwrap_err(),
            DecoderBlockError::ComponentWidthMismatch {
                component: DecoderBlockComponent::FeedForwardOutput,
                expected: MODEL_WIDTH,
                actual: 3,
            }
        );

        assert!(matches!(
            DecoderBlock::from_parts(
                norm("decoder.block.0.attention.query.weight", MODEL_WIDTH),
                attention(),
                norm("decoder.block.0.ffn_norm.gain", MODEL_WIDTH),
                feed_forward(MODEL_WIDTH, MODEL_WIDTH),
            ),
            Err(DecoderBlockError::Initialization(
                InitializationError::DuplicateName { .. }
            ))
        ));
    }

    #[test]
    fn forward_error_precedence_names_the_first_owned_stage() {
        let block = fixture();
        let wrong_width = TensorValue::constant(tensor(&[1, 3], &[0.0; 3])).unwrap();
        assert!(matches!(
            block.forward(&wrong_width, 0),
            Err(DecoderBlockError::AttentionNorm(
                RmsNormError::InputWidthMismatch {
                    expected: MODEL_WIDTH,
                    actual: 3,
                }
            ))
        ));

        let rank_one = TensorValue::constant(tensor(&[MODEL_WIDTH], &[1.0; MODEL_WIDTH])).unwrap();
        assert!(matches!(
            block.forward(&rank_one, 0),
            Err(DecoderBlockError::Attention(
                MultiHeadAttentionError::InputRank { rank: 1 }
            ))
        ));

        let empty_tokens = TensorValue::constant(tensor(&[1, 0, MODEL_WIDTH], &[])).unwrap();
        assert!(matches!(
            block.forward(&empty_tokens, 0),
            Err(DecoderBlockError::Attention(
                MultiHeadAttentionError::EmptyTokens
            ))
        ));
        let input = TensorValue::constant(tensor(&[1, TOKENS, MODEL_WIDTH], &INPUT)).unwrap();
        assert!(matches!(
            block.forward(&input, MAX_POSITIONS - 1),
            Err(DecoderBlockError::Attention(
                MultiHeadAttentionError::PositionRangeExceeded { .. }
            ))
        ));

        let leaf = TensorValue::parameter(tensor(&[1, TOKENS, MODEL_WIDTH], &INPUT)).unwrap();
        let scale = TensorValue::constant(tensor(&[], &[1.0])).unwrap();
        let released = leaf.mul(&scale).unwrap();
        released
            .backward_with_seed(
                &tensor(&[1, TOKENS, MODEL_WIDTH], &[1.0; 12]).view(),
                GraphRetention::Release,
            )
            .unwrap();
        assert!(matches!(
            block.forward(&released, 0),
            Err(DecoderBlockError::AttentionNorm(RmsNormError::Autodiff {
                source: crate::autograd::tensor_core::TensorAutodiffError::ReleasedOperand {
                    operation: TensorOperation::Multiply,
                    ..
                },
                ..
            }))
        ));
    }

    #[test]
    fn complete_backward_reaches_input_and_all_nine_parameters() {
        let block = fixture();
        let input = TensorValue::parameter(tensor(&[1, TOKENS, MODEL_WIDTH], &INPUT)).unwrap();
        let pass = block.forward(&input, 0).unwrap();
        pass.output()
            .backward_with_seed(
                &tensor(
                    &[1, TOKENS, MODEL_WIDTH],
                    &[
                        1.0, -0.5, 0.25, 0.75, -0.3, 0.8, 1.2, -0.4, 0.6, 0.1, -0.7, 0.9,
                    ],
                )
                    .view(),
                GraphRetention::Retain,
            )
            .unwrap();
        assert!(
            input
                .gradient()
                .unwrap()
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(block.parameters().iter().all(|parameter| {
            parameter
                .tensor()
                .gradient()
                .is_some_and(|gradient| gradient.as_slice().iter().all(|value| value.is_finite()))
        }));
    }
}
