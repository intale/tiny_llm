//! One unmasked, scaled dot-product self-attention head

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::tensor::storage::{Tensor, TensorError};

/// One of the three inputs to a self-attention head
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SelfAttentionInput {
    Query,
    Key,
    Value,
}

impl fmt::Display for SelfAttentionInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
        })
    }
}

/// The forward stage at which a cumulative tensor operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfAttentionStage {
    KeyTranspose,
    RawScores,
    ScaleTensor,
    ScaledScores,
    LogSoftmax,
    Probabilities,
    ValueMixture,
}

impl fmt::Display for SelfAttentionStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::KeyTranspose => "key transpose",
            Self::RawScores => "raw query-key scores",
            Self::ScaleTensor => "score scale",
            Self::ScaledScores => "scaled query-key scores",
            Self::LogSoftmax => "row log-softmax",
            Self::Probabilities => "attention probabilities",
            Self::ValueMixture => "weighted value mixture",
        })
    }
}

/// A rejected Q/K/V shape or cumulative tensor operation.
#[derive(Clone, Debug, PartialEq)]
pub enum SelfAttentionError {
    InputRank {
        input: SelfAttentionInput,
        rank: usize,
    },
    BatchMismatch {
        query: usize,
        key: usize,
        value: usize,
    },
    TokenMismatch {
        query: usize,
        key: usize,
        value: usize,
    },
    QueryKeyWidthMismatch {
        query: usize,
        key: usize,
    },
    EmptyTokens,
    EmptyFeatureWidth {
        input: SelfAttentionInput,
    },
    Tensor {
        stage: SelfAttentionStage,
        source: TensorError,
    },
    Autodiff {
        stage: SelfAttentionStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for SelfAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputRank { input, rank } => write!(
                formatter,
                "self-attention {input} must have rank three [batch, tokens, features], got rank {rank}"
            ),
            Self::BatchMismatch { query, key, value } => write!(
                formatter,
                "self-attention batch sizes must match, got query {query}, key {key}, value {value}"
            ),
            Self::TokenMismatch { query, key, value } => write!(
                formatter,
                "unmasked self-attention token counts must match, got query {query}, key {key}, value {value}"
            ),
            Self::QueryKeyWidthMismatch { query, key } => write!(
                formatter,
                "self-attention query and key widths must match, got query {query}, key {key}"
            ),
            Self::EmptyTokens => formatter.write_str(
                "unmasked self-attention needs at least one token so every probability row has a key",
            ),
            Self::EmptyFeatureWidth { input } => write!(
                formatter,
                "self-attention {input} needs a nonzero feature width"
            ),
            Self::Tensor { stage, source } => {
                write!(formatter, "self-attention {stage}: {source}")
            }
            Self::Autodiff { stage, source } => {
                write!(formatter, "self-attention {stage}: {source}")
            }
        }
    }
}

impl Error for SelfAttentionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor { source, .. } => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn autodiff_error(
    stage: SelfAttentionStage,
) -> impl FnOnce(TensorAutodiffError) -> SelfAttentionError {
    move |source| SelfAttentionError::Autodiff { stage, source }
}

/// Shared, validated score preparation for unmasked adn causally masked heads
#[derive(Clone, Debug)]
pub struct ScaledSelfAttentionScores {
    pub raw_scores: TensorValue,
    pub scaled_scores: TensorValue,
    pub scale: f64,
    pub key_width: usize,
    pub value_width: usize,
}

/// Inspectable evidence from one unmasked attention head
#[derive(Clone, Debug)]
pub struct SelfAttentionForward {
    raw_scores: TensorValue,
    scaled_scores: TensorValue,
    weights: TensorValue,
    output: TensorValue,
    scale: f64,
    key_width: usize,
    value_width: usize,
}

impl SelfAttentionForward {
    /// The unnormalized matrix `Q K^T` with shape `[batch, tokens, tokens]`.
    pub fn raw_scores(&self) -> &TensorValue {
        &self.raw_scores
    }

    /// Alias that emphasizes that each raw cell is one query-key dot product.
    pub fn dot_products(&self) -> &TensorValue {
        &self.raw_scores
    }

    /// The raw scores divided by the square root of the query/key width.
    pub fn scaled_scores(&self) -> &TensorValue {
        &self.scaled_scores
    }

    /// Row-normalized probabilities over key positions.
    pub fn weights(&self) -> &TensorValue {
        &self.weights
    }

    /// Alias for the row-normalized attention weights.
    pub fn probabilities(&self) -> &TensorValue {
        &self.weights
    }

    /// The weighted value rows with shape `[batch, tokens, value_width]`.
    pub fn output(&self) -> &TensorValue {
        &self.output
    }

    /// The fixed score multiplier `1 / sqrt(key_width)` used by this pass.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn key_width(&self) -> usize {
        self.key_width
    }

    pub fn value_width(&self) -> usize {
        self.value_width
    }

    pub fn into_output(self) -> TensorValue {
        self.output
    }

    pub fn into_parts(self) -> (TensorValue, TensorValue, TensorValue, TensorValue) {
        (
            self.raw_scores,
            self.scaled_scores,
            self.weights,
            self.output,
        )
    }
}

/// Computes one unmasked scaled dot-product self-attention head
///
/// Q, K and V must describe the same batch and token positions. Q and K share one nonzero feature
/// width; V may use a different nonzero output width
pub fn scaled_dot_product_self_attention(
    query: &TensorValue,
    key: &TensorValue,
    value: &TensorValue,
) -> Result<SelfAttentionForward, SelfAttentionError> {
    scaled_dot_product_self_attention_with_context(AutogradContext::recording(), query, key, value)
}

/// Computes one unmasked scaled dot-product self-attention head.
///
/// Self-attention lets each query position gather information from other token
/// positions. Q and K determine how strongly each position should contribute,
/// while V provides the feature vectors to be combined.
///
/// The computation is:
///
/// ```text
/// scores  = (Q @ Kᵀ) / √dₖ
/// weights = softmax(scores, key axis)
/// output  = weights @ V
/// ```
///
/// Tensor shapes:
///
/// ```text
/// Q, K:    [B, T, dₖ]
/// V:       [B, T, dᵥ]
/// scores:  [B, T, T]
/// weights: [B, T, T]
/// output:  [B, T, dᵥ]
/// ```
///
/// Each row of `weights` represents one query's attention distribution over
/// all key positions. Each output row is the weighted sum Σⱼ Aᵢⱼ · Vⱼ.
///
/// The implementation obtains weights through `exp(log_softmax(scores))`,
/// reusing existing differentiable tensor operations.
///
/// Attention is unmasked: queries can access past, current, and future tokens.
/// Autoregressive language models require causal masking to prevent future
/// token information from leaking into earlier positions.
///
/// Returns intermediate scores, normalized attention weights, and the resulting
/// Value mixtures for inspection and testing.
///
/// `context` controls whether the operations are recorded for backward.
pub fn scaled_dot_product_self_attention_with_context(
    context: AutogradContext,
    query: &TensorValue,
    key: &TensorValue,
    value: &TensorValue,
) -> Result<SelfAttentionForward, SelfAttentionError> {
    let prepared = scaled_self_attention_scores_with_context(context, query, key, value)?;
    let log_weights = prepared
        .scaled_scores
        .log_softmax_with_context(context, 2)
        .map_err(autodiff_error(SelfAttentionStage::LogSoftmax))?;
    let weights = log_weights
        .exp_with_context(context)
        .map_err(autodiff_error(SelfAttentionStage::Probabilities))?;
    let output = weights
        .matmul_with_context(context, value)
        .map_err(autodiff_error(SelfAttentionStage::ValueMixture))?;

    Ok(SelfAttentionForward {
        raw_scores: prepared.raw_scores,
        scaled_scores: prepared.scaled_scores,
        weights,
        output,
        scale: prepared.scale,
        key_width: prepared.key_width,
        value_width: prepared.value_width,
    })
}

/// Validates Q/K/V shapes and computes scaled query-key compatibility scores.
///
/// This helper prepares scores shared by unmasked and causal self-attention.
/// It measures how strongly each query matches each key, but does not yet
/// normalize those scores or combine Value vectors.
///
/// ```text
/// raw_scores    = Q @ Kᵀ
/// scale         = 1 / √dₖ
/// scaled_scores = raw_scores × scale
/// ```
///
/// Tensor shapes:
///
/// ```text
/// Q, K:          [B, T, dₖ]
/// V:             [B, T, dᵥ]
/// raw_scores:    [B, T, T]
/// scaled_scores: [B, T, T]
/// ```
///
/// Every score Sᵢⱼ is the dot product of query i and key j, scaled by 1/√dₖ.
/// The first token axis identifies the querying position; the second
/// identifies the key position being evaluated.
///
/// Q and K must have equal feature widths because their vectors participate
/// in dot products. V may have a different width because it is used later
/// for weighted feature aggregation, not for computing compatibility scores.
///
/// All inputs must have rank 3 and share the same batch size and token count.
/// Token count and feature widths must be nonzero; an empty batch is allowed.
///
/// Dividing by √dₖ controls the typical magnitude of query-key dot products
/// before softmax, reducing the risk of excessively sharp attention weights.
///
/// V is validated here even though its values are not used to compute scores.
/// This ensures that the subsequent `weights @ V` operation is shape-compatible.
///
/// Returns raw scores, scaled scores, the scale factor, and Q/K and V widths.
///
/// The returned scores are not probabilities. The caller must apply the
/// appropriate softmax, optionally with causal masking, before mixing V.
///
/// `context` controls whether the operations are recorded for backward.
pub fn scaled_self_attention_scores_with_context(
    context: AutogradContext,
    query: &TensorValue,
    key: &TensorValue,
    value: &TensorValue,
) -> Result<ScaledSelfAttentionScores, SelfAttentionError> {
    let query_shape = query.shape();
    let key_shape = key.shape();
    let value_shape = value.shape();

    for (input, shape) in [
        (SelfAttentionInput::Query, query_shape.as_slice()),
        (SelfAttentionInput::Key, key_shape.as_slice()),
        (SelfAttentionInput::Value, value_shape.as_slice()),
    ] {
        if shape.len() != 3 {
            return Err(SelfAttentionError::InputRank {
                input,
                rank: shape.len(),
            });
        }
    }

    if query_shape[0] != key_shape[0] || query_shape[0] != value_shape[0] {
        return Err(SelfAttentionError::BatchMismatch {
            query: query_shape[0],
            key: key_shape[0],
            value: value_shape[0],
        });
    }
    if query_shape[1] != key_shape[1] || query_shape[1] != value_shape[1] {
        return Err(SelfAttentionError::TokenMismatch {
            query: query_shape[1],
            key: key_shape[1],
            value: value_shape[1],
        });
    }
    if query_shape[1] == 0 {
        return Err(SelfAttentionError::EmptyTokens);
    }
    if query_shape[2] == 0 {
        return Err(SelfAttentionError::EmptyFeatureWidth {
            input: SelfAttentionInput::Query,
        });
    }
    if key_shape[2] == 0 {
        return Err(SelfAttentionError::EmptyFeatureWidth {
            input: SelfAttentionInput::Key,
        });
    }
    if query_shape[2] != key_shape[2] {
        return Err(SelfAttentionError::QueryKeyWidthMismatch {
            query: query_shape[2],
            key: key_shape[2],
        });
    }
    if value_shape[2] == 0 {
        return Err(SelfAttentionError::EmptyFeatureWidth {
            input: SelfAttentionInput::Value,
        });
    }

    let key_transposed = key
        .transpose_with_context(context, 1, 2)
        .map_err(autodiff_error(SelfAttentionStage::KeyTranspose))?;
    let raw_scores = query
        .matmul_with_context(context, &key_transposed)
        .map_err(autodiff_error(SelfAttentionStage::RawScores))?;

    let scale = 1.0 / (query_shape[2] as f64).sqrt();
    let scale_tensor =
        Tensor::from_vec(Vec::new(), vec![scale]).map_err(|source| SelfAttentionError::Tensor {
            stage: SelfAttentionStage::ScaleTensor,
            source,
        })?;
    let scale_value = TensorValue::constant(scale_tensor)
        .map_err(autodiff_error(SelfAttentionStage::ScaleTensor))?;
    let scaled_scores = raw_scores
        .mul_with_context(context, &scale_value)
        .map_err(autodiff_error(SelfAttentionStage::ScaledScores))?;

    Ok(ScaledSelfAttentionScores {
        raw_scores,
        scaled_scores,
        scale,
        key_width: query_shape[2],
        value_width: value_shape[2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fn_scaled_dot_product_self_attention_with_context {
        use super::*;

        mod when_some_tensor_rank_isnt_3 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::InputRank {
                        input: SelfAttentionInput::Value,
                        rank: 2,
                    }
                );
            }
        }

        mod when_query_and_key_batch_size_mismatch {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![3, 1, 2], vec![0.; 6]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::BatchMismatch {
                        query: 3,
                        key: 1,
                        value: 1,
                    }
                );
            }
        }

        mod when_query_and_value_batch_size_mismatch {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![3, 1, 2], vec![0.; 6]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::BatchMismatch {
                        query: 1,
                        key: 1,
                        value: 3,
                    }
                );
            }
        }

        mod when_query_and_key_tokens_size_mismatch {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 3, 2], vec![0.; 6]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::TokenMismatch {
                        query: 3,
                        key: 1,
                        value: 1,
                    }
                );
            }
        }

        mod when_query_and_value_tokens_size_mismatch {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 3, 2], vec![0.; 6]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::TokenMismatch {
                        query: 1,
                        key: 1,
                        value: 3,
                    }
                );
            }
        }

        mod when_tokens_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 0, 2], vec![]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 0, 2], vec![]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 0, 2], vec![]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::EmptyTokens
                );
            }
        }

        mod when_query_last_axis_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 0], vec![]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::EmptyFeatureWidth {
                        input: SelfAttentionInput::Query,
                    }
                );
            }
        }

        mod when_key_last_axis_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 0], vec![]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::EmptyFeatureWidth {
                        input: SelfAttentionInput::Key,
                    }
                );
            }
        }

        mod when_key_last_axis_does_not_match_key_last_axis {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 3], vec![0.; 3]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::QueryKeyWidthMismatch {
                        query: 2,
                        key: 3,
                    }
                );
            }
        }

        mod when_value_last_axis_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let q_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let k_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 2], vec![0.; 2]).unwrap())
                        .unwrap();
                let v_param =
                    TensorValue::parameter(Tensor::from_vec(vec![1, 1, 0], vec![]).unwrap())
                        .unwrap();
                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    SelfAttentionError::EmptyFeatureWidth {
                        input: SelfAttentionInput::Value,
                    }
                );
            }
        }

        mod feed_forward_calculation {
            use super::*;

            #[test]
            fn it_calculates_unmasked_self_attention_head() {
                let q_tensor = Tensor::from_vec(vec![1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
                let k_tensor = Tensor::from_vec(vec![1, 2, 2], vec![3., 4., 5., 6.]).unwrap();
                let v_tensor =
                    Tensor::from_vec(vec![1, 2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                        .unwrap();

                let q_param = TensorValue::parameter(q_tensor).unwrap();
                let k_param = TensorValue::parameter(k_tensor).unwrap();
                let v_param = TensorValue::parameter(v_tensor).unwrap();

                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();

                assert_eq!(
                    result.raw_scores.value().clone(),
                    Tensor::from_vec(vec![1, 2, 2], vec![11.0, 17.0, 25.0, 39.0]).unwrap()
                );
                assert_eq!(
                    result.scaled_scores.value().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 2],
                        vec![
                            7.778174593052022,
                            12.020815280171307,
                            17.677669529663685,
                            27.577164466275352
                        ]
                    )
                    .unwrap()
                );
                assert_eq!(
                    result.weights.value().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 2],
                        vec![
                            0.014166035876688398,
                            0.9858339641233116,
                            5.019750993518885e-5,
                            0.9999498024900648
                        ]
                    )
                    .unwrap()
                );
                assert_eq!(
                    result.output.value().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 4],
                        vec![
                            0.4943335856493246,
                            0.5943335856493246,
                            0.6943335856493247,
                            0.7943335856493247,
                            0.49997992099602595,
                            0.5999799209960259,
                            0.6999799209960258,
                            0.799979920996026
                        ]
                    )
                    .unwrap()
                );
                assert_eq!(result.scale, 0.7071067811865475);
                assert_eq!(result.key_width, 2);
                assert_eq!(result.value_width, 4);
            }
        }

        mod gradient_calculation {
            use super::*;
            use crate::support::gradcheck::{assert_sampled_gradient, backward_with_test_seed};

            #[test]
            fn it_calculates_gradients() {
                let q_tensor = Tensor::from_vec(vec![1, 2, 1], vec![1., 2.]).unwrap();
                let k_tensor = Tensor::from_vec(vec![1, 2, 1], vec![3., 4.]).unwrap();
                let v_tensor = Tensor::from_vec(vec![1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();

                let q_param = TensorValue::parameter(q_tensor.clone()).unwrap();
                let k_param = TensorValue::parameter(k_tensor.clone()).unwrap();
                let v_param = TensorValue::parameter(v_tensor.clone()).unwrap();

                let context = AutogradContext::recording();
                let result = scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                let seed = backward_with_test_seed(&result.output);

                assert_eq!(
                    q_param.gradient().unwrap().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 1],
                        vec![0.02949178998622222, 0.03674775489122725]
                    )
                    .unwrap()
                );
                assert_eq!(
                    k_param.gradient().unwrap().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 1],
                        vec![-0.10298729976867685, 0.10298729976867682]
                    )
                    .unwrap()
                );
                assert_eq!(
                    v_param.gradient().unwrap().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 2],
                        vec![
                            0.15663754685908693,
                            0.25367363270711507,
                            0.8433624531409131,
                            1.2463263672928848
                        ]
                    )
                    .unwrap()
                );

                // Query gradient check
                assert_sampled_gradient(&q_param, &q_tensor, &seed, |candidate| {
                    let candidate_param = TensorValue::parameter(candidate.clone()).unwrap();
                    scaled_dot_product_self_attention_with_context(
                        context,
                        &candidate_param,
                        &k_param,
                        &v_param,
                    )
                    .unwrap()
                    .output
                });

                // Key gradient check
                assert_sampled_gradient(&k_param, &k_tensor, &seed, |candidate| {
                    let candidate_param = TensorValue::parameter(candidate.clone()).unwrap();
                    scaled_dot_product_self_attention_with_context(
                        context,
                        &q_param,
                        &candidate_param,
                        &v_param,
                    )
                    .unwrap()
                    .output
                });

                // Value gradient check
                assert_sampled_gradient(&v_param, &v_tensor, &seed, |candidate| {
                    let candidate_param = TensorValue::parameter(candidate.clone()).unwrap();
                    scaled_dot_product_self_attention_with_context(
                        context,
                        &q_param,
                        &k_param,
                        &candidate_param,
                    )
                    .unwrap()
                    .output
                });
            }
        }
    }
}
