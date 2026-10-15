//! Lower-triangular masking for autoregressive self-attention

use std::error::Error;
use std::fmt;

use super::self_attention::{SelfAttentionError, scaled_self_attention_scores_with_context};
use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::tensor::storage::{Tensor, TensorError};

/// The causal-only forward stage at which a cumulative operation failed
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CausalMaskingStage {
    MaskedSoftmax,
    ValueMixture,
}

impl fmt::Display for CausalMaskingStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MaskedSoftmax => "causal row softmax",
            Self::ValueMixture => "causal weighted value mixture",
        })
    }
}

/// A rejected causal mask, self-attention input, or cumulative operation.
#[derive(Clone, Debug, PartialEq)]
pub enum CausalMaskingError {
    SelfAttention(SelfAttentionError),
    MaskAllocationFailed {
        elements: usize,
    },
    MaskTensor(TensorError),
    Autodiff {
        stage: CausalMaskingStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for CausalMaskingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelfAttention(source) => source.fmt(formatter),
            Self::MaskAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate a causal mask for {elements} score cells"
            ),
            Self::MaskTensor(source) => write!(formatter, "causal mask tensor: {source}"),
            Self::Autodiff { stage, source } => {
                write!(formatter, "causal self-attention {stage}: {source}")
            }
        }
    }
}

impl Error for CausalMaskingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SelfAttention(source) => Some(source),
            Self::MaskTensor(source) => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            Self::MaskAllocationFailed { .. } => None,
        }
    }
}

impl From<SelfAttentionError> for CausalMaskingError {
    fn from(source: SelfAttentionError) -> Self {
        Self::SelfAttention(source)
    }
}

fn autodiff_error(
    stage: CausalMaskingStage,
) -> impl FnOnce(TensorAutodiffError) -> CausalMaskingError {
    move |source| CausalMaskingError::Autodiff { stage, source }
}

/// Builds an additive square mask with zero for `key <= query` and negative infinity for future
/// keys
pub fn causal_additive_mask(tokens: usize) -> Result<Tensor, CausalMaskingError> {
    let elements = tokens
        .checked_mul(tokens)
        .ok_or(CausalMaskingError::MaskTensor(TensorError::ShapeOverflow))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| CausalMaskingError::MaskAllocationFailed { elements })?;

    for query in 0..tokens {
        for key in 0..tokens {
            values.push(if key <= query { 0.0 } else { f64::NEG_INFINITY });
        }
    }

    Tensor::from_vec(vec![tokens, tokens], values).map_err(CausalMaskingError::MaskTensor)
}

/// Inspectable evidence from one causally masked attention head
#[derive(Clone, Debug)]
pub struct CausalSelfAttentionForward {
    raw_scores: TensorValue,
    scaled_scores: TensorValue,
    additive_mask: Tensor,
    weights: TensorValue,
    output: TensorValue,
    scale: f64,
    key_width: usize,
    value_width: usize,
}

impl CausalSelfAttentionForward {
    pub fn raw_scores(&self) -> &TensorValue {
        &self.raw_scores
    }

    pub fn dot_products(&self) -> &TensorValue {
        &self.raw_scores
    }

    pub fn scaled_scores(&self) -> &TensorValue {
        &self.scaled_scores
    }

    /// The plain additive mask. It is intentionally not a tape value because its blocked cells
    /// contain negative infinity.
    pub fn additive_mask(&self) -> &Tensor {
        &self.additive_mask
    }

    pub fn weights(&self) -> &TensorValue {
        &self.weights
    }

    pub fn probabilities(&self) -> &TensorValue {
        &self.weights
    }

    pub fn output(&self) -> &TensorValue {
        &self.output
    }

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
}

/// Computes one scaled self-attention head with an inclusive-prefix mask
pub fn causal_scaled_dot_produce_self_attention(
    query: &TensorValue,
    key: &TensorValue,
    value: &TensorValue,
) -> Result<CausalSelfAttentionForward, CausalMaskingError> {
    causal_scaled_dot_product_self_attention_with_context(
        AutogradContext::recording(),
        query,
        key,
        value,
    )
}

/// Computes causal attention under the caller's explicit recording policy.
pub fn causal_scaled_dot_product_self_attention_with_context(
    context: AutogradContext,
    query: &TensorValue,
    key: &TensorValue,
    value: &TensorValue,
) -> Result<CausalSelfAttentionForward, CausalMaskingError> {
    let prepared = scaled_self_attention_scores_with_context(context, query, key, value)?;
    let tokens = query.shape()[1];
    let additive_mask = causal_additive_mask(tokens)?;
    let weights = prepared
        .scaled_scores
        .causal_softmax_with_context(context)
        .map_err(autodiff_error(CausalMaskingStage::MaskedSoftmax))?;
    let output = weights
        .matmul_with_context(context, value)
        .map_err(autodiff_error(CausalMaskingStage::ValueMixture))?;

    Ok(CausalSelfAttentionForward {
        raw_scores: prepared.raw_scores,
        scaled_scores: prepared.scaled_scores,
        additive_mask,
        weights,
        output,
        scale: prepared.scale,
        key_width: prepared.key_width,
        value_width: prepared.value_width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fn_causal_scaled_dot_product_self_attention_with_context {
        use super::*;

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
                let result = causal_scaled_dot_product_self_attention_with_context(
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
                        vec![1.0, 0.0, 5.01975099351888e-5, 0.9999498024900648]
                    )
                    .unwrap()
                );
                assert_eq!(
                    result.output.value().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 4],
                        vec![
                            0.1,
                            0.2,
                            0.3,
                            0.4,
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
                let result = causal_scaled_dot_product_self_attention_with_context(
                    context, &q_param, &k_param, &v_param,
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                let seed = backward_with_test_seed(&result.output);

                assert_eq!(
                    q_param.gradient().unwrap().clone(),
                    Tensor::from_vec(vec![1, 2, 1], vec![0.0, 0.03674775489122728]).unwrap()
                );
                assert_eq!(
                    k_param.gradient().unwrap().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 1],
                        vec![-0.07349550978245455, 0.07349550978245455]
                    )
                    .unwrap()
                );
                assert_eq!(
                    v_param.gradient().unwrap().clone(),
                    Tensor::from_vec(
                        vec![1, 2, 2],
                        vec![
                            0.3394021915165881,
                            0.6192029220221176,
                            0.6605978084834118,
                            0.8807970779778823
                        ]
                    )
                    .unwrap()
                );

                // Query gradient check
                assert_sampled_gradient(&q_param, &q_tensor, &seed, |candidate| {
                    let candidate_param = TensorValue::parameter(candidate.clone()).unwrap();
                    causal_scaled_dot_product_self_attention_with_context(
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
                    causal_scaled_dot_product_self_attention_with_context(
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
                    causal_scaled_dot_product_self_attention_with_context(
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
