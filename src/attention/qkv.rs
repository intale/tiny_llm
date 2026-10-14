//! Three independent, bias-free projections for one self-attention input

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::nn::linear::{Linear, LinearError};

/// The projection branch that rejected construction or a delegated operation
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QkvProjection {
    Query,
    Key,
    Value,
}

impl fmt::Display for QkvProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
        })
    }
}

/// A rejected Q/K/V parameter set or hidden-state input.
#[derive(Clone, Debug, PartialEq)]
pub enum QkvError {
    Projection {
        projection: QkvProjection,
        source: LinearError,
    },
    InputRank {
        rank: usize,
    },
    InputWidthMismatch {
        expected: usize,
        actual: usize,
    },
    BranchInputWidthMismatch {
        query: usize,
        key: usize,
        value: usize,
    },
    BranchOutputWidthMismatch {
        query: usize,
        key: usize,
        value: usize,
    },
    Initialization(InitializationError),
}

impl fmt::Display for QkvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection { projection, source } => {
                write!(formatter, "{projection} projection: {source}")
            }
            Self::InputRank { rank } => write!(
                formatter,
                "Q/K/V input must have rank three [batch, tokens, model_width], got rank {rank}"
            ),
            Self::InputWidthMismatch { expected, actual } => write!(
                formatter,
                "Q/K/V input final width must equal model width {expected}, got {actual}"
            ),
            Self::BranchInputWidthMismatch { query, key, value } => write!(
                formatter,
                "Q/K/V model widths must match, got query {query}, key {key}, value {value}"
            ),
            Self::BranchOutputWidthMismatch { query, key, value } => write!(
                formatter,
                "Q/K/V head widths must match, got query {query}, key {key}, value {value}"
            ),
            Self::Initialization(source) => source.fmt(formatter),
        }
    }
}

impl Error for QkvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Projection { source, .. } => Some(source),
            Self::Initialization(source) => Some(source),
            _ => None,
        }
    }
}

impl From<InitializationError> for QkvError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

fn projection_error(projection: QkvProjection) -> impl FnOnce(LinearError) -> QkvError {
    move |source| QkvError::Projection { projection, source }
}

/// The three projected views of the same branch and token positions
#[derive(Clone, Debug)]
pub struct QkvForward {
    query: TensorValue,
    key: TensorValue,
    value: TensorValue,
}

impl QkvForward {
    pub fn query(&self) -> &TensorValue {
        &self.query
    }

    pub fn key(&self) -> &TensorValue {
        &self.key
    }

    pub fn value(&self) -> &TensorValue {
        &self.value
    }

    pub fn into_parts(self) -> (TensorValue, TensorValue, TensorValue) {
        (self.query, self.key, self.value)
    }
}

/// Three independent `[model_width, head_width]` linear maps with no biases
#[derive(Clone, Debug)]
pub struct QkvProjections {
    query: Linear,
    key: Linear,
    value: Linear,
    parameters: NamedParameters,
    model_width: usize,
    head_width: usize,
}

impl QkvProjections {
    /// Initializes Q, K and V in that order without partially advancing `rng`
    pub fn new(
        parameter_prefix: impl Into<String>,
        model_width: usize,
        head_width: usize,
        rng: &mut SplitMix64,
    ) -> Result<Self, QkvError> {
        let parameter_prefix = parameter_prefix.into();
        let mut trial = rng.clone();
        let query = Linear::new(
            format!("{parameter_prefix}.query"),
            model_width,
            head_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(QkvProjection::Query))?;
        let key = Linear::new(
            format!("{parameter_prefix}.key"),
            model_width,
            head_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(QkvProjection::Key))?;
        let value = Linear::new(
            format!("{parameter_prefix}.value"),
            model_width,
            head_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(QkvProjection::Value))?;
        let projections = Self::from_layers(query, key, value)?;
        *rng = trial;
        Ok(projections)
    }

    /// Gives Q/K/V semantics to three existing matrix parameters
    pub fn from_weights(
        query_weight: NamedParameter,
        key_weight: NamedParameter,
        value_weight: NamedParameter,
    ) -> Result<Self, QkvError> {
        let query = Linear::from_parameters(query_weight, None)
            .map_err(projection_error(QkvProjection::Query))?;
        let key = Linear::from_parameters(key_weight, None)
            .map_err(projection_error(QkvProjection::Key))?;
        let value = Linear::from_parameters(value_weight, None)
            .map_err(projection_error(QkvProjection::Value))?;
        Self::from_layers(query, key, value)
    }

    fn from_layers(query: Linear, key: Linear, value: Linear) -> Result<Self, QkvError> {
        let input_width = (query.input_width(), key.input_width(), value.input_width());
        if input_width.0 != input_width.1 || input_width.0 != input_width.2 {
            return Err(QkvError::BranchInputWidthMismatch {
                query: input_width.0,
                key: input_width.1,
                value: input_width.2,
            });
        }

        let output_width = (
            query.output_width(),
            key.output_width(),
            value.output_width(),
        );
        if output_width.0 != output_width.1 || output_width.0 != output_width.2 {
            return Err(QkvError::BranchOutputWidthMismatch {
                query: output_width.0,
                key: output_width.1,
                value: output_width.2,
            });
        }

        let parameters = NamedParameters::try_new(vec![
            query.weight().clone(),
            key.weight().clone(),
            value.weight().clone(),
        ])?;

        Ok(Self {
            query,
            key,
            value,
            parameters,
            model_width: input_width.0,
            head_width: output_width.0,
        })
    }

    /// Projects exactly `[batch, tokens, model_width]` into three head-width views
    pub fn forward(&self, input: &TensorValue) -> Result<QkvForward, QkvError> {
        self.forward_with_context(AutogradContext::recording(), input)
    }

    /// Projects Q, K and V under the caller's explicit recording policy
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
    ) -> Result<QkvForward, QkvError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(QkvError::InputRank { rank: shape.len() });
        }
        if shape[2] != self.model_width {
            return Err(QkvError::InputWidthMismatch {
                expected: self.model_width,
                actual: shape[2],
            });
        }

        let query = self
            .query
            .forward_with_context(context, input)
            .map_err(projection_error(QkvProjection::Query))?;
        let key = self
            .key
            .forward_with_context(context, input)
            .map_err(projection_error(QkvProjection::Key))?;
        let value = self
            .value
            .forward_with_context(context, input)
            .map_err(projection_error(QkvProjection::Value))?;

        Ok(QkvForward { query, key, value })
    }

    pub fn query(&self) -> &Linear {
        &self.query
    }

    pub fn key(&self) -> &Linear {
        &self.key
    }

    pub fn value(&self) -> &Linear {
        &self.value
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn model_width(&self) -> usize {
        self.model_width
    }

    pub fn head_width(&self) -> usize {
        self.head_width
    }

    pub fn parameter_count(&self) -> usize {
        3 * self.model_width * self.head_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod qkv_projections {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_prefix_is_invalid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let mut rng = SplitMix64::from_seed(0);
                    let result = QkvProjections::new("", 2, 3, &mut rng);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        QkvError::Projection {
                            projection: QkvProjection::Query,
                            source: LinearError::Initialization(
                                InitializationError::EmptyNameSegment { index: 0 }
                            )
                        }
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_qkv_projection() {
                    let mut rng = SplitMix64::from_seed(0);
                    let result = QkvProjections::new("foo", 2, 3, &mut rng);

                    assert!(result.is_ok(), "{result:?}");
                }
            }
        }

        mod fn_forward_with_context {
            use super::*;

            mod when_input_rank_is_not_3 {
                use super::*;
                use crate::tensor::storage::Tensor;

                #[test]
                fn it_returns_error() {
                    let mut rng = SplitMix64::from_seed(0);
                    let qkv_projections = QkvProjections::new("foo", 2, 3, &mut rng).unwrap();
                    let input_param =
                        TensorValue::parameter(Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap())
                            .unwrap();
                    let context = AutogradContext::recording();
                    let result = qkv_projections.forward_with_context(context, &input_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), QkvError::InputRank { rank: 2 });
                }
            }

            mod when_model_width_mismatch {
                use super::*;
                use crate::tensor::storage::Tensor;

                #[test]
                fn it_returns_error() {
                    let mut rng = SplitMix64::from_seed(0);
                    let qkv_projections = QkvProjections::new("foo", 2, 3, &mut rng).unwrap();
                    let input_param = TensorValue::parameter(
                        Tensor::from_vec(vec![2, 1, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
                    )
                    .unwrap();
                    let context = AutogradContext::recording();
                    let result = qkv_projections.forward_with_context(context, &input_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        QkvError::InputWidthMismatch {
                            expected: 2,
                            actual: 3,
                        }
                    );
                }
            }

            mod feed_forward_calculation {
                use super::*;
                use crate::tensor::storage::Tensor;

                #[test]
                fn it_calculates_qkv() {
                    let q_tensor =
                        Tensor::from_vec(vec![3, 2], vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1]).unwrap();
                    let k_tensor =
                        Tensor::from_vec(vec![3, 2], vec![1.2, 2.2, 3.2, 4.2, 5.2, 6.2]).unwrap();
                    let v_tensor =
                        Tensor::from_vec(vec![3, 2], vec![1.3, 2.3, 3.3, 4.3, 5.3, 6.3]).unwrap();

                    let qkv_projections = QkvProjections::from_weights(
                        NamedParameter::from_tensor("query", q_tensor.clone()).unwrap(),
                        NamedParameter::from_tensor("key", k_tensor.clone()).unwrap(),
                        NamedParameter::from_tensor("value", v_tensor.clone()).unwrap(),
                    )
                    .unwrap();

                    let context = AutogradContext::recording();
                    let input_tensor =
                        Tensor::from_vec(vec![1, 2, 3], vec![10., 20., 30., 40., 50., 60.])
                            .unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();

                    let result = qkv_projections.forward_with_context(context, &input_param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(
                        result.query.value().clone(),
                        Tensor::from_vec(
                            vec![1, 2, 2],
                            vec![
                                q_tensor.as_slice()[0] * input_tensor.as_slice()[0]
                                    + q_tensor.as_slice()[2] * input_tensor.as_slice()[1]
                                    + q_tensor.as_slice()[4] * input_tensor.as_slice()[2],
                                q_tensor.as_slice()[1] * input_tensor.as_slice()[0]
                                    + q_tensor.as_slice()[3] * input_tensor.as_slice()[1]
                                    + q_tensor.as_slice()[5] * input_tensor.as_slice()[2],
                                q_tensor.as_slice()[0] * input_tensor.as_slice()[3]
                                    + q_tensor.as_slice()[2] * input_tensor.as_slice()[4]
                                    + q_tensor.as_slice()[4] * input_tensor.as_slice()[5],
                                q_tensor.as_slice()[1] * input_tensor.as_slice()[3]
                                    + q_tensor.as_slice()[3] * input_tensor.as_slice()[4]
                                    + q_tensor.as_slice()[5] * input_tensor.as_slice()[5],
                            ]
                        )
                        .unwrap()
                    );
                    assert_eq!(
                        result.key.value().clone(),
                        Tensor::from_vec(
                            vec![1, 2, 2],
                            vec![
                                k_tensor.as_slice()[0] * input_tensor.as_slice()[0]
                                    + k_tensor.as_slice()[2] * input_tensor.as_slice()[1]
                                    + k_tensor.as_slice()[4] * input_tensor.as_slice()[2],
                                k_tensor.as_slice()[1] * input_tensor.as_slice()[0]
                                    + k_tensor.as_slice()[3] * input_tensor.as_slice()[1]
                                    + k_tensor.as_slice()[5] * input_tensor.as_slice()[2],
                                k_tensor.as_slice()[0] * input_tensor.as_slice()[3]
                                    + k_tensor.as_slice()[2] * input_tensor.as_slice()[4]
                                    + k_tensor.as_slice()[4] * input_tensor.as_slice()[5],
                                k_tensor.as_slice()[1] * input_tensor.as_slice()[3]
                                    + k_tensor.as_slice()[3] * input_tensor.as_slice()[4]
                                    + k_tensor.as_slice()[5] * input_tensor.as_slice()[5],
                            ]
                        )
                        .unwrap()
                    );
                    assert_eq!(
                        result.value.value().clone(),
                        Tensor::from_vec(
                            vec![1, 2, 2],
                            vec![
                                v_tensor.as_slice()[0] * input_tensor.as_slice()[0]
                                    + v_tensor.as_slice()[2] * input_tensor.as_slice()[1]
                                    + v_tensor.as_slice()[4] * input_tensor.as_slice()[2],
                                v_tensor.as_slice()[1] * input_tensor.as_slice()[0]
                                    + v_tensor.as_slice()[3] * input_tensor.as_slice()[1]
                                    + v_tensor.as_slice()[5] * input_tensor.as_slice()[2],
                                v_tensor.as_slice()[0] * input_tensor.as_slice()[3]
                                    + v_tensor.as_slice()[2] * input_tensor.as_slice()[4]
                                    + v_tensor.as_slice()[4] * input_tensor.as_slice()[5],
                                v_tensor.as_slice()[1] * input_tensor.as_slice()[3]
                                    + v_tensor.as_slice()[3] * input_tensor.as_slice()[4]
                                    + v_tensor.as_slice()[5] * input_tensor.as_slice()[5],
                            ]
                        )
                        .unwrap()
                    );
                }
            }

            mod gradient_calculation {
                use super::*;
                use crate::support::gradcheck::{assert_sampled_gradient, backward_with_test_seed};
                use crate::tensor::storage::Tensor;

                #[test]
                fn it_calculates_gradients_on_backward_pass() {
                    let q_tensor = Tensor::from_vec(vec![3, 1], vec![1.1, 2.1, 3.1]).unwrap();
                    let k_tensor = Tensor::from_vec(vec![3, 1], vec![1.2, 2.2, 3.2]).unwrap();
                    let v_tensor = Tensor::from_vec(vec![3, 1], vec![1.3, 2.3, 3.3]).unwrap();

                    let qkv_projections = QkvProjections::from_weights(
                        NamedParameter::from_tensor("query", q_tensor.clone()).unwrap(),
                        NamedParameter::from_tensor("key", k_tensor.clone()).unwrap(),
                        NamedParameter::from_tensor("value", v_tensor.clone()).unwrap(),
                    )
                    .unwrap();

                    let context = AutogradContext::recording();
                    let input_tensor =
                        Tensor::from_vec(vec![1, 1, 3], vec![10., 20., 30.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();

                    let result = qkv_projections.forward_with_context(context, &input_param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();

                    // Query gradient check
                    let q_seed = backward_with_test_seed(&result.query);
                    assert_sampled_gradient(
                        qkv_projections.query.weight().tensor(),
                        &q_tensor,
                        &q_seed,
                        |candidate| {
                            let right = TensorValue::constant(input_tensor.clone()).unwrap();
                            let proj = QkvProjections::from_weights(
                                NamedParameter::from_tensor("query", candidate.clone()).unwrap(),
                                NamedParameter::from_tensor("key", k_tensor.clone()).unwrap(),
                                NamedParameter::from_tensor("value", v_tensor.clone()).unwrap(),
                            )
                            .unwrap();
                            proj.forward_with_context(AutogradContext::recording(), &right)
                                .unwrap()
                                .query
                        },
                    );

                    // Key gradient check
                    let k_seed = backward_with_test_seed(&result.key);
                    assert_sampled_gradient(
                        qkv_projections.key.weight().tensor(),
                        &k_tensor,
                        &k_seed,
                        |candidate| {
                            let right = TensorValue::constant(input_tensor.clone()).unwrap();
                            let proj = QkvProjections::from_weights(
                                NamedParameter::from_tensor("query", q_tensor.clone()).unwrap(),
                                NamedParameter::from_tensor("key", candidate.clone()).unwrap(),
                                NamedParameter::from_tensor("value", v_tensor.clone()).unwrap(),
                            )
                            .unwrap();
                            proj.forward_with_context(AutogradContext::recording(), &right)
                                .unwrap()
                                .key
                        },
                    );

                    input_param.zero_grad().unwrap();

                    // Value gradient check
                    let v_seed = backward_with_test_seed(&result.value);
                    assert_sampled_gradient(
                        qkv_projections.key.weight().tensor(),
                        &v_tensor,
                        &v_seed,
                        |candidate| {
                            let right = TensorValue::constant(input_tensor.clone()).unwrap();
                            let proj = QkvProjections::from_weights(
                                NamedParameter::from_tensor("query", q_tensor.clone()).unwrap(),
                                NamedParameter::from_tensor("key", k_tensor.clone()).unwrap(),
                                NamedParameter::from_tensor("value", candidate.clone()).unwrap(),
                            )
                            .unwrap();
                            proj.forward_with_context(AutogradContext::recording(), &right)
                                .unwrap()
                                .value
                        },
                    );

                    assert_eq!(
                        input_param.gradient().unwrap().as_slice(),
                        vec![1.3, 2.3, 3.3]
                    );
                    assert_eq!(
                        qkv_projections
                            .query
                            .weight()
                            .tensor()
                            .gradient()
                            .unwrap()
                            .as_slice(),
                        vec![10.0, 20.0, 30.0]
                    );
                    assert_eq!(
                        qkv_projections
                            .key
                            .weight()
                            .tensor()
                            .gradient()
                            .unwrap()
                            .as_slice(),
                        vec![10.0, 20.0, 30.0]
                    );
                    assert_eq!(
                        qkv_projections
                            .value
                            .weight()
                            .tensor()
                            .gradient()
                            .unwrap()
                            .as_slice(),
                        vec![10.0, 20.0, 30.0]
                    );
                }
            }
        }
    }
}
