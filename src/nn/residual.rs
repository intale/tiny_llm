use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};

/// An exact-shape residual merge that could not be formed.
#[derive(Clone, Debug, PartialEq)]
pub enum ResidualError {
    ShapeMismatch {
        identity: Vec<usize>,
        branch: Vec<usize>,
    },
    Autodiff(TensorAutodiffError),
}

impl fmt::Display for ResidualError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { identity, branch } => write!(
                formatter,
                "residual paths must have exactly equal shapes; identity {identity:?}, branch {branch:?}"
            ),
            Self::Autodiff(error) => write!(formatter, "residual addition failed: {error}"),
        }
    }
}

impl Error for ResidualError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ShapeMismatch { .. } => None,
            Self::Autodiff(error) => Some(error),
        }
    }
}

impl From<TensorAutodiffError> for ResidualError {
    fn from(error: TensorAutodiffError) -> Self {
        Self::Autodiff(error)
    }
}

impl TensorValue {
    /// Adds the identity input to this residual branch output.
    ///
    /// Unlike ordinary tensor addition, residual addition requires both operands to have exactly
    /// the same shape. Broadcasting is not allowed.
    ///
    /// The result is `self + identity`, preserving both operands' autograd dependencies so their
    /// gradient contributions are accumulated.
    pub fn residual_add(&self, identity: &TensorValue) -> Result<TensorValue, ResidualError> {
        self.residual_add_with_context(AutogradContext::recording(), identity)
    }

    /// Adds the identity input under the caller's explicit recording policy.
    pub fn residual_add_with_context(
        &self,
        context: AutogradContext,
        identity: &TensorValue,
    ) -> Result<TensorValue, ResidualError> {
        let identity_shape = identity.shape();
        let branch_shape = self.shape();
        if identity_shape != branch_shape {
            return Err(ResidualError::ShapeMismatch {
                identity: identity_shape,
                branch: branch_shape,
            });
        }

        Ok(self.add_with_context(context, identity)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::storage::Tensor;

    mod tensor_value {
        use super::*;

        mod fn_residual_add_with_context {
            use super::*;

            mod when_identity_tensor_shape_differs {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let branch_output_tensor = Tensor::from_vec(vec![2, 3], vec![0.; 6]).unwrap();
                    let identity_tensor = Tensor::from_vec(vec![3, 2], vec![0.; 6]).unwrap();
                    let branch_output_param =
                        TensorValue::parameter(branch_output_tensor.clone()).unwrap();
                    let identity_param = TensorValue::parameter(identity_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        branch_output_param.residual_add_with_context(context, &identity_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        ResidualError::ShapeMismatch {
                            identity: identity_tensor.shape().to_vec(),
                            branch: branch_output_param.shape().to_vec()
                        }
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;
                use crate::autograd::tensor_core::{ParentEdgeTest, TensorOperation};

                #[test]
                fn it_adds_identity_tensor() {
                    let branch_output_tensor = Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap();
                    let identity_tensor = Tensor::from_vec(vec![1, 2], vec![3., 4.]).unwrap();
                    let branch_output_param =
                        TensorValue::parameter(branch_output_tensor.clone()).unwrap();
                    let identity_param = TensorValue::parameter(identity_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        branch_output_param.residual_add_with_context(context, &identity_param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = &result.as_ref().unwrap();
                    let expected_tensor = Tensor::from_vec(vec![1, 2], vec![4., 6.]).unwrap();
                    assert_eq!(result.value().clone(), expected_tensor.clone());
                    assert_eq!(result.operation(), TensorOperation::Add);
                }
            }

            mod gradient_calculation {
                use super::*;
                use crate::support::gradcheck::{assert_sampled_gradient, backward_with_test_seed};

                #[test]
                fn it_calculates_gradient_correctly() {
                    let branch_output_tensor = Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap();
                    let identity_tensor = Tensor::from_vec(vec![1, 2], vec![3., 4.]).unwrap();
                    let branch_output_param =
                        TensorValue::parameter(branch_output_tensor.clone()).unwrap();
                    let identity_param = TensorValue::parameter(identity_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        branch_output_param.residual_add_with_context(context, &identity_param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    let seed = backward_with_test_seed(result);
                    assert_sampled_gradient(
                        &branch_output_param,
                        &branch_output_tensor,
                        &seed,
                        |candidate| {
                            let candidate = TensorValue::parameter(candidate.clone()).unwrap();
                            let right = TensorValue::constant(identity_tensor.clone()).unwrap();
                            candidate.residual_add_with_context(context, &right).unwrap()
                        },
                    );
                }
            }
        }
    }
}
