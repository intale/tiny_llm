//! A named trainable projection over the final feature axis.

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::tensor::storage::Tensor;

/// A rejected parameter set, input shape, allocation, or delegated tape operation.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearError {
    Initialization(InitializationError),
    Autodiff(TensorAutodiffError),
    WeightRank { rank: usize },
    ZeroInputWidth,
    ZeroOutputWidth,
    BiasRank { rank: usize },
    BiasWidthMismatch { expected: usize, actual: usize },
    InputRank { rank: usize },
    InputWidthMismatch { expected: usize, actual: usize },
    BiasAllocationFailed { elements: usize },
}

impl fmt::Display for LinearError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization(error) => error.fmt(formatter),
            Self::Autodiff(error) => error.fmt(formatter),
            Self::WeightRank { rank } => {
                write!(
                    formatter,
                    "linear weight must have rank two, got rank {rank}"
                )
            }
            Self::ZeroInputWidth => {
                formatter.write_str("linear input width must be greater than zero")
            }
            Self::ZeroOutputWidth => {
                formatter.write_str("linear output width must be greater than zero")
            }
            Self::BiasRank { rank } => {
                write!(formatter, "linear bias must have rank one, got rank {rank}")
            }
            Self::BiasWidthMismatch { expected, actual } => write!(
                formatter,
                "linear bias width must equal output width {expected}, got {actual}"
            ),
            Self::InputRank { rank } => write!(
                formatter,
                "linear input must have at least one feature axis, got rank {rank}"
            ),
            Self::InputWidthMismatch { expected, actual } => write!(
                formatter,
                "linear input final width must equal {expected}, got {actual}"
            ),
            Self::BiasAllocationFailed { elements } => write!(
                formatter,
                "could not reserve storage for {elements} linear bias values"
            ),
        }
    }
}

impl Error for LinearError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Initialization(error) => Some(error),
            Self::Autodiff(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InitializationError> for LinearError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

impl From<TensorAutodiffError> for LinearError {
    fn from(error: TensorAutodiffError) -> Self {
        Self::Autodiff(error)
    }
}

/// One `[input_width, output_width]` feature projection with optional bias
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    parameters: NamedParameters,
    input_width: usize,
    output_width: usize,
    has_bias: bool,
}

impl Linear {
    /// Initializes a named weight and optional zero bias without partially advancing `rng`
    pub fn new(
        parameter_prefix: impl Into<String>,
        input_width: usize,
        output_width: usize,
        with_bias: bool,
        rng: &mut SplitMix64,
    ) -> Result<Self, LinearError> {
        let parameter_prefix = parameter_prefix.into();
        let mut trial = rng.clone();
        let weight = NamedParameter::xavier_uniform(
            format!("{parameter_prefix}.weight"),
            input_width,
            output_width,
            &mut trial,
        )
        .map_err(|error| match error {
            InitializationError::ZeroFanIn => LinearError::ZeroInputWidth,
            InitializationError::ZeroFanOut => LinearError::ZeroOutputWidth,
            other => LinearError::Initialization(other),
        })?;

        let bias = if with_bias {
            let mut values = Vec::new();
            values.try_reserve_exact(output_width).map_err(|_| {
                LinearError::BiasAllocationFailed {
                    elements: output_width,
                }
            })?;
            values.resize(output_width, 0.0);
            let tensor = Tensor::from_vec(vec![output_width], values)
                .map_err(InitializationError::Tensor)?;
            Some(NamedParameter::from_tensor(
                format!("{parameter_prefix}.bias"),
                tensor,
            )?)
        } else {
            None
        };

        let layer = Self::from_parameters(weight, bias)?;
        *rng = trial;
        Ok(layer)
    }

    /// Gives layer semantics to an existing weight and optional bias
    pub fn from_parameters(
        weight: NamedParameter,
        bias: Option<NamedParameter>,
    ) -> Result<Self, LinearError> {
        let weight_shape = weight.tensor().shape();
        if weight_shape.len() != 2 {
            return Err(LinearError::WeightRank {
                rank: weight_shape.len(),
            });
        }

        let input_width = weight_shape[0];
        let output_width = weight_shape[1];
        if input_width == 0 {
            return Err(LinearError::ZeroInputWidth);
        }
        if output_width == 0 {
            return Err(LinearError::ZeroOutputWidth);
        }

        if let Some(parameter) = &bias {
            let bias_shape = parameter.tensor().shape();
            if bias_shape.len() != 1 {
                return Err(LinearError::BiasRank {
                    rank: bias_shape.len(),
                });
            }
            if bias_shape[0] != output_width {
                return Err(LinearError::BiasWidthMismatch {
                    expected: output_width,
                    actual: bias_shape[0],
                });
            }
        }

        let has_bias = bias.is_some();
        let mut parameters = vec![weight];
        if let Some(bias) = bias {
            parameters.push(bias);
        }

        Ok(Self {
            parameters: NamedParameters::try_new(parameters)?,
            input_width,
            output_width,
            has_bias,
        })
    }

    /// Projects only the final feature axis and preserves every leading axis
    pub fn forward(&self, input: &TensorValue) -> Result<TensorValue, LinearError> {
        self.forward_with_context(AutogradContext::recording(), input)
    }

    /// Projects the final feature axis under an explicit recording policy. Related formula is:
    /// Y = XW + b
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue, // our X
    ) -> Result<TensorValue, LinearError> {
        let input_shape = input.shape();
        if input_shape.is_empty() {
            return Err(LinearError::InputRank { rank: 0 });
        }

        let actual_width = *input_shape.last().expect("nonempty input shape");
        if actual_width != self.input_width {
            return Err(LinearError::InputWidthMismatch {
                expected: self.input_width,
                actual: actual_width,
            });
        }

        let projected = if input_shape.len() == 1 {
            let promoted = input.reshape_with_context(context, &[1, self.input_width])?;
            let output = promoted.matmul_with_context(context, self.weight().tensor())?;
            let output = match self.bias() {
                Some(bias) => output.add_with_context(context, bias.tensor())?,
                None => output,
            };
            output.reshape_with_context(context, &[self.output_width])?
        } else {
            let output = input.matmul_with_context(context, self.weight().tensor())?;
            match self.bias() {
                Some(bias) => output.add_with_context(context, bias.tensor())?,
                None => output,
            }
        };

        Ok(projected)
    }

    pub fn weight(&self) -> &NamedParameter {
        &self.parameters.as_slice()[0]
    }

    pub fn bias(&self) -> Option<&NamedParameter> {
        self.has_bias.then(|| &self.parameters.as_slice()[1])
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn has_bias(&self) -> bool {
        self.has_bias
    }

    pub fn parameter_count(&self) -> usize {
        let weight_count = self.input_width * self.output_width;
        if self.has_bias {
            weight_count + self.output_width
        } else {
            weight_count
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO_SEED_NEXT_U64: u64 = 16294208416658607535;
    const ZERO_SEED_NEXT_U64_AFTER_2X1_TENSOR: u64 = 487617019471545679;

    mod linear {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_error_raises {
                use super::*;

                #[test]
                fn it_does_not_change_rng_state() {
                    let param_prefix = "foo";
                    let input_width = 0;
                    let output_width = 2;
                    let with_bias = false;
                    let mut rng = SplitMix64::from_seed(0);
                    let result =
                        Linear::new(param_prefix, input_width, output_width, with_bias, &mut rng);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64);
                }
            }

            mod when_all_is_ok {
                use super::*;

                mod when_bias_is_not_required {
                    use super::*;

                    #[test]
                    fn it_computes_linear_and_persists_rng_changes() {
                        let param_prefix = "foo";
                        let input_width = 2;
                        let output_width = 1;
                        let with_bias = false;
                        let mut rng = SplitMix64::from_seed(0);
                        let result = Linear::new(
                            param_prefix,
                            input_width,
                            output_width,
                            with_bias,
                            &mut rng,
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        assert_eq!(result.input_width, input_width);
                        assert_eq!(result.output_width, output_width);
                        assert!(result.bias().is_none());
                        assert_eq!(
                            result.weight().tensor().value().clone(),
                            Tensor::from_vec(
                                vec![2, 1],
                                vec![1.0841666871598514, -0.1936680704336956]
                            )
                            .unwrap()
                        );
                        assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64_AFTER_2X1_TENSOR);
                    }
                }

                mod when_bias_is_required {
                    use super::*;

                    #[test]
                    fn it_computes_linear() {
                        let param_prefix = "foo";
                        let input_width = 2;
                        let output_width = 1;
                        let with_bias = true;
                        let mut rng = SplitMix64::from_seed(0);
                        let result = Linear::new(
                            param_prefix,
                            input_width,
                            output_width,
                            with_bias,
                            &mut rng,
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        assert_eq!(result.input_width, input_width);
                        assert_eq!(result.output_width, output_width);
                        assert!(result.bias().is_some());
                        assert_eq!(
                            result.bias().unwrap().tensor().value().clone(),
                            Tensor::from_vec(vec![output_width], vec![0.]).unwrap()
                        );
                        assert_eq!(result.weight().name(), "foo.weight");
                        assert_eq!(
                            result.weight().tensor().value().clone(),
                            Tensor::from_vec(
                                vec![2, 1],
                                vec![1.0841666871598514, -0.1936680704336956]
                            )
                            .unwrap()
                        );
                        assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64_AFTER_2X1_TENSOR);
                    }
                }
            }
        }

        mod fn_parameters {
            use super::*;

            mod when_weight_rank_is_not_2 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 2, 2], vec![0.; 4]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let bias = None;
                    let result = Linear::from_parameters(weight_param, bias);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), LinearError::WeightRank { rank: 3 })
                }
            }

            mod when_input_width_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![0, 2], vec![]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let bias = None;
                    let result = Linear::from_parameters(weight_param, bias);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), LinearError::ZeroInputWidth)
                }
            }

            mod when_output_width_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2, 0], vec![]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let bias = None;
                    let result = Linear::from_parameters(weight_param, bias);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), LinearError::ZeroOutputWidth)
                }
            }

            mod when_bias_is_absent {
                use super::*;

                #[test]
                fn it_computes_linear_without_bias() {
                    let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let bias = None;
                    let result = Linear::from_parameters(weight_param, bias);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.input_width, 2);
                    assert_eq!(result.output_width, 1);
                    assert_eq!(result.parameters.len(), 1);

                    assert_eq!(result.parameters.as_slice()[0].name(), "foo");
                    assert_eq!(
                        result.parameters.as_slice()[0].tensor().value().clone(),
                        tensor
                    );
                }
            }

            mod when_bias_is_present {
                use super::*;

                mod when_bias_rank_is_not_1 {
                    use super::*;

                    #[test]
                    fn it_returns_error() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let bias_tensor = Tensor::from_vec(vec![], vec![1.]).unwrap();
                        let bias = Some(
                            NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap(),
                        );
                        let result = Linear::from_parameters(weight_param, bias);

                        assert!(result.is_err(), "{result:?}");
                        assert_eq!(result.err().unwrap(), LinearError::BiasRank { rank: 0 });
                    }
                }

                mod when_bias_width_does_not_match_output_width {
                    use super::*;

                    #[test]
                    fn it_returns_error() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let bias_tensor = Tensor::from_vec(vec![2], vec![1., 2.]).unwrap();
                        let bias = Some(
                            NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap(),
                        );
                        let result = Linear::from_parameters(weight_param, bias);

                        assert!(result.is_err(), "{result:?}");
                        assert_eq!(
                            result.err().unwrap(),
                            LinearError::BiasWidthMismatch {
                                expected: 1,
                                actual: 2,
                            }
                        );
                    }
                }

                mod when_all_is_ok {
                    use super::*;

                    #[test]
                    fn it_computes_linear_without_bias() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let bias_tensor = Tensor::from_vec(vec![1], vec![3.]).unwrap();
                        let bias = Some(
                            NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap(),
                        );
                        let result = Linear::from_parameters(weight_param, bias);

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        assert_eq!(result.input_width, 2);
                        assert_eq!(result.output_width, 1);
                        assert_eq!(result.parameters.len(), 2);

                        assert_eq!(result.parameters.as_slice()[0].name(), "foo");
                        assert_eq!(
                            result.parameters.as_slice()[0].tensor().value().clone(),
                            tensor
                        );

                        assert_eq!(result.parameters.as_slice()[1].name(), "foo.bias");
                        assert_eq!(
                            result.parameters.as_slice()[1].tensor().value().clone(),
                            bias_tensor
                        );
                    }
                }
            }
        }

        mod fn_forward_with_context {
            use super::*;

            mod when_input_is_scalar {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let bias_tensor = Tensor::from_vec(vec![1], vec![3.]).unwrap();
                    let bias =
                        Some(NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap());
                    let linear = Linear::from_parameters(weight_param, bias).unwrap();
                    let context = AutogradContext::recording();
                    let input_tensor = Tensor::from_vec(vec![], vec![1.]).unwrap();
                    let result = linear.forward_with_context(
                        context,
                        &TensorValue::parameter(input_tensor.clone()).unwrap(),
                    );

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), LinearError::InputRank { rank: 0 });
                }
            }

            mod when_input_tensor_width_does_not_match_linear_input_width {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                    let weight_param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let bias_tensor = Tensor::from_vec(vec![1], vec![3.]).unwrap();
                    let bias =
                        Some(NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap());
                    let linear = Linear::from_parameters(weight_param, bias).unwrap();
                    let context = AutogradContext::recording();
                    let input_tensor = Tensor::from_vec(vec![3], vec![1., 2., 3.]).unwrap();
                    let result = linear.forward_with_context(
                        context,
                        &TensorValue::parameter(input_tensor.clone()).unwrap(),
                    );

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        LinearError::InputWidthMismatch {
                            expected: 2,
                            actual: 3,
                        }
                    );
                }
            }

            mod when_input_tensor_rank_is_1 {
                use super::*;

                mod when_bias_is_absent {
                    use super::*;
                    use crate::support::all_forward_ops;

                    #[test]
                    fn it_performs_forward_pass() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let linear = Linear::from_parameters(weight_param, None).unwrap();
                        let context = AutogradContext::recording();
                        let input_tensor = Tensor::from_vec(vec![2], vec![3., 4.]).unwrap();
                        let result = linear.forward_with_context(
                            context,
                            &TensorValue::parameter(input_tensor.clone()).unwrap(),
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        println!("{:?}", result.parents());
                        assert_eq!(
                            result.value().clone(),
                            Tensor::from_vec(vec![1], vec![1. * 3. + 2. * 4.]).unwrap()
                        );
                        assert_eq!(
                            all_forward_ops(result),
                            vec!["parameter", "reshape", "parameter", "matmul", "reshape"]
                        );
                    }
                }

                mod when_bias_is_present {
                    use super::*;
                    use crate::support::all_forward_ops;

                    #[test]
                    fn it_performs_forward_pass() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let bias_tensor = Tensor::from_vec(vec![1], vec![5.]).unwrap();
                        let bias = Some(
                            NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap(),
                        );
                        let linear = Linear::from_parameters(weight_param, bias).unwrap();
                        let context = AutogradContext::recording();
                        let input_tensor = Tensor::from_vec(vec![2], vec![3., 4.]).unwrap();
                        let result = linear.forward_with_context(
                            context,
                            &TensorValue::parameter(input_tensor.clone()).unwrap(),
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        println!("{:?}", result.parents());
                        assert_eq!(
                            result.value().clone(),
                            Tensor::from_vec(vec![1], vec![1. * 3. + 2. * 4. + 5.]).unwrap()
                        );
                        assert_eq!(
                            all_forward_ops(result),
                            vec![
                                "parameter",
                                "reshape",
                                "parameter",
                                "matmul",
                                "parameter",
                                "add",
                                "reshape"
                            ]
                        );
                    }
                }
            }

            mod when_input_tensor_rank_is_2 {
                use super::*;

                mod when_bias_is_absent {
                    use super::*;
                    use crate::support::all_forward_ops;

                    #[test]
                    fn it_performs_forward_pass() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let linear = Linear::from_parameters(weight_param, None).unwrap();
                        let context = AutogradContext::recording();
                        let input_tensor = Tensor::from_vec(vec![1, 2], vec![3., 4.]).unwrap();
                        let result = linear.forward_with_context(
                            context,
                            &TensorValue::parameter(input_tensor.clone()).unwrap(),
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        println!("{:?}", result.parents());
                        assert_eq!(
                            result.value().clone(),
                            Tensor::from_vec(vec![1, 1], vec![1. * 3. + 2. * 4.]).unwrap()
                        );
                        assert_eq!(
                            all_forward_ops(result),
                            vec!["parameter", "parameter", "matmul"]
                        );
                    }
                }

                mod when_bias_is_present {
                    use super::*;
                    use crate::support::all_forward_ops;

                    #[test]
                    fn it_performs_forward_pass() {
                        let tensor = Tensor::from_vec(vec![2, 1], vec![1., 2.]).unwrap();
                        let weight_param =
                            NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let bias_tensor = Tensor::from_vec(vec![1], vec![5.]).unwrap();
                        let bias = Some(
                            NamedParameter::from_tensor("foo.bias", bias_tensor.clone()).unwrap(),
                        );
                        let linear = Linear::from_parameters(weight_param, bias).unwrap();
                        let context = AutogradContext::recording();
                        let input_tensor = Tensor::from_vec(vec![1, 2], vec![3., 4.]).unwrap();
                        let result = linear.forward_with_context(
                            context,
                            &TensorValue::parameter(input_tensor.clone()).unwrap(),
                        );

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        println!("{:?}", result.parents());
                        assert_eq!(
                            result.value().clone(),
                            Tensor::from_vec(vec![1, 1], vec![1. * 3. + 2. * 4. + 5.]).unwrap()
                        );
                        assert_eq!(
                            all_forward_ops(result),
                            vec!["parameter", "parameter", "matmul", "parameter", "add"]
                        );
                    }
                }
            }
        }

        mod fn_parameter_count {
            use super::*;

            mod when_has_bias {
                use super::*;

                #[test]
                fn it_takes_into_account_its_data_size() {
                    let param_prefix = "foo";
                    let input_width = 3;
                    let output_width = 2;
                    let with_bias = true;
                    let mut rng = SplitMix64::from_seed(0);
                    let linear =
                        Linear::new(param_prefix, input_width, output_width, with_bias, &mut rng)
                            .unwrap();

                    assert_eq!(
                        linear.parameter_count(),
                        input_width * output_width + output_width
                    );
                }
            }

            mod when_does_not_have_bias {
                use super::*;

                #[test]
                fn it_does_not_take_into_account_bias_size() {
                    let param_prefix = "foo";
                    let input_width = 3;
                    let output_width = 2;
                    let with_bias = false;
                    let mut rng = SplitMix64::from_seed(0);
                    let linear =
                        Linear::new(param_prefix, input_width, output_width, with_bias, &mut rng)
                            .unwrap();

                    assert_eq!(linear.parameter_count(), input_width * output_width);
                }
            }
        }
    }
}
