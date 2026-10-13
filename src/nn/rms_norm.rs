//! Differentiable root-mean-square normalization over the final feature axis.

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, validate_name};
use crate::tensor::storage::Tensor;

const DEFAULT_GAIN_VALUE: f64 = 1.0;

/// The composed tape stage that rejected an RMSNorm forward pass
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormStage {
    Square,
    MeanSquare,
    EpsilonConstant,
    Stabilize,
    LogMeanSquare,
    ExponentConstant,
    ScaleLog,
    ReciprocalRoot,
    Normalize,
    ApplyGain,
}

impl fmt::Display for RmsNormStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Square => "square input",
            Self::MeanSquare => "mean square",
            Self::EpsilonConstant => "epsilon constant",
            Self::Stabilize => "add epsilon",
            Self::LogMeanSquare => "log stabilized mean square",
            Self::ExponentConstant => "negative-half exponent constant",
            Self::ScaleLog => "scale logarithm",
            Self::ReciprocalRoot => "reciprocal square root",
            Self::Normalize => "normalize input",
            Self::ApplyGain => "apply learned gain",
        })
    }
}

/// A rejected RMSNorm configuration, input, or delegated tape operation.
#[derive(Clone, Debug, PartialEq)]
pub enum RmsNormError {
    InvalidEpsilon {
        value: f64,
    },
    EmptyFeatureWidth,
    GainRank {
        shape: Vec<usize>,
    },
    GainAllocationFailed {
        elements: usize,
    },
    InputRankZero,
    InputWidthMismatch {
        expected: usize,
        actual: usize,
    },
    ZeroEnergyRow {
        row: usize,
    },
    Initialization(InitializationError),
    Autodiff {
        stage: RmsNormStage,
        source: TensorAutodiffError,
    },
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEpsilon { value } => write!(
                formatter,
                "RMSNorm epsilon must be finite and nonnegative, got {value:?}"
            ),
            Self::EmptyFeatureWidth => {
                formatter.write_str("RMSNorm feature width must be greater than zero")
            }
            Self::GainRank { shape } => {
                write!(
                    formatter,
                    "RMSNorm gain must have rank 1, got shape {shape:?}"
                )
            }
            Self::GainAllocationFailed { elements } => write!(
                formatter,
                "could not reserve storage for {elements} RMSNorm gain values"
            ),
            Self::InputRankZero => formatter.write_str("RMSNorm input must have at least one axis"),
            Self::InputWidthMismatch { expected, actual } => write!(
                formatter,
                "RMSNorm input feature width must be {expected}, got {actual}"
            ),
            Self::ZeroEnergyRow { row } => write!(
                formatter,
                "RMSNorm epsilon is zero but feature row {row} has zero mean square"
            ),
            Self::Initialization(source) => source.fmt(formatter),
            Self::Autodiff { stage, source } => {
                write!(formatter, "RMSNorm {stage}: {source}")
            }
        }
    }
}

impl Error for RmsNormError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Initialization(source) => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<InitializationError> for RmsNormError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

fn autodiff_error(stage: RmsNormStage) -> impl FnOnce(TensorAutodiffError) -> RmsNormError {
    move |source| RmsNormError::Autodiff { stage, source }
}

fn validate_epsilon(epsilon: f64) -> Result<(), RmsNormError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(RmsNormError::InvalidEpsilon { value: epsilon });
    }
    Ok(())
}

fn scalar_constant(value: f64, stage: RmsNormStage) -> Result<TensorValue, RmsNormError> {
    let tensor = Tensor::from_vec(Vec::new(), vec![value])
        .map_err(TensorAutodiffError::from)
        .map_err(autodiff_error(stage))?;
    TensorValue::constant(tensor).map_err(autodiff_error(stage))
}

/// Inspectable tensors from one composed RMSNorm forward pass
#[derive(Clone, Debug)]
pub struct RmsNormForward {
    mean_square: TensorValue,
    inverse_rms: TensorValue,
    normalized: TensorValue,
    output: TensorValue,
}

impl RmsNormForward {
    pub fn mean_square(&self) -> &TensorValue {
        &self.mean_square
    }

    pub fn inverse_rms(&self) -> &TensorValue {
        &self.inverse_rms
    }

    pub fn normalized(&self) -> &TensorValue {
        &self.normalized
    }

    pub fn output(&self) -> &TensorValue {
        &self.output
    }

    pub fn into_output(self) -> TensorValue {
        self.output
    }
}

/// One learned gain applied after final-axis root-mean-square rescaling
#[derive(Clone, Debug)]
pub struct RmsNorm {
    epsilon: f64,
    feature_width: usize,
    parameters: NamedParameters,
}

impl RmsNorm {
    /// Creates a gain vector initialized to one
    pub fn new(
        gain_name: impl Into<String>,
        feature_width: usize,
        epsilon: f64,
    ) -> Result<Self, RmsNormError> {
        if feature_width == 0 {
            return Err(RmsNormError::EmptyFeatureWidth);
        }
        let gain_name = gain_name.into();
        validate_name(&gain_name)?;
        let mut values = Vec::new();
        values.try_reserve_exact(feature_width).map_err(|_| {
            RmsNormError::GainAllocationFailed {
                elements: feature_width,
            }
        })?;
        values.resize(feature_width, DEFAULT_GAIN_VALUE);
        let gain = NamedParameter::from_tensor(
            gain_name,
            Tensor::from_vec(vec![feature_width], values).map_err(InitializationError::from)?,
        )?;
        Self::from_gain(gain, epsilon)
    }

    /// Builds the layer from one externally named rank-one trainable gain
    pub fn from_gain(gain: NamedParameter, epsilon: f64) -> Result<Self, RmsNormError> {
        validate_epsilon(epsilon)?;
        let shape = gain.tensor().shape();
        if shape.len() != 1 {
            return Err(RmsNormError::GainRank { shape });
        }
        let feature_width = shape[0];
        if feature_width == 0 {
            return Err(RmsNormError::EmptyFeatureWidth);
        }

        Ok(Self {
            epsilon,
            feature_width,
            parameters: NamedParameters::try_new(vec![gain])?,
        })
    }

    /// Normalizes the final feature axis and returns only the scaled output
    pub fn forward(&self, input: &TensorValue) -> Result<TensorValue, RmsNormError> {
        self.forward_with_context(AutogradContext::recording(), input)
    }

    /// Normalizes under the caller's explicit graph-recording policy
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
    ) -> Result<TensorValue, RmsNormError> {
        self.forward_with_intermediates_and_context(context, input)
            .map(RmsNormForward::into_output)
    }

    /// Normalizes the final feature axis and preserves each teaching value.
    pub fn forward_with_intermediates(
        &self,
        input: &TensorValue,
    ) -> Result<RmsNormForward, RmsNormError> {
        self.forward_with_intermediates_and_context(AutogradContext::recording(), input)
    }

    /// Computes RMSNorm along the final feature axis and exposes its intermediates.
    ///
    /// For each feature vector `x` of width `D`, RMSNorm computes:
    ///
    /// ```text
    /// mean_square = sum(x[i] * x[i]) / D
    /// inverse_rms = 1 / sqrt(mean_square + epsilon)
    /// normalized  = x * inverse_rms
    /// output      = normalized * gain
    /// ```
    ///
    /// Each combination of leading-axis coordinates defines an independent feature
    /// vector. The normalization statistic is computed separately for each vector,
    /// while the same trainable `gain` of shape `[D]` is shared across all vectors.
    ///
    /// For example, an input of shape `[2, 3, 4]` produces:
    ///
    /// ```text
    /// input:       [2, 3, 4]
    /// mean_square: [2, 3, 1]
    /// inverse_rms: [2, 3, 1]
    /// normalized:  [2, 3, 4]
    /// gain:              [4]
    /// output:      [2, 3, 4]
    /// ```
    ///
    /// # Why use logarithm and exponentiation?
    ///
    /// Mathematically, the inverse RMS can be calculated directly:
    ///
    /// ```text
    /// inverse_rms = 1 / sqrt(z)
    ///
    /// where z = mean_square + epsilon
    /// ```
    ///
    /// However, the current tensor autodiff implementation does not provide a
    /// differentiable reciprocal-square-root operation.
    ///
    /// Instead of introducing a specialized operation and implementing its VJP,
    /// this method expresses the inverse square root through existing
    /// differentiable `log`, `mul`, and `exp` operations.
    ///
    /// For any positive `z`, the following identity holds:
    ///
    /// ```text
    /// 1 / sqrt(z) = z^(-1/2) = exp(-0.5 * ln(z))
    /// ```
    ///
    /// Therefore, the computation is decomposed into:
    ///
    /// ```text
    /// stabilized = mean_square + epsilon
    /// log_value  = ln(stabilized)
    /// scaled_log = -0.5 * log_value
    /// inverse_rms = exp(scaled_log)
    /// ```
    ///
    /// Every operation already has a corresponding VJP in the tensor autodiff
    /// engine. Consequently, the backward pass can apply the chain rule through
    /// the existing computation graph to obtain gradients for both the input
    /// tensor and the trainable gain.
    ///
    /// This is an implementation choice favoring composability and reuse of
    /// verified autodiff primitives. It is not a numerical or performance
    /// advantage over a direct reciprocal square root.
    ///
    /// In fact, the composed approach introduces additional intermediate tensors,
    /// graph nodes, and backward operations. Floating-point rounding may also
    /// differ from a direct `1.0 / z.sqrt()` implementation.
    ///
    /// A future optimized RMSNorm could replace this decomposition with a
    /// specialized forward operation and VJP while preserving the same
    /// mathematical contract.
    ///
    /// # Numerical behavior
    ///
    /// `epsilon` is added before computing the inverse square root to avoid
    /// division by zero for zero-energy feature vectors.
    ///
    /// When `epsilon == 0.0`, a feature vector whose computed mean square is zero
    /// is rejected before evaluating `log(0)`.
    ///
    /// A positive epsilon also reduces sensitivity to very small input magnitudes,
    /// although it makes normalization only approximately invariant to positive
    /// input scaling.
    ///
    /// # Autograd behavior
    ///
    /// The supplied `context` controls whether the intermediate operations
    /// participate in the computation graph.
    ///
    /// When graph recording is enabled, gradients propagate through the entire
    /// normalization expression. The shared gain accumulates contributions from
    /// every feature vector through the existing broadcasting VJP.
    ///
    /// The returned `RmsNormForward` exposes `mean_square`, `inverse_rms`,
    /// `normalized`, and `output` for inspection and testing.
    pub fn forward_with_intermediates_and_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
    ) -> Result<RmsNormForward, RmsNormError> {
        let shape = input.shape();
        let Some(&actual_width) = shape.last() else {
            return Err(RmsNormError::InputRankZero);
        };
        if actual_width != self.feature_width {
            return Err(RmsNormError::InputWidthMismatch {
                expected: self.feature_width,
                actual: actual_width,
            });
        }
        let feature_axis = shape.len() - 1;
        let squared = input
            .mul_with_context(context, input)
            .map_err(autodiff_error(RmsNormStage::Square))?;
        let mean_square = squared
            .mean_axis_with_context(context, feature_axis, true)
            .map_err(autodiff_error(RmsNormStage::MeanSquare))?;
        if self.epsilon == 0.0 {
            for (row, value) in mean_square.value().as_slice().iter().enumerate() {
                if *value == 0.0 {
                    return Err(RmsNormError::ZeroEnergyRow { row });
                }
            }
        }
        let epsilon = scalar_constant(self.epsilon, RmsNormStage::EpsilonConstant)?;
        let stabilized = mean_square
            .add_with_context(context, &epsilon)
            .map_err(autodiff_error(RmsNormStage::Stabilize))?;
        let log_mean_square = stabilized
            .log_with_context(context)
            .map_err(autodiff_error(RmsNormStage::LogMeanSquare))?;
        let exponent = scalar_constant(-0.5, RmsNormStage::ExponentConstant)?;
        let scaled_log = log_mean_square
            .mul_with_context(context, &exponent)
            .map_err(autodiff_error(RmsNormStage::ScaleLog))?;
        let inverse_rms = scaled_log
            .exp_with_context(context)
            .map_err(autodiff_error(RmsNormStage::ReciprocalRoot))?;
        let normalized = input
            .mul_with_context(context, &inverse_rms)
            .map_err(autodiff_error(RmsNormStage::Normalize))?;
        let output = normalized
            .mul_with_context(context, self.gain().tensor())
            .map_err(autodiff_error(RmsNormStage::ApplyGain))?;

        Ok(RmsNormForward {
            mean_square,
            inverse_rms,
            normalized,
            output,
        })
    }

    pub fn gain(&self) -> &NamedParameter {
        &self.parameters.as_slice()[0]
    }

    pub fn parameters(&self) -> &[NamedParameter] {
        self.parameters.as_slice()
    }

    pub fn feature_width(&self) -> usize {
        self.feature_width
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod rms_norm {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_feature_width_is_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gain_name = "foo";
                    let feature_width = 0;
                    let epsilon = 0.1;
                    let result = RmsNorm::new(gain_name, feature_width, epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), RmsNormError::EmptyFeatureWidth);
                }
            }

            mod when_name_is_invalid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gain_name = "";
                    let feature_width = 1;
                    let epsilon = 0.1;
                    let result = RmsNorm::new(gain_name, feature_width, epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::Initialization(InitializationError::EmptyName)
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_rms_norm() {
                    let gain_name = "foo";
                    let feature_width = 2;
                    let epsilon = 0.1;
                    let result = RmsNorm::new(gain_name, feature_width, epsilon);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.epsilon, epsilon);
                    assert_eq!(result.feature_width, feature_width);
                    assert_eq!(
                        result.gain().tensor().value().clone(),
                        Tensor::from_vec(vec![2], vec![DEFAULT_GAIN_VALUE, DEFAULT_GAIN_VALUE])
                            .unwrap()
                    );
                }
            }
        }

        mod fn_from_gain {
            use super::*;

            mod when_epsilon_is_invalid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let epsilon = -0.1;
                    let gain = NamedParameter::from_tensor(
                        "foo",
                        Tensor::from_vec(vec![1], vec![1.]).unwrap(),
                    ).unwrap();
                    let result = RmsNorm::from_gain(gain, epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::InvalidEpsilon { value: epsilon }
                    );
                }
            }

            mod gain_rank_is_not_1 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let epsilon = 0.1;
                    let gain = NamedParameter::from_tensor(
                        "foo",
                        Tensor::from_vec(vec![1, 1], vec![1.]).unwrap(),
                    ).unwrap();
                    let result = RmsNorm::from_gain(gain, epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::GainRank { shape: vec![1, 1] }
                    );
                }
            }

            mod when_feature_width_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let epsilon = 0.1;
                    let gain = NamedParameter::from_tensor(
                        "foo",
                        Tensor::from_vec(vec![0], vec![]).unwrap(),
                    ).unwrap();
                    let result = RmsNorm::from_gain(gain, epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::EmptyFeatureWidth
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_rms_norm() {
                    let epsilon = 0.1;
                    let gain = NamedParameter::from_tensor(
                        "foo",
                        Tensor::from_vec(vec![2], vec![1., 1.]).unwrap(),
                    ).unwrap();
                    let result = RmsNorm::from_gain(gain, epsilon);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.epsilon, epsilon);
                    assert_eq!(result.feature_width, 2);
                    assert_eq!(
                        result.gain().tensor().value().clone(),
                        Tensor::from_vec(vec![2], vec![1., 1.])
                            .unwrap()
                    );
                }
            }
        }

        mod fn_forward_with_intermediates_and_context {
            use super::*;

            mod when_epsilon_is_0_and_mean_square_value_is_also_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let rms_norm = RmsNorm::new("foo", 2, 0.0).unwrap();
                    let input_tensor = Tensor::from_vec(vec![2, 2], vec![1., 2., 0., 0.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        rms_norm.forward_with_intermediates_and_context(context, &input_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::ZeroEnergyRow { row: 1 }
                    );
                }
            }

            mod when_input_is_scalar {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let rms_norm = RmsNorm::new("foo", 1, 0.1).unwrap();
                    let input_tensor = Tensor::from_vec(vec![], vec![1.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        rms_norm.forward_with_intermediates_and_context(context, &input_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), RmsNormError::InputRankZero);
                }
            }

            mod when_input_feature_with_mismatches {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let rms_norm = RmsNorm::new("foo", 1, 0.1).unwrap();
                    let input_tensor = Tensor::from_vec(vec![1, 3], vec![1., 2., 3.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        rms_norm.forward_with_intermediates_and_context(context, &input_param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        RmsNormError::InputWidthMismatch {
                            expected: 1,
                            actual: 3,
                        }
                    );
                }
            }

            mod feed_forward_calculations {
                use super::*;

                #[test]
                fn it_applies_rms_norm() {
                    let epsilon = 0.1;
                    let gain_tensor = Tensor::from_vec(vec![2], vec![0.5, 2.5]).unwrap();
                    let gain =
                        NamedParameter::from_tensor("gain.weight", gain_tensor.clone()).unwrap();
                    let rms_norm = RmsNorm::from_gain(gain, epsilon).unwrap();
                    let input_tensor =
                        Tensor::from_vec(vec![3, 2], vec![1.6, 2., 2., 4., 5., 6.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        rms_norm.forward_with_intermediates_and_context(context, &input_param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(
                        result.mean_square.value().clone(),
                        Tensor::from_vec(
                            vec![3, 1],
                            vec![
                                (1.6_f64.powf(2.) + 2_f64.powf(2.)) / 2.,
                                (2_f64.powf(2.) + 4_f64.powf(2.)) / 2.,
                                (5_f64.powf(2.) + 6_f64.powf(2.)) / 2.
                            ]
                        )
                        .unwrap()
                    );
                    let mean_square_tensor = result.mean_square.value().clone();
                    assert_eq!(
                        result.inverse_rms.value().clone(),
                        Tensor::from_vec(
                            vec![3, 1],
                            vec![
                                1.0 / (mean_square_tensor.as_slice()[0] + epsilon).sqrt(),
                                1.0 / (mean_square_tensor.as_slice()[1] + epsilon).sqrt(),
                                1.0 / (mean_square_tensor.as_slice()[2] + epsilon).sqrt()
                            ]
                        )
                        .unwrap()
                    );
                    let inverse_rms_tensor = result.inverse_rms.value().clone();
                    assert_eq!(
                        result.normalized.value().clone(),
                        Tensor::from_vec(
                            vec![3, 2],
                            vec![
                                input_tensor.as_slice()[0] * inverse_rms_tensor.as_slice()[0],
                                input_tensor.as_slice()[1] * inverse_rms_tensor.as_slice()[0],
                                input_tensor.as_slice()[2] * inverse_rms_tensor.as_slice()[1],
                                input_tensor.as_slice()[3] * inverse_rms_tensor.as_slice()[1],
                                input_tensor.as_slice()[4] * inverse_rms_tensor.as_slice()[2],
                                input_tensor.as_slice()[5] * inverse_rms_tensor.as_slice()[2],
                            ]
                        )
                        .unwrap()
                    );
                    let normalized_tensor = result.normalized.value().clone();
                    assert_eq!(
                        result.output.value().clone(),
                        Tensor::from_vec(
                            vec![3, 2],
                            vec![
                                normalized_tensor.as_slice()[0] * gain_tensor.as_slice()[0],
                                normalized_tensor.as_slice()[1] * gain_tensor.as_slice()[1],
                                normalized_tensor.as_slice()[2] * gain_tensor.as_slice()[0],
                                normalized_tensor.as_slice()[3] * gain_tensor.as_slice()[1],
                                normalized_tensor.as_slice()[4] * gain_tensor.as_slice()[0],
                                normalized_tensor.as_slice()[5] * gain_tensor.as_slice()[1],
                            ]
                        )
                        .unwrap()
                    );
                }
            }

            mod gradient_calculation {
                use super::*;
                use crate::support::gradcheck::{assert_sampled_gradient, backward_with_test_seed};

                #[test]
                fn it_calculates_gradient_correctly() {
                    let epsilon = 0.1;
                    let gain_tensor = Tensor::from_vec(vec![2], vec![0.5, 2.5]).unwrap();
                    let gain =
                        NamedParameter::from_tensor("gain.weight", gain_tensor.clone()).unwrap();
                    let rms_norm = RmsNorm::from_gain(gain.clone(), epsilon).unwrap();
                    let input_tensor =
                        Tensor::from_vec(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                    let input_param = TensorValue::parameter(input_tensor.clone()).unwrap();
                    let context = AutogradContext::recording();
                    let result =
                        rms_norm.forward_with_intermediates_and_context(context, &input_param);

                    assert!(result.is_ok(), "{result:?}");
                    let output = result.unwrap().into_output();
                    let seed = backward_with_test_seed(&output);

                    assert_sampled_gradient(gain.tensor(), &gain_tensor, &seed, |candidate| {
                        let candidate_param =
                            NamedParameter::from_tensor("gain.weight", candidate.clone()).unwrap();
                        let candidate = RmsNorm::from_gain(candidate_param, epsilon).unwrap();
                        let right = TensorValue::constant(input_tensor.clone()).unwrap();
                        candidate
                            .forward_with_intermediates_and_context(context, &right)
                            .unwrap()
                            .into_output()
                    });
                }
            }
        }
    }
}
