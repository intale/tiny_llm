//! A bias-free, position-wise SwiGLU feed-forward network.

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, NamedParameters, SplitMix64};
use crate::nn::linear::{Linear, LinearError};

/// The projection whose construction or forward pass failed
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SwiGluProjection {
    Gate,
    Up,
    Down,
}

impl fmt::Display for SwiGluProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SwiGluOperation {
    SiluGate,
    ElementwiseGate,
}

impl fmt::Display for SwiGluOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SiluGate => "SiLU gate",
            Self::ElementwiseGate => "elementwise gate",
        })
    }
}

/// A rejected parameter set, input, or delegated differentiable operation.
#[derive(Clone, Debug, PartialEq)]
pub enum SwiGluError {
    Projection {
        projection: SwiGluProjection,
        source: LinearError,
    },
    Autodiff {
        operation: SwiGluOperation,
        source: TensorAutodiffError,
    },
    BranchInputWidthMismatch {
        gate: usize,
        up: usize,
    },
    BranchHiddenWidthMismatch {
        gate: usize,
        up: usize,
    },
    DownInputWidthMismatch {
        hidden: usize,
        down: usize,
    },
    Initialization(InitializationError),
}

impl fmt::Display for SwiGluError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection { projection, source } => {
                write!(formatter, "SwiGLU {projection} projection: {source}")
            }
            Self::Autodiff { operation, source } => {
                write!(formatter, "SwiGLU {operation}: {source}")
            }
            Self::BranchInputWidthMismatch { gate, up } => write!(
                formatter,
                "SwiGLU gate and up input widths must match, got {gate} and {up}"
            ),
            Self::BranchHiddenWidthMismatch { gate, up } => write!(
                formatter,
                "SwiGLU gate and up hidden widths must match, got {gate} and {up}"
            ),
            Self::DownInputWidthMismatch { hidden, down } => write!(
                formatter,
                "SwiGLU down input width must equal hidden width {hidden}, got {down}"
            ),
            Self::Initialization(source) => source.fmt(formatter),
        }
    }
}

impl Error for SwiGluError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Projection { source, .. } => Some(source),
            Self::Autodiff { source, .. } => Some(source),
            Self::Initialization(source) => Some(source),
            _ => None,
        }
    }
}

fn projection_error(projection: SwiGluProjection) -> impl FnOnce(LinearError) -> SwiGluError {
    move |source| SwiGluError::Projection { projection, source }
}

fn autodiff_error(operation: SwiGluOperation) -> impl FnOnce(TensorAutodiffError) -> SwiGluError {
    move |source| SwiGluError::Autodiff { operation, source }
}

/// The exact tensors produced by one composed SwiGLU forward pass
#[derive(Clone, Debug)]
pub struct SwiGluForward {
    gate_linear: TensorValue,
    gate_silu: TensorValue,
    up: TensorValue,
    product: TensorValue,
    output: TensorValue,
}

impl SwiGluForward {
    pub fn gate_linear(&self) -> &TensorValue {
        &self.gate_linear
    }

    pub fn gate_silu(&self) -> &TensorValue {
        &self.gate_silu
    }

    pub fn up(&self) -> &TensorValue {
        &self.up
    }

    pub fn product(&self) -> &TensorValue {
        &self.product
    }

    pub fn output(&self) -> &TensorValue {
        &self.output
    }

    pub fn into_output(self) -> TensorValue {
        self.output
    }
}

/// Three bias-free projections with a SiLU-activated multiplicative gate
#[derive(Clone, Debug)]
pub struct SwiGlu {
    gate: Linear, // Wg
    up: Linear,   // Wu
    down: Linear, // W2
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
}

impl SwiGlu {
    pub fn new(
        parameter_prefix: impl Into<String>,
        input_width: usize,
        hidden_width: usize,
        output_width: usize,
        rng: &mut SplitMix64,
    ) -> Result<Self, SwiGluError> {
        let parameter_prefix = parameter_prefix.into();
        let mut trial = rng.clone();

        let gate = Linear::new(
            format!("{parameter_prefix}.gate"),
            input_width,
            hidden_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(SwiGluProjection::Gate))?;

        let up = Linear::new(
            format!("{parameter_prefix}.up"),
            input_width,
            hidden_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(SwiGluProjection::Up))?;

        let down = Linear::new(
            format!("{parameter_prefix}.down"),
            hidden_width,
            output_width,
            false,
            &mut trial,
        )
        .map_err(projection_error(SwiGluProjection::Down))?;

        let layer = Self::from_linears(gate, up, down)?;

        *rng = trial;
        Ok(layer)
    }

    /// Gives SwiGLU semantics to three existing bias-free weight matrices
    pub fn from_parameters(
        gate_weight: NamedParameter,
        up_weight: NamedParameter,
        down_weight: NamedParameter,
    ) -> Result<Self, SwiGluError> {
        let gate = Linear::from_parameters(gate_weight, None)
            .map_err(projection_error(SwiGluProjection::Gate))?;
        let up = Linear::from_parameters(up_weight, None)
            .map_err(projection_error(SwiGluProjection::Up))?;
        let down = Linear::from_parameters(down_weight, None)
            .map_err(projection_error(SwiGluProjection::Down))?;
        Self::from_linears(gate, up, down)
    }

    fn from_linears(gate: Linear, up: Linear, down: Linear) -> Result<Self, SwiGluError> {
        if gate.input_width() != up.input_width() {
            return Err(SwiGluError::BranchInputWidthMismatch {
                gate: gate.input_width(),
                up: up.input_width(),
            });
        }
        if gate.output_width() != up.output_width() {
            return Err(SwiGluError::BranchHiddenWidthMismatch {
                gate: gate.output_width(),
                up: up.output_width(),
            });
        }
        if down.input_width() != gate.output_width() {
            return Err(SwiGluError::DownInputWidthMismatch {
                hidden: gate.output_width(),
                down: down.input_width(),
            });
        }

        let input_width = gate.input_width();
        let hidden_width = gate.output_width();
        let output_width = down.output_width();

        Ok(Self {
            gate,
            up,
            down,
            input_width,
            hidden_width,
            output_width,
        })
    }

    /// Applies the same gated feature transformation at every leading position
    pub fn forward(&self, input: &TensorValue) -> Result<TensorValue, SwiGluError> {
        self.forward_with_context(AutogradContext::recording(), input)
    }

    /// Applies the gated transformation under an explicit recording policy
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
    ) -> Result<TensorValue, SwiGluError> {
        Ok(self
            .forward_with_intermediates_and_context(context, input)?
            .into_output())
    }

    /// Returns each branch tensor for inspection without changing the computation
    pub fn forward_with_intermediates(
        &self,
        input: &TensorValue,
    ) -> Result<SwiGluForward, SwiGluError> {
        self.forward_with_intermediates_and_context(AutogradContext::recording(), input)
    }

    /// Returns branch tensors under an explicit graph-recording policy
    pub fn forward_with_intermediates_and_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
    ) -> Result<SwiGluForward, SwiGluError> {
        let gate_linear = self
            .gate
            .forward_with_context(context, input)
            .map_err(projection_error(SwiGluProjection::Gate))?;
        let gate_silu = gate_linear
            .silu_with_context(context)
            .map_err(autodiff_error(SwiGluOperation::SiluGate))?;
        let up = self
            .up
            .forward_with_context(context, input)
            .map_err(projection_error(SwiGluProjection::Up))?;
        let product = gate_silu
            .mul_with_context(context, &up)
            .map_err(autodiff_error(SwiGluOperation::ElementwiseGate))?;
        let output = self
            .down
            .forward_with_context(context, &product)
            .map_err(projection_error(SwiGluProjection::Down))?;

        Ok(SwiGluForward {
            gate_linear,
            gate_silu,
            up,
            product,
            output,
        })
    }

    pub fn gate(&self) -> &Linear {
        &self.gate
    }

    pub fn up(&self) -> &Linear {
        &self.up
    }

    pub fn down(&self) -> &Linear {
        &self.down
    }

    pub fn parameters(&self) -> [&NamedParameter; 3] {
        [self.gate.weight(), self.up.weight(), self.down.weight()]
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn hidden_width(&self) -> usize {
        self.hidden_width
    }

    pub fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn parameter_count(&self) -> usize {
        2 * self.input_width * self.hidden_width + self.hidden_width * self.output_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO_SEED_NEXT_U64: u64 = 16294208416658607535;
    const ZERO_SEED_NEXT_U64_AFTER_SWI_GLU_INIT: u64 = 11741057589345805078;

    mod swi_glu {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_error_raiese {
                use super::*;

                #[test]
                fn it_does_not_change_rng() {
                    let param_prefix = "foo";
                    let input_width = 0;
                    let hidden_width = 4;
                    let output_width = 3;
                    let mut rng = SplitMix64::from_seed(0);
                    let result = SwiGlu::new(
                        param_prefix,
                        input_width,
                        hidden_width,
                        output_width,
                        &mut rng,
                    );

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_swi_glu_and_persists_rng_changes() {
                    let param_prefix = "foo";
                    let input_width = 2;
                    let hidden_width = 4;
                    let output_width = 3;
                    let mut rng = SplitMix64::from_seed(0);
                    let result = SwiGlu::new(
                        param_prefix,
                        input_width,
                        hidden_width,
                        output_width,
                        &mut rng,
                    );

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64_AFTER_SWI_GLU_INIT);

                    assert!(result.gate.bias().is_none());
                    assert_eq!(result.gate.weight().name(), "foo.gate.weight");
                    assert_eq!(result.gate.input_width(), input_width);
                    assert_eq!(result.gate.output_width(), hidden_width);

                    assert!(result.up.bias().is_none());
                    assert_eq!(result.up.weight().name(), "foo.up.weight");
                    assert_eq!(result.up.input_width(), input_width);
                    assert_eq!(result.up.output_width(), hidden_width);

                    assert!(result.down.bias().is_none());
                    assert_eq!(result.down.weight().name(), "foo.down.weight");
                    assert_eq!(result.down.input_width(), hidden_width);
                    assert_eq!(result.down.output_width(), output_width);
                }
            }
        }

        mod fn_from_parameters {
            use super::*;

            #[test]
            fn it_computes_swi_glu() {
                let mut rng = SplitMix64::from_seed(0);
                let gate_weight = NamedParameter::xavier_uniform("foo", 2, 2, &mut rng).unwrap();
                let up_weight = NamedParameter::xavier_uniform("bar", 2, 2, &mut rng).unwrap();
                let down_weight = NamedParameter::xavier_uniform("baz", 2, 2, &mut rng).unwrap();
                let result = SwiGlu::from_parameters(
                    gate_weight.clone(),
                    up_weight.clone(),
                    down_weight.clone(),
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(
                    result.gate.clone(),
                    Linear::from_parameters(gate_weight, None).unwrap()
                );
                assert_eq!(
                    result.up.clone(),
                    Linear::from_parameters(up_weight, None).unwrap()
                );
                assert_eq!(
                    result.down.clone(),
                    Linear::from_parameters(down_weight, None).unwrap()
                );
            }
        }

        mod fn_forward_with_intermediates_and_context {
            use super::*;
            use crate::support::all_forward_ops;
            use crate::tensor::storage::Tensor;

            #[test]
            fn it_performs_feed_forward() {
                let input_width = 2;
                let output_width = 1;
                let hidden_width = 4;

                let gate_tensor = Tensor::from_vec(
                    vec![input_width, hidden_width],
                    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                )
                .unwrap();
                let gate_weight = NamedParameter::from_tensor("foo", gate_tensor.clone()).unwrap();

                let up_tensor = Tensor::from_vec(
                    vec![input_width, hidden_width],
                    vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                )
                .unwrap();
                let up_weight = NamedParameter::from_tensor("bar", up_tensor.clone()).unwrap();

                let down_tensor =
                    Tensor::from_vec(vec![hidden_width, output_width], vec![2.1, 2.2, 2.3, 2.4])
                        .unwrap();
                let down_weight = NamedParameter::from_tensor("baz", down_tensor.clone()).unwrap();

                let swi_glu = SwiGlu::from_parameters(gate_weight, up_weight, down_weight).unwrap();

                let input_tensor =
                    Tensor::from_vec(vec![3, 2], vec![10., 20., 30., 40., 50., 60.]).unwrap();
                let input_value = TensorValue::parameter(input_tensor.clone()).unwrap();

                let context = AutogradContext::recording();

                let result = swi_glu.forward_with_intermediates_and_context(context, &input_value);

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                let expected_gate_linear_tensor = Tensor::from_vec(
                    vec![3, 4],
                    vec![
                        11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
                    ],
                )
                .unwrap();
                let expected_gate_silu_tensor = Tensor::from_vec(
                    vec![3, 4],
                    vec![
                        10.999816284359673,
                        13.999988358607611,
                        16.999999296210614,
                        19.999999958776925,
                        22.99999999763977,
                        29.999999999997197,
                        37.0,
                        44.0,
                        34.99999999999998,
                        46.0,
                        57.0,
                        68.0,
                    ],
                )
                .unwrap();
                let expected_up_tensor = Tensor::from_vec(
                    vec![3, 4],
                    vec![
                        41.0, 44.0, 47.0, 50.0, 93.0, 100.0, 107.0, 114.0, 145.0, 156.0, 167.0,
                        178.0,
                    ],
                )
                .unwrap();
                let expected_product_tensor = Tensor::from_vec(
                    vec![3, 4],
                    vec![
                        450.9924676587466,
                        615.9994877787349,
                        798.9999669218988,
                        999.9999979388463,
                        2138.9999997804985,
                        2999.99999999972,
                        3959.0,
                        5016.0,
                        5074.999999999997,
                        7176.0,
                        9519.0,
                        12104.0,
                    ],
                )
                .unwrap();
                let expected_output_tensor = Tensor::from_vec(
                    vec![3, 1],
                    vec![6539.982974170183, 32235.999999538428, 77388.0],
                )
                .unwrap();
                assert_eq!(
                    result.gate_linear.value().clone(),
                    expected_gate_linear_tensor
                );
                assert_eq!(result.gate_silu.value().clone(), expected_gate_silu_tensor);
                assert_eq!(result.up.value().clone(), expected_up_tensor);
                assert_eq!(result.product.value().clone(), expected_product_tensor);
                assert_eq!(result.output.value().clone(), expected_output_tensor);
                assert_eq!(
                    all_forward_ops(&result.output),
                    vec![
                        "parameter",
                        "parameter",
                        "matmul",
                        "silu",
                        "parameter",
                        "parameter",
                        "matmul",
                        "mul",
                        "parameter",
                        "matmul"
                    ]
                );
            }
        }

        mod fn_parameter_count {
            use super::*;

            #[test]
            fn it_returns_parameters_count() {
                let param_prefix = "foo";
                let input_width = 2;
                let hidden_width = 4;
                let output_width = 3;
                let mut rng = SplitMix64::from_seed(0);
                let swi_glu = SwiGlu::new(
                    param_prefix,
                    input_width,
                    hidden_width,
                    output_width,
                    &mut rng,
                )
                .unwrap();

                assert_eq!(
                    swi_glu.parameter_count(),
                    2 * input_width * hidden_width + hidden_width * output_width
                );
            }
        }
    }
}
