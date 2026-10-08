//! Bias-corrected Adam moments with decoupled weight decay for named parameters
/*
AdamW architecture and data flow
================================

The implementation separates:

    1. optimizer hyperparameters,
    2. per-parameter-group policy,
    3. long-lived Adam moment state,
    4. optional gradient preprocessing,
    5. preparation and validation of one optimizer step,
    6. persistence/checkpoint state.


Configuration
-------------

    AdamWConfig
    +--------------------------------+
    | learning_rate = eta            |
    | beta1                          |
    | beta2                          |
    | epsilon                        |
    | weight_decay = lambda          |
    +---------------+----------------+
                    |
                    v

    AdamWParameterGroups
    +--------------------------------+
    | parameter name ->              |
    |     Decay                      |
    |     NoDecay                    |
    +---------------+----------------+
                    |
                    |
                    v

    AdamW
    +----------------------------------------------------+
    | config: AdamWConfig                                |
    | groups: Option<AdamWParameterGroups>               |
    | step: u64                                          |
    | beta1_power: beta1^t                               |
    | beta2_power: beta2^t                               |
    | states: parameter_name -> AdamWMomentState         |
    +----------------------+-----------------------------+
                           |
                           |
                           +-----------------------------+
                                                         |
                                                         v
                                              AdamWMomentState
                                              +----------------+
                                              | shape          |
                                              | first  = m     |
                                              | second = v     |
                                              +----------------+

`AdamWMomentState` is the optimizer's memory for one named parameter.
Its `first` and `second` tensors have the same shape as that parameter.


Input to an optimizer step
--------------------------

`backward()` runs before AdamW and populates gradients on model parameters:

    NamedParameter
    +-----------------------------+
    | name                        |
    | value    = W                |
    | gradient = dL/dW            |
    +--------------+--------------+
                   |
                   | raw gradient g
                   v


Gradient preprocessing
----------------------

Before the gradient enters Adam's moment calculations it may be transformed:

    AdamWGradientTransform

        Uniform { scale }

            g_effective = g * scale

        Normalized {
            divisor,
            multiplier,
        }

            g_effective = (g / divisor) * multiplier

The normalized form is mathematically equivalent to:

    g * (multiplier / divisor)

but avoids prematurely forming a very small scalar ratio that could underflow
to zero in `f64`.

With the identity transform:

    scale = 1

the effective gradient is exactly the raw gradient:

    g_effective = g

This transform changes only the gradient consumed by Adam. It does not mutate
the raw gradient stored on the parameter.


Preparing one parameter update
------------------------------

For one named parameter, `AdamWPreparation` contains the already-resolved
information required by `prepare_parameter_update(...)`:

    AdamWPreparation
    +--------------------------------+
    | effective config               |
    | decay_applied                  |
    | first bias correction          |
    | second bias correction         |
    | parameter name                 |
    | gradient transform             |
    +---------------+----------------+
                    |
                    v

          prepare_parameter_update(...)
                    |
                    v


Adaptive Adam update
--------------------

For every scalar coordinate `i` of a parameter:

    current parameter value

                W[i]

    raw gradient produced by backward

                g[i]

                  |
                  v

        AdamWGradientTransform

                  |
                  v

        g_effective[i]

                  |
                  v

    first moment update:

        m_t[i]
          =
        beta1 * m_{t-1}[i]
          +
        (1 - beta1) * g_effective[i]

                  |
                  v

    second moment update:

        v_t[i]
          =
        beta2 * v_{t-1}[i]
          +
        (1 - beta2) * g_effective[i]^2

                  |
                  v

    bias correction:

        m_hat[i]
          =
        m_t[i] / (1 - beta1^t)

        v_hat[i]
          =
        v_t[i] / (1 - beta2^t)

                  |
                  v

    adaptive direction:

        direction[i]
          =
        m_hat[i]
          /
        (sqrt(v_hat[i]) + epsilon)

                  |
                  v

    adaptive optimizer contribution:

        adaptive_delta[i]
          =
        learning_rate * direction[i]


Optional weight-decay term
--------------------------

The current parameter value is also used to compute a separate decay term.

This is not an alternative execution path. It is an additional term in the
same final update formula.

The only condition is whether this particular parameter belongs to a group
that receives weight decay:

                            parameter name
                                  |
                                  v
                          decay_applied ?
                           /           \
                         yes            no
                          |              |
                          v              v
                  lambda_effective     0.0
                          |
                          v

                    decay_delta[i]
                      =
                    learning_rate
                      * lambda_effective
                      * W[i]

For a `NoDecay` parameter:

        lambda_effective = 0

therefore:

        decay_delta[i] = 0

No separate optimizer algorithm is selected.


Final scalar update
-------------------

The adaptive contribution and the optional decay contribution are combined
in one update:

        W_new[i]
          =
        W[i]
          - adaptive_delta[i]
          - decay_delta[i]


The complete scalar flow is therefore:

    raw gradient g[i]
            |
            v
    gradient transform
            |
            v
    effective gradient
            |
            v
        update m_t
            |
            v
        update v_t
            |
            v
       bias correction
            |
            v
     adaptive direction
            |
            v
      * learning_rate
            |
            v
      adaptive_delta
            |
            |                    W[i]
            |                      |
            |                      v
            |               decay_applied ?
            |                /          \
            |              yes           no
            |               |             |
            |               v             v
            |            lambda          0.0
            |               |
            |               v
            |          decay_delta
            |               |
            +-------+-------+
                    |
                    v

                W_new[i]
                  =
                W[i]
                  - adaptive_delta
                  - decay_delta


Per-scalar temporary result
---------------------------

The intermediate values computed for one scalar coordinate are represented by
`AdamWScalarResult`.

Conceptually it contains values such as:

    AdamWScalarResult
    +--------------------------------+
    | before = W[i]                  |
    | effective gradient             |
    | first moment = m_t[i]          |
    | second moment = v_t[i]         |
    | corrected first moment         |
    | corrected second moment        |
    | adaptive direction             |
    | adaptive delta                 |
    | decay delta                    |
    | after = W_new[i]               |
    +--------------------------------+

This is temporary computation data. It is not the long-lived optimizer state.


Candidate state before commit
-----------------------------

The implementation first prepares new values without immediately mutating the
persistent model/optimizer state.

For each parameter it obtains:

    candidate parameter value
        W_new

    candidate moment state
        AdamWMomentState {
            first:  m_t,
            second: v_t,
        }

After all required calculations and validation succeed:

                         all candidates
                               |
                               v
                            valid?
                         /           \
                       no             yes
                       |               |
                       v               v
                     ERROR           COMMIT
                                       |
                   +-------------------+-------------------+
                   |                   |                   |
                   v                   v                   v
                W := W_new          m := m_t            v := v_t

The optimizer therefore avoids leaving a partially updated parameter/moment
state when preparation fails.


Global Adam state after a successful step
-----------------------------------------

After the parameter updates are committed:

    step := step + 1

    beta1_power := beta1_power * beta1

    beta2_power := beta2_power * beta2

Thus the optimizer carries forward:

    AdamW
    |
    +-- current optimizer step
    |
    +-- beta1^t
    |
    +-- beta2^t
    |
    +-- m and v for every named parameter


Full update overview
--------------------

    backward()
        |
        v
    raw gradients dL/dW
        |
        v
    AdamWGradientTransform
        |
        v
    effective gradients
        |
        v
    update first moments m
        |
        v
    update second moments v
        |
        v
    bias correction
        |
        v
    adaptive direction
        |
        v
    adaptive_delta = eta * direction
        |
        |
        |                         current W
        |                            |
        |                            v
        |                    parameter groups
        |                            |
        |                            v
        |                    decay_applied ?
        |                     /           \
        |                   yes            no
        |                    |              |
        |                    v              v
        |                 lambda           0.0
        |                    |
        |                    v
        |          decay_delta = eta
        |                        * lambda_effective
        |                        * W
        |                    |
        +--------------------+
                 |
                 v

        W_new
          =
        W
          - adaptive_delta
          - decay_delta

                 |
                 v
              validate
                 |
                 v
               commit
                 |
        +--------+--------+
        |        |        |
        v        v        v
      W_new     m_t      v_t
                 |
                 v
             step += 1


Persistence
-----------

The live optimizer can be converted into graph-independent checkpoint state:

    AdamW
      |
      | persistence_state()
      v
    AdamWState
    +--------------------------------------+
    | config                               |
    | parameter groups                     |
    | step                                 |
    | beta1_power                          |
    | beta2_power                          |
    | entries: Vec<AdamWStateEntry>        |
    +-------------------+------------------+
                        |
                        v

                 AdamWStateEntry
                 +-----------------------+
                 | parameter name        |
                 | AdamWMomentState      |
                 +-----------------------+

The optimizer can later be reconstructed from that state:

    AdamWState
        |
        | AdamW::from_persistence_state(...)
        v
      AdamW

The restored optimizer state must correspond to matching model parameters,
because its moment tensors are histories accumulated for those parameters.


Role summary
------------

    AdamWConfig
        Hyperparameters controlling AdamW math:
        eta, beta1, beta2, epsilon, lambda.

    AdamWParameterGroups
        Determines whether weight decay applies to each named parameter.

    AdamWGradientTransform
        Preprocesses the raw gradient before it enters Adam's first and second
        moments.

    AdamWMomentState
        Long-lived first- and second-moment tensors for one parameter.

    AdamW
        Owns the optimizer's long-lived state and commits optimizer steps.

    AdamWPreparation
        Bundles the resolved information required to prepare one named
        parameter update.

    AdamWScalarResult
        Temporary intermediate result for one scalar parameter coordinate.

    AdamWStateEntry
        Persistable moment state for one named parameter.

    AdamWState
        Complete checkpointable optimizer state.
*/

use crate::nn::init::NamedParameter;
use crate::tensor::storage::{Tensor, TensorError};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// The two explicit parameter groups
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AdamWGroup {
    Decay,
    NoDecay,
}

impl fmt::Display for AdamWGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decay => formatter.write_str("decay"),
            Self::NoDecay => formatter.write_str("no-decay"),
        }
    }
}

/// The arithmetic stage that first produced a non-finite candidate value
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AdamWArithmetic {
    FirstMoment,
    SquaredGradient,
    SecondMoment,
    CorrectedFirstMoment,
    CorrectedSecondMoment,
    AdaptiveDirection,
    AdaptiveDelta,
    DecayDelta,
    Parameter,
}

impl fmt::Display for AdamWArithmetic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::FirstMoment => "first moment",
            Self::SquaredGradient => "squared gradient",
            Self::SecondMoment => "second moment",
            Self::CorrectedFirstMoment => "bias-corrected first moment",
            Self::CorrectedSecondMoment => "bias-corrected second moment",
            Self::AdaptiveDirection => "adaptive direction",
            Self::AdaptiveDelta => "adaptive update",
            Self::DecayDelta => "decoupled decay update",
            Self::Parameter => "updated parameter",
        };
        formatter.write_str(name)
    }
}

/// A deterministic rejection that leaves parameters and optimizer state intact.
#[derive(Clone, Debug, PartialEq)]
pub enum AdamWError {
    InvalidLearningRate {
        value: f64,
    },
    InvalidDecayProduct {
        learning_rate: f64,
        weight_decay: f64,
        product: f64,
    },
    InvalidGradientScale {
        value: f64,
    },
    InvalidGradientNormalization {
        divisor: f64,
        multiplier: f64,
    },
    InvalidBeta1 {
        value: f64,
    },
    InvalidBeta2 {
        value: f64,
    },
    InvalidEpsilon {
        value: f64,
    },
    InvalidWeightDecay {
        value: f64,
    },
    EmptyParameterGroups,
    EmptyGroupedParameterName {
        group: AdamWGroup,
    },
    DuplicateGroupedParameter {
        group: AdamWGroup,
        name: String,
    },
    ParameterInMultipleGroups {
        name: String,
    },
    EmptyParameterSet,
    DuplicateParameterName {
        name: String,
        first: usize,
        repeated: usize,
    },
    ParameterSetChanged {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    ParameterShapeChanged {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    MissingGradient {
        name: String,
    },
    GradientShapeMismatch {
        name: String,
        parameter: Vec<usize>,
        gradient: Vec<usize>,
    },
    ParameterRevisionOverflow {
        name: String,
    },
    ParameterValueBorrowed {
        name: String,
    },
    StepOverflow,
    NonFiniteArithmetic {
        name: String,
        index: usize,
        stage: AdamWArithmetic,
        value: f64,
    },
    Tensor(TensorError),
}

impl fmt::Display for AdamWError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLearningRate { value } => write!(
                formatter,
                "learning rate must be finite and greater than zero, got {value}"
            ),
            Self::InvalidDecayProduct {
                learning_rate,
                weight_decay,
                product,
            } => write!(
                formatter,
                "positive weight decay requires represented learning_rate * weight_decay in (0,1), got {learning_rate} * {weight_decay} = {product}"
            ),
            Self::InvalidGradientScale { value } => write!(
                formatter,
                "gradient scale must be finite in the closed interval [0,1], got {value}"
            ),
            Self::InvalidGradientNormalization {
                divisor,
                multiplier,
            } => write!(
                formatter,
                "normalized gradient transform requires a finite positive divisor and a finite multiplier in (0,divisor], got divisor {divisor} and multiplier {multiplier}"
            ),
            Self::InvalidBeta1 { value } => write!(
                formatter,
                "beta1 must be finite in the half-open interval [0,1), got {value}"
            ),
            Self::InvalidBeta2 { value } => write!(
                formatter,
                "beta2 must be finite in the half-open interval [0,1), got {value}"
            ),
            Self::InvalidEpsilon { value } => write!(
                formatter,
                "epsilon must be finite and greater than zero, got {value}"
            ),
            Self::InvalidWeightDecay { value } => write!(
                formatter,
                "weight decay must be finite and non-negative, got {value}"
            ),
            Self::EmptyParameterGroups => formatter
                .write_str("explicit AdamW parameter groups must assign at least one stable name"),
            Self::EmptyGroupedParameterName { group } => {
                write!(
                    formatter,
                    "the {group} group contains an empty parameter name"
                )
            }
            Self::DuplicateGroupedParameter { group, name } => write!(
                formatter,
                "parameter name {name:?} repeats inside the {group} group"
            ),
            Self::ParameterInMultipleGroups { name } => write!(
                formatter,
                "parameter name {name:?} appears in both the decay and no-decay groups"
            ),
            Self::EmptyParameterSet => {
                formatter.write_str("AdamW needs at least one named parameter")
            }
            Self::DuplicateParameterName {
                name,
                first,
                repeated,
            } => write!(
                formatter,
                "parameter name {name:?} first appears at index {first} and repeats at index {repeated}"
            ),
            Self::ParameterSetChanged { expected, actual } => write!(
                formatter,
                "parameter-name set changed from {expected:?} to {actual:?}"
            ),
            Self::ParameterShapeChanged {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "parameter {name:?} changed shape from {expected:?} to {actual:?}"
            ),
            Self::MissingGradient { name } => {
                write!(formatter, "parameter {name:?} has no stored gradient")
            }
            Self::GradientShapeMismatch {
                name,
                parameter,
                gradient,
            } => write!(
                formatter,
                "parameter {name:?} has shape {parameter:?}, but its gradient has shape {gradient:?}"
            ),
            Self::ParameterRevisionOverflow { name } => {
                write!(
                    formatter,
                    "parameter {name:?} value revision overflowed u64"
                )
            }
            Self::ParameterValueBorrowed { name } => write!(
                formatter,
                "parameter {name:?} cannot be updated while its value is borrowed"
            ),
            Self::StepOverflow => formatter.write_str("AdamW step counter overflowed u64"),
            Self::NonFiniteArithmetic {
                name,
                index,
                stage,
                value,
            } => write!(
                formatter,
                "parameter {name:?} produced non-finite {stage} at flat index {index}: {value}"
            ),
            Self::Tensor(error) => error.fmt(formatter),
        }
    }
}

impl Error for AdamWError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for AdamWError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

/// A malformed optimizer snapshot rejected before an AdamW instance is built.
#[derive(Clone, Debug, PartialEq)]
pub enum AdamWStateError {
    InvalidBetaPower {
        name: &'static str,
        value: f64,
        step: u64,
    },
    StatePresence {
        step: u64,
        entries: usize,
    },
    EmptyParameterName,
    DuplicateParameterName {
        name: String,
    },
    ShapeProductOverflow {
        name: String,
    },
    MomentLengthMismatch {
        name: String,
        shape: Vec<usize>,
        expected: usize,
        first: usize,
        second: usize,
    },
    NonFiniteMoment {
        name: String,
        kind: &'static str,
        index: usize,
        value: f64,
    },
    NegativeSecondMoment {
        name: String,
        index: usize,
        value: f64,
    },
    ParameterGroupsMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
}

impl fmt::Display for AdamWStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBetaPower { name, value, step } => write!(
                formatter,
                "{name} power must be exactly 1 at step zero or finite in [0,1) later, got {value} at step {step}"
            ),
            Self::StatePresence { step, entries } => write!(
                formatter,
                "AdamW step {step} is incompatible with {entries} persisted moment entries"
            ),
            Self::EmptyParameterName => {
                formatter.write_str("an AdamW persistence entry has an empty parameter name")
            }
            Self::DuplicateParameterName { name } => write!(
                formatter,
                "AdamW persistence parameter name {name:?} appears more than once"
            ),
            Self::ShapeProductOverflow { name } => write!(
                formatter,
                "AdamW persistence shape for parameter {name:?} overflows usize"
            ),
            Self::MomentLengthMismatch {
                name,
                shape,
                expected,
                first,
                second,
            } => write!(
                formatter,
                "AdamW persistence parameter {name:?} with shape {shape:?} needs {expected} moments, got {first} first and {second} second"
            ),
            Self::NonFiniteMoment {
                name,
                kind,
                index,
                value,
            } => write!(
                formatter,
                "AdamW persistence parameter {name:?} has non-finite {kind} moment at flat index {index}: {value}"
            ),
            Self::NegativeSecondMoment { name, index, value } => write!(
                formatter,
                "AdamW persistence parameter {name:?} has negative second moment at flat index {index}: {value}"
            ),
            Self::ParameterGroupsMismatch { expected, actual } => write!(
                formatter,
                "AdamW persistence groups name {expected:?}, but moments name {actual:?}"
            ),
        }
    }
}

impl Error for AdamWStateError {}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AdamWConfig {
    learning_rate: f64, // 𝝶
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64, // λ
    decay_product: f64,
    shrinkage_factor: f64,
}

impl AdamWConfig {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Result<Self, AdamWError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(AdamWError::InvalidLearningRate {
                value: learning_rate,
            });
        }
        if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
            return Err(AdamWError::InvalidBeta1 { value: beta1 });
        }
        if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
            return Err(AdamWError::InvalidBeta2 { value: beta2 });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(AdamWError::InvalidEpsilon { value: epsilon });
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(AdamWError::InvalidWeightDecay {
                value: weight_decay,
            });
        }

        let decay_product = learning_rate * weight_decay;
        if weight_decay != 0.0
            && (!decay_product.is_finite() || decay_product <= 0.0 || decay_product >= 1.0)
        {
            return Err(AdamWError::InvalidDecayProduct {
                learning_rate,
                weight_decay,
                product: decay_product,
            });
        }

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            decay_product,
            shrinkage_factor: 1.0 - decay_product,
        })
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// The represented product that controls decoupled shrinkage.
    pub fn decay_product(&self) -> f64 {
        self.decay_product
    }

    /// The represented factor corresponding to the decoupled decay product.
    pub fn shrinkage_factor(&self) -> f64 {
        self.shrinkage_factor
    }

    /// Revalidates only the scheduled learning rate while preserving AdamW's moment and decay
    /// controls
    pub fn with_learning_rate(&self, learning_rate: f64) -> Result<Self, AdamWError> {
        Self::new(
            learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
        )
    }
}

fn validate_gradient_scale(value: f64) -> Result<(), AdamWError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(AdamWError::InvalidGradientScale { value })
    }
}

/// A candidate non-amplifying transformation of every raw gradient coordinate (gradient
/// preprocessing coefficient).
///
/// The enum variants, basically, does same thing: `gradient * scale`. But `Normalized` variant
/// explicitly splits `scale` as `multiplier / divisor`. This is need to prevent f64 overflow and
/// collapsing very small value of `multiplier / divisor` operation into `0.0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AdamWGradientTransform {
    /// Multiplies every coordinate by one shared represented scalar.
    Uniform { scale: f64 },
    /// Computes `(gradient / divisor) * multiplier` without first forming a possibly
    /// unrepresentable uniform scalar
    Normalized { divisor: f64, multiplier: f64 },
}

impl AdamWGradientTransform {
    pub fn uniform(scale: f64) -> Self {
        Self::Uniform { scale }
    }

    pub fn normalized(divisor: f64, multiplier: f64) -> Self {
        Self::Normalized {
            divisor,
            multiplier,
        }
    }

    fn validate(self) -> Result<Self, AdamWError> {
        match self {
            Self::Uniform { scale } => {
                validate_gradient_scale(scale)?;
            }
            Self::Normalized {
                divisor,
                multiplier,
            } => {
                if !divisor.is_finite()
                    || divisor <= 0.0
                    || !multiplier.is_finite()
                    || multiplier <= 0.0
                    || multiplier > divisor
                {
                    return Err(AdamWError::InvalidGradientNormalization {
                        divisor,
                        multiplier,
                    });
                }
            }
        }
        Ok(self)
    }

    /// Applies the represented transformation to one raw coordinate
    pub fn apply(&self, gradient: f64) -> f64 {
        match self {
            Self::Uniform { scale: 1.0 } => gradient,
            Self::Uniform { scale } => gradient * scale,
            Self::Normalized {
                divisor,
                multiplier,
            } => (gradient / divisor) * multiplier,
        }
    }
}

fn collect_group<I, S>(names: I, group: AdamWGroup) -> Result<BTreeSet<String>, AdamWError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut collected = BTreeSet::new();
    for name in names {
        let name = name.into();
        if name.is_empty() {
            return Err(AdamWError::EmptyGroupedParameterName { group });
        }
        if !collected.insert(name.clone()) {
            return Err(AdamWError::DuplicateGroupedParameter { group, name });
        }
    }
    Ok(collected)
}

/// Exact stable-name assignments for decayed and decay-excluded parameters
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdamWParameterGroups {
    decay: BTreeSet<String>,
    no_decay: BTreeSet<String>,
}

impl AdamWParameterGroups {
    pub fn new<D, N, DS, NS>(decay: D, no_decay: N) -> Result<Self, AdamWError>
    where
        D: IntoIterator<Item = DS>,
        N: IntoIterator<Item = NS>,
        DS: Into<String>,
        NS: Into<String>,
    {
        let decay = collect_group(decay, AdamWGroup::Decay)?;
        let no_decay = collect_group(no_decay, AdamWGroup::NoDecay)?;
        if decay.is_empty() && no_decay.is_empty() {
            return Err(AdamWError::EmptyParameterGroups);
        }
        if let Some(name) = decay.intersection(&no_decay).next() {
            return Err(AdamWError::ParameterInMultipleGroups {
                name: name.to_owned(),
            });
        }

        Ok(Self { decay, no_decay })
    }

    pub fn decayed_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.decay.iter().map(String::as_str)
    }

    pub fn excluded_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.no_decay.iter().map(String::as_str)
    }

    fn parameter_names(&self) -> Vec<String> {
        self.decay.union(&self.no_decay).cloned().collect()
    }

    fn decays(&self, name: &str) -> bool {
        self.decay.contains(name)
    }
}

/// Name-keyed optimizer memory for one parameter tensor
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWMomentState {
    shape: Vec<usize>,
    first: Vec<f64>,
    second: Vec<f64>,
}

impl AdamWMomentState {
    fn zeros(shape: &[usize], elements: usize) -> Self {
        Self {
            shape: shape.to_vec(),
            first: vec![0.; elements],
            second: vec![0.; elements],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn first_moment(&self) -> &[f64] {
        &self.first
    }

    pub fn second_moment(&self) -> &[f64] {
        &self.second
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdamWStateEntry {
    name: String,
    moments: AdamWMomentState,
}

impl AdamWStateEntry {
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        first_moment: Vec<f64>,
        second_moment: Vec<f64>,
    ) -> Result<Self, AdamWStateError> {
        let name = name.into();
        if name.is_empty() {
            return Err(AdamWStateError::EmptyParameterName);
        }
        let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
            product.checked_mul(dimension)
        });
        let Some(elements) = elements else {
            return Err(AdamWStateError::ShapeProductOverflow { name });
        };
        if first_moment.len() != elements || second_moment.len() != elements {
            return Err(AdamWStateError::MomentLengthMismatch {
                name,
                shape,
                expected: elements,
                first: first_moment.len(),
                second: second_moment.len(),
            });
        }

        for (kind, values) in [
            ("first", first_moment.as_slice()),
            ("second", second_moment.as_slice()),
        ] {
            if let Some((index, &value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(AdamWStateError::NonFiniteMoment {
                    name,
                    kind,
                    index,
                    value,
                });
            }
        }

        if let Some((index, &value)) = second_moment
            .iter()
            .enumerate()
            .find(|(_, value)| **value < 0.0)
        {
            return Err(AdamWStateError::NegativeSecondMoment { name, index, value });
        }

        Ok(Self {
            name,
            moments: AdamWMomentState {
                shape,
                first: first_moment,
                second: second_moment,
            },
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn moments(&self) -> &AdamWMomentState {
        &self.moments
    }
}

fn validate_beta_power(name: &'static str, value: f64, step: u64) -> Result<(), AdamWStateError> {
    let valid = if step == 0 {
        value == 1.0
    } else {
        value.is_finite() && (0.0..1.0).contains(&value)
    };
    if valid {
        Ok(())
    } else {
        Err(AdamWStateError::InvalidBetaPower { name, value, step })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdamWState {
    config: AdamWConfig,
    groups: Option<AdamWParameterGroups>,
    step: u64, // How many optimizer updates this AdamWState has performed
    beta1_power: f64,
    beta2_power: f64,
    states: BTreeMap<String, AdamWMomentState>,
}

impl AdamWState {
    pub fn new(
        config: AdamWConfig,
        groups: Option<AdamWParameterGroups>,
        step: u64,
        beta1_power: f64,
        beta2_power: f64,
        entries: Vec<AdamWStateEntry>,
    ) -> Result<Self, AdamWStateError> {
        validate_beta_power("beta1", beta1_power, step)?;
        validate_beta_power("beta2", beta2_power, step)?;
        if (step == 0) != entries.is_empty() {
            return Err(AdamWStateError::StatePresence {
                step,
                entries: entries.len(),
            });
        }

        let mut states = BTreeMap::new();
        for entry in entries {
            if states.insert(entry.name.clone(), entry.moments).is_some() {
                return Err(AdamWStateError::DuplicateParameterName { name: entry.name });
            }
        }
        if let Some(groups) = &groups {
            let expected = groups.parameter_names();
            let actual = states.keys().cloned().collect::<Vec<_>>();
            if step > 0 && expected != actual {
                return Err(AdamWStateError::ParameterGroupsMismatch { expected, actual });
            }
        }

        Ok(Self {
            config,
            groups,
            step,
            beta1_power,
            beta2_power,
            states,
        })
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub fn parameter_groups(&self) -> Option<&AdamWParameterGroups> {
        self.groups.as_ref()
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn beta1_power(&self) -> f64 {
        self.beta1_power
    }

    pub fn beta2_power(&self) -> f64 {
        self.beta2_power
    }

    pub fn parameter_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.states.keys().map(String::as_str)
    }

    pub fn state(&self, name: &str) -> Option<&AdamWMomentState> {
        self.states.get(name)
    }
}

/// Exact elementwise evidence prepared for one named parameter in a step
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWParameterUpdate {
    name: String,
    shape: Vec<usize>,
    before: Vec<f64>,
    gradient: Vec<f64>,
    decay_applied: bool,
    effective_weight_decay: f64,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    corrected_first_moment: Vec<f64>,
    corrected_second_moment: Vec<f64>,
    adaptive_direction: Vec<f64>,
    adaptive_delta: Vec<f64>,
    decay_delta: Vec<f64>,
    after: Vec<f64>,
}

impl AdamWParameterUpdate {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn before(&self) -> &[f64] {
        &self.before
    }

    /// The effective gradient used by the moment calculation.
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    pub fn decay_applied(&self) -> bool {
        self.decay_applied
    }

    pub fn effective_weight_decay(&self) -> f64 {
        self.effective_weight_decay
    }

    pub fn first_moment(&self) -> &[f64] {
        &self.first_moment
    }

    pub fn second_moment(&self) -> &[f64] {
        &self.second_moment
    }

    pub fn corrected_first_moment(&self) -> &[f64] {
        &self.corrected_first_moment
    }

    pub fn corrected_second_moment(&self) -> &[f64] {
        &self.corrected_second_moment
    }

    pub fn adaptive_direction(&self) -> &[f64] {
        &self.adaptive_direction
    }

    pub fn adaptive_delta(&self) -> &[f64] {
        &self.adaptive_delta
    }

    pub fn decay_delta(&self) -> &[f64] {
        &self.decay_delta
    }

    pub fn after(&self) -> &[f64] {
        &self.after
    }
}

/// The optional trace for one committed multi-parameter update
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWStep {
    step: u64,
    learning_rate: f64,
    first_correction: f64,
    second_correction: f64,
    updates: Vec<AdamWParameterUpdate>,
}

impl AdamWStep {
    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn first_correction(&self) -> f64 {
        self.first_correction
    }

    pub fn second_correction(&self) -> f64 {
        self.second_correction
    }

    pub fn updates(&self) -> &[AdamWParameterUpdate] {
        &self.updates
    }
}

struct AdamWParameterContext<'a> {
    name: &'a str,
    shape: &'a [usize],
    before: &'a [f64],
    decay_applied: bool,
    effective_weight_decay: f64,
}

#[derive(Copy, Clone)]
struct AdamWScalarResult {
    gradient: f64,
    first_moment: f64,
    second_moment: f64,
    corrected_first_moment: f64,
    corrected_second_moment: f64,
    adaptive_direction: f64,
    adaptive_delta: f64,
    decay_delta: f64,
    after: f64,
}

trait AdamWStepObserver: Sized {
    type Parameter;
    type Output;

    fn begin_parameter(
        &mut self,
        context: AdamWParameterContext<'_>,
        elements: usize,
    ) -> Self::Parameter;

    fn observe_scalar(&mut self, parameter: &mut Self::Parameter, result: AdamWScalarResult);

    fn finish_parameter(&mut self, parameter: Self::Parameter);

    fn finish(
        self,
        step: u64,
        learning_rate: f64,
        first_correction: f64,
        second_correction: f64,
    ) -> Self::Output;
}

#[derive(Copy, Clone, Default)]
struct NoAdamWTrace;

impl AdamWStepObserver for NoAdamWTrace {
    type Parameter = ();
    type Output = u64;

    fn begin_parameter(
        &mut self,
        _context: AdamWParameterContext<'_>,
        _elements: usize,
    ) -> Self::Parameter {
    }

    fn observe_scalar(&mut self, _parameter: &mut Self::Parameter, _result: AdamWScalarResult) {}

    fn finish_parameter(&mut self, _parameter: Self::Parameter) {}

    fn finish(
        self,
        step: u64,
        _learning_rate: f64,
        _first_correction: f64,
        _second_correction: f64,
    ) -> Self::Output {
        step
    }
}

struct RecordAdamWTrace {
    updates: Vec<AdamWParameterUpdate>,
}

impl RecordAdamWTrace {
    fn with_capacity(parameters: usize) -> Self {
        Self {
            updates: Vec::with_capacity(parameters),
        }
    }
}

impl AdamWStepObserver for RecordAdamWTrace {
    type Parameter = AdamWParameterUpdate;
    type Output = AdamWStep;

    fn begin_parameter(
        &mut self,
        context: AdamWParameterContext<'_>,
        elements: usize,
    ) -> Self::Parameter {
        AdamWParameterUpdate {
            name: context.name.to_owned(),
            shape: context.shape.to_vec(),
            before: context.before.to_vec(),
            gradient: Vec::with_capacity(elements),
            decay_applied: context.decay_applied,
            effective_weight_decay: context.effective_weight_decay,
            first_moment: Vec::with_capacity(elements),
            second_moment: Vec::with_capacity(elements),
            corrected_first_moment: Vec::with_capacity(elements),
            corrected_second_moment: Vec::with_capacity(elements),
            adaptive_direction: Vec::with_capacity(elements),
            adaptive_delta: Vec::with_capacity(elements),
            decay_delta: Vec::with_capacity(elements),
            after: Vec::with_capacity(elements),
        }
    }

    fn observe_scalar(&mut self, parameter: &mut Self::Parameter, result: AdamWScalarResult) {
        parameter.gradient.push(result.gradient);
        parameter.first_moment.push(result.first_moment);
        parameter.second_moment.push(result.second_moment);
        parameter
            .corrected_first_moment
            .push(result.corrected_first_moment);
        parameter
            .corrected_second_moment
            .push(result.corrected_second_moment);
        parameter.adaptive_direction.push(result.adaptive_direction);
        parameter.adaptive_delta.push(result.adaptive_delta);
        parameter.decay_delta.push(result.decay_delta);
        parameter.after.push(result.after);
    }

    fn finish_parameter(&mut self, parameter: Self::Parameter) {
        self.updates.push(parameter);
    }

    fn finish(
        self,
        step: u64,
        learning_rate: f64,
        first_correction: f64,
        second_correction: f64,
    ) -> Self::Output {
        AdamWStep {
            step,
            learning_rate,
            first_correction,
            second_correction,
            updates: self.updates,
        }
    }
}

fn validate_parameter_names(parameters: &[NamedParameter]) -> Result<Vec<String>, AdamWError> {
    if parameters.is_empty() {
        return Err(AdamWError::EmptyParameterSet);
    }

    let mut indices = BTreeMap::<&str, usize>::new();
    for (repeated, parameter) in parameters.iter().enumerate() {
        if let Some(&first) = indices.get(parameter.name()) {
            return Err(AdamWError::DuplicateParameterName {
                name: parameter.name().to_owned(),
                first,
                repeated,
            });
        }
        indices.insert(parameter.name(), repeated);
    }
    Ok(indices.keys().map(|name| (*name).to_owned()).collect())
}

fn finite(name: &str, index: usize, stage: AdamWArithmetic, value: f64) -> Result<f64, AdamWError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AdamWError::NonFiniteArithmetic {
            name: name.to_owned(),
            index,
            stage,
            value,
        })
    }
}

struct AdamWPreparation<'a> {
    config: AdamWConfig,
    decay_applied: bool,
    first_correction: f64,
    second_correction: f64,
    name: &'a str,
    gradient_transform: AdamWGradientTransform,
}

fn prepare_parameter_update<Observer: AdamWStepObserver>(
    preparation: AdamWPreparation<'_>,
    before: &Tensor,
    gradient: &Tensor,
    state: &mut AdamWMomentState,
    observer: &mut Observer,
) -> Result<Vec<f64>, AdamWError> {
    let AdamWPreparation {
        config,
        decay_applied,
        first_correction,
        second_correction,
        name,
        gradient_transform,
    } = preparation;
    let effective_weight_decay = if decay_applied {
        config.weight_decay
    } else {
        0.0
    };
    let mut parameter_observation = observer.begin_parameter(
        AdamWParameterContext {
            name,
            shape: before.shape(),
            before: before.as_slice(),
            decay_applied,
            effective_weight_decay,
        },
        before.len(),
    );
    let mut after = Vec::with_capacity(before.len());

    for (index, ((value, raw_gradient), (first, second))) in before
        .as_slice()
        .iter()
        .zip(gradient.as_slice())
        .zip(state.first.iter_mut().zip(&mut state.second))
        .enumerate()
    {
        let gradient = gradient_transform.apply(*raw_gradient);
        let next_first = finite(
            name,
            index,
            AdamWArithmetic::FirstMoment,
            config.beta1 * *first + (1.0 - config.beta1) * gradient,
        )?;
        let squared_gradient = finite(
            name,
            index,
            AdamWArithmetic::SquaredGradient,
            gradient * gradient,
        )?;
        let next_second = finite(
            name,
            index,
            AdamWArithmetic::SecondMoment,
            config.beta2 * *second + (1.0 - config.beta2) * squared_gradient,
        )?;
        let corrected_first = finite(
            name,
            index,
            AdamWArithmetic::CorrectedFirstMoment,
            next_first / first_correction,
        )?;
        let corrected_second = finite(
            name,
            index,
            AdamWArithmetic::CorrectedSecondMoment,
            next_second / second_correction,
        )?;
        let direction = finite(
            name,
            index,
            AdamWArithmetic::AdaptiveDirection,
            corrected_first / (corrected_second.sqrt() + config.epsilon),
        )?;
        let adaptive = finite(
            name,
            index,
            AdamWArithmetic::AdaptiveDelta,
            config.learning_rate * direction,
        )?;
        let decay = finite(
            name,
            index,
            AdamWArithmetic::DecayDelta,
            config.learning_rate * effective_weight_decay * value,
        )?;
        let next_value = finite(
            name,
            index,
            AdamWArithmetic::Parameter,
            value - adaptive - decay,
        )?;

        *first = next_first;
        *second = next_second;
        after.push(next_value);
        observer.observe_scalar(
            &mut parameter_observation,
            AdamWScalarResult {
                gradient,
                first_moment: next_first,
                second_moment: next_second,
                corrected_first_moment: corrected_first,
                corrected_second_moment: corrected_second,
                adaptive_direction: direction,
                adaptive_delta: adaptive,
                decay_delta: decay,
                after: next_value,
            },
        )
    }

    observer.finish_parameter(parameter_observation);
    Ok(after)
}

/// A deterministic AdamW optimizer whose state follors stable parameter names
#[derive(Clone, Debug, PartialEq)]
pub struct AdamW {
    config: AdamWConfig,
    groups: Option<AdamWParameterGroups>,
    step: u64,
    beta1_power: f64,
    beta2_power: f64,
    states: BTreeMap<String, AdamWMomentState>,
}

impl AdamW {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            groups: None,
            step: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            states: BTreeMap::new(),
        }
    }

    /// Uses exact stable-name groups to apply or exclude configured decay
    pub fn with_parameter_groups(config: AdamWConfig, groups: AdamWParameterGroups) -> Self {
        Self {
            groups: Some(groups),
            ..Self::new(config)
        }
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn parameter_groups(&self) -> Option<&AdamWParameterGroups> {
        self.groups.as_ref()
    }

    pub fn parameter_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.states.keys().map(String::as_str)
    }

    pub fn state(&self, name: &str) -> Option<&AdamWMomentState> {
        self.states.get(name)
    }

    /// Captures every value required to reproduce the next optimizer update
    pub fn persistence_state(&self) -> AdamWState {
        AdamWState {
            config: self.config,
            groups: self.groups.clone(),
            step: self.step,
            beta1_power: self.beta1_power,
            beta2_power: self.beta2_power,
            states: self.states.clone(),
        }
    }

    /// Restores a previously validated continuation snapshot without arithmetic
    pub fn from_persistence_state(state: &AdamWState) -> Self {
        Self {
            config: state.config,
            groups: state.groups.clone(),
            step: state.step,
            beta1_power: state.beta1_power,
            beta2_power: state.beta2_power,
            states: state.states.clone(),
        }
    }

    /// Reads the accumulated gradients and atomically updates every live leaf
    ///
    /// All arithmetic, tensor construction, and optimizer-state changes are prepared first. An
    /// error leaves both the supplied parameters and this optimizer bit-identical. A successful
    /// commit preserves every parameter node and leaves its accumulated gradient for the caller to
    /// clear. The result is only the committed step number; use `step_with_trace()` when the
    /// elementwise update vectors are needed for inspection.
    pub fn step(&mut self, parameters: &[NamedParameter]) -> Result<u64, AdamWError> {
        self.step_with_config(
            parameters,
            self.config,
            AdamWGradientTransform::uniform(1.0),
            NoAdamWTrace,
        )
    }

    /// Applies the same transaction while recording every elementwise update
    pub fn step_with_trace(
        &mut self,
        parameters: &[NamedParameter],
    ) -> Result<AdamWStep, AdamWError> {
        let observer = RecordAdamWTrace::with_capacity(parameters.len());
        self.step_with_config(
            parameters,
            self.config,
            AdamWGradientTransform::uniform(1.0),
            observer,
        )
    }

    /// Applies one validated scheduled learning rate without resetting moments.
    ///
    /// The override belongs only to this update; `config()` keeps the optimizer's base rate. An
    /// invalid rate or any later preparation error leaves the parameters, moments, powers, and step
    /// counter unchanged
    pub fn step_with_learning_rate(
        &mut self,
        parameters: &[NamedParameter],
        learning_rate: f64,
    ) -> Result<u64, AdamWError> {
        let step_config = self.config.with_learning_rate(learning_rate)?;
        self.step_with_config(
            parameters,
            step_config,
            AdamWGradientTransform::uniform(1.0),
            NoAdamWTrace,
        )
    }

    /// Applies one scheduled rate and one validated global gradient scale.
    ///
    /// The scale multiplies only the gradient used by Adam's moments. The decoupled weight-decay
    /// branch continues to use the unscaled parameter.
    pub fn step_with_learning_rate_and_gradient_scale(
        &mut self,
        parameters: &[NamedParameter],
        learning_rate: f64,
        gradient_scale: f64,
    ) -> Result<u64, AdamWError> {
        self.step_with_learning_rate_and_gradient_transform(
            parameters,
            learning_rate,
            AdamWGradientTransform::uniform(gradient_scale),
        )
    }

    /// Applies one scheduled rate and one validated gradient transformation.
    ///
    /// Both forms affect only Adam's moment input. The normalized form prevents the complete
    /// transform from collapsing solely because its equivalent uniform scalar underflowed;
    /// individual tiny coordinates may still round to zero. Neither form changes raw gradients or
    /// the decoupled weight-decay branch.
    pub fn step_with_learning_rate_and_gradient_transform(
        &mut self,
        parameters: &[NamedParameter],
        learning_rate: f64,
        gradient_transform: AdamWGradientTransform,
    ) -> Result<u64, AdamWError> {
        let step_config = self.config.with_learning_rate(learning_rate)?;
        self.step_with_config(parameters, step_config, gradient_transform, NoAdamWTrace)
    }

    /// Applies a scheduled learning rate and records the complete update trace.
    pub fn step_with_learning_rate_and_trace(
        &mut self,
        parameters: &[NamedParameter],
        learning_rate: f64,
    ) -> Result<AdamWStep, AdamWError> {
        let step_config = self.config.with_learning_rate(learning_rate)?;
        let observer = RecordAdamWTrace::with_capacity(parameters.len());
        self.step_with_config(
            parameters,
            step_config,
            AdamWGradientTransform::uniform(1.0),
            observer,
        )
    }

    fn step_with_config<Observer: AdamWStepObserver>(
        &mut self,
        parameters: &[NamedParameter],
        step_config: AdamWConfig,
        gradient_transform: AdamWGradientTransform,
        mut observer: Observer,
    ) -> Result<Observer::Output, AdamWError> {
        let gradient_transform = gradient_transform.validate()?;
        let actual_names = validate_parameter_names(parameters)?;
        if let Some(groups) = &self.groups {
            let expected_names = groups.parameter_names();
            if expected_names != actual_names {
                return Err(AdamWError::ParameterSetChanged {
                    expected: expected_names,
                    actual: actual_names,
                });
            }
        }
        let next_step = self.step.checked_add(1).ok_or(AdamWError::StepOverflow)?;
        let next_beta1_power = self.beta1_power * step_config.beta1;
        let next_beta2_power = self.beta2_power * step_config.beta2;
        let first_correction = 1.0 - next_beta1_power;
        let second_correction = 1.0 - next_beta2_power;

        let mut candidate_states = if self.step == 0 {
            let mut states = BTreeMap::new();
            for parameter in parameters.iter() {
                let value = parameter.tensor().value();
                states.insert(
                    parameter.name().to_owned(),
                    AdamWMomentState::zeros(value.shape(), value.len()),
                );
            }
            states
        } else {
            let expected_names = self.states.keys().cloned().collect::<Vec<_>>();
            if expected_names != actual_names {
                return Err(AdamWError::ParameterSetChanged {
                    expected: expected_names,
                    actual: actual_names,
                });
            }
            self.states.clone()
        };

        let mut candidate_values = Vec::with_capacity(parameters.len());
        let mut next_revisions = Vec::with_capacity(parameters.len());

        for parameter in parameters.iter() {
            let name = parameter.name();
            let before = parameter.tensor().value();
            let gradient =
                parameter
                    .tensor()
                    .gradient()
                    .ok_or_else(|| AdamWError::MissingGradient {
                        name: name.to_owned(),
                    })?;
            if before.shape() != gradient.shape() {
                return Err(AdamWError::GradientShapeMismatch {
                    name: name.to_owned(),
                    parameter: before.shape().to_vec(),
                    gradient: gradient.shape().to_vec(),
                });
            }

            let state = candidate_states
                .get_mut(name)
                .expect("validated parameter names have candidate state");
            if state.shape != before.shape() {
                return Err(AdamWError::ParameterShapeChanged {
                    name: name.to_owned(),
                    expected: state.shape.clone(),
                    actual: before.shape().to_vec(),
                });
            }

            let after = prepare_parameter_update(
                AdamWPreparation {
                    config: step_config,
                    decay_applied: self
                        .groups
                        .as_ref()
                        .is_none_or(|groups| groups.decays(name)),
                    first_correction,
                    second_correction,
                    name,
                    gradient_transform,
                },
                &before,
                &gradient,
                state,
                &mut observer,
            )?;
            let tensor = Tensor::from_vec(before.shape().to_vec(), after)?;
            let next_revision = parameter.tensor().next_value_revision().ok_or_else(|| {
                AdamWError::ParameterRevisionOverflow {
                    name: name.to_owned(),
                }
            })?;
            candidate_values.push(tensor);
            next_revisions.push(next_revision);
        }

        let observation = observer.finish(
            next_step,
            step_config.learning_rate,
            first_correction,
            second_correction,
        );

        let mut value_writes = Vec::with_capacity(parameters.len());
        for parameter in parameters {
            let write = parameter.tensor().try_value_write().map_err(|_| {
                AdamWError::ParameterValueBorrowed {
                    name: parameter.name().to_owned(),
                }
            })?;
            value_writes.push(write);
        }

        for ((write, value), revision) in value_writes
            .into_iter()
            .zip(candidate_values)
            .zip(next_revisions)
        {
            write.commit(value, revision);
        }
        self.step = next_step;
        self.beta1_power = next_beta1_power;
        self.beta2_power = next_beta2_power;
        self.states = candidate_states;

        Ok(observation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod adam_w_config {
        use super::*;

        mod fn_new {
            use super::*;

            const VALID_LEARNING_RATE: f64 = 0.1;
            const VALID_BETA1: f64 = 0.2;
            const VALID_BETA2: f64 = 0.3;
            const VALID_EPSILON: f64 = 3.0;
            const VALID_WEIGHT_DECAY: f64 = 0.4;

            mod learning_rate_validation {
                use super::*;

                fn with_learning_rate(learning_rate: f64) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        learning_rate,
                        VALID_BETA1,
                        VALID_BETA2,
                        VALID_EPSILON,
                        VALID_WEIGHT_DECAY,
                    )
                }

                fn assert_invalid_learning_rate(learning_rate: f64, message: &str) {
                    let result = with_learning_rate(learning_rate);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidLearningRate {
                            value: learning_rate,
                        },
                        "{}",
                        message
                    );
                }

                fn assert_valid_learning_rate(learning_rate: f64) {
                    let result = with_learning_rate(learning_rate);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, learning_rate);
                    assert_eq!(result.beta1, VALID_BETA1);
                    assert_eq!(result.beta2, VALID_BETA2);
                    assert_eq!(result.epsilon, VALID_EPSILON);
                    assert_eq!(result.weight_decay, VALID_WEIGHT_DECAY);
                    assert_eq!(result.decay_product, learning_rate * VALID_WEIGHT_DECAY);
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_learning_rate(f64::INFINITY, "learning_rate must be finite");
                    assert_invalid_learning_rate(-1.0, "learning_rate must gte 0.0");
                    assert_invalid_learning_rate(0.0, "learning_rate must gte 0.0");
                    assert_valid_learning_rate(0.1);
                }
            }

            mod beta1_validation {
                use super::*;

                fn with_beta1(beta1: f64) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        VALID_LEARNING_RATE,
                        beta1,
                        VALID_BETA2,
                        VALID_EPSILON,
                        VALID_WEIGHT_DECAY,
                    )
                }

                fn assert_invalid_beta1(beta1: f64, message: &str) {
                    let result = with_beta1(beta1);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidBeta1 { value: beta1 },
                        "{}",
                        message
                    );
                }

                fn assert_valid_beta1(beta1: f64) {
                    let result = with_beta1(beta1);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, VALID_LEARNING_RATE);
                    assert_eq!(result.beta1, beta1);
                    assert_eq!(result.beta2, VALID_BETA2);
                    assert_eq!(result.epsilon, VALID_EPSILON);
                    assert_eq!(result.weight_decay, VALID_WEIGHT_DECAY);
                    assert_eq!(
                        result.decay_product,
                        VALID_LEARNING_RATE * VALID_WEIGHT_DECAY
                    );
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_beta1(f64::INFINITY, "beta1 must be finite");
                    assert_invalid_beta1(-1.0, "beta1 must in range [0.0, 1.0)");
                    assert_invalid_beta1(1.0, "beta1 must in range [0.0, 1.0)");
                    assert_valid_beta1(VALID_BETA1);
                    assert_valid_beta1(0.0);
                }
            }

            mod beta2_validation {
                use super::*;

                fn with_beta2(beta2: f64) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        VALID_LEARNING_RATE,
                        VALID_BETA1,
                        beta2,
                        VALID_EPSILON,
                        VALID_WEIGHT_DECAY,
                    )
                }

                fn assert_invalid_beta2(beta2: f64, message: &str) {
                    let result = with_beta2(beta2);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidBeta2 { value: beta2 },
                        "{}",
                        message
                    );
                }

                fn assert_valid_beta2(beta2: f64) {
                    let result = with_beta2(beta2);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, VALID_LEARNING_RATE);
                    assert_eq!(result.beta1, VALID_BETA1);
                    assert_eq!(result.beta2, beta2);
                    assert_eq!(result.epsilon, VALID_EPSILON);
                    assert_eq!(result.weight_decay, VALID_WEIGHT_DECAY);
                    assert_eq!(
                        result.decay_product,
                        VALID_LEARNING_RATE * VALID_WEIGHT_DECAY
                    );
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_beta2(f64::INFINITY, "beta2 must be finite");
                    assert_invalid_beta2(-1.0, "beta2 must in range [0.0, 1.0)");
                    assert_invalid_beta2(1.0, "beta2 must in range [0.0, 1.0)");
                    assert_valid_beta2(VALID_BETA2);
                    assert_valid_beta2(0.0);
                }
            }

            mod epsilon_validation {
                use super::*;

                fn with_epsilon(epsilon: f64) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        VALID_LEARNING_RATE,
                        VALID_BETA1,
                        VALID_BETA2,
                        epsilon,
                        VALID_WEIGHT_DECAY,
                    )
                }

                fn assert_invalid_epsilon(epsilon: f64, message: &str) {
                    let result = with_epsilon(epsilon);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidEpsilon { value: epsilon },
                        "{}",
                        message
                    );
                }

                fn assert_valid_epsilon(epsilon: f64) {
                    let result = with_epsilon(epsilon);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, VALID_LEARNING_RATE);
                    assert_eq!(result.beta1, VALID_BETA1);
                    assert_eq!(result.beta2, VALID_BETA2);
                    assert_eq!(result.epsilon, epsilon);
                    assert_eq!(result.weight_decay, VALID_WEIGHT_DECAY);
                    assert_eq!(
                        result.decay_product,
                        VALID_LEARNING_RATE * VALID_WEIGHT_DECAY
                    );
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_epsilon(f64::INFINITY, "epsilon must be finite");
                    assert_invalid_epsilon(-1.0, "epsilon must in gt 0.0");
                    assert_invalid_epsilon(0.0, "epsilon must in gt 0.0");
                    assert_valid_epsilon(VALID_EPSILON);
                }
            }

            mod weight_decay_validation {
                use super::*;

                fn with_weight_decay(weight_decay: f64) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        VALID_LEARNING_RATE,
                        VALID_BETA1,
                        VALID_BETA2,
                        VALID_EPSILON,
                        weight_decay,
                    )
                }

                fn assert_invalid_weight_decay(weight_decay: f64, message: &str) {
                    let result = with_weight_decay(weight_decay);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidWeightDecay {
                            value: weight_decay,
                        },
                        "{}",
                        message
                    );
                }

                fn assert_valid_weight_decay(weight_decay: f64) {
                    let result = with_weight_decay(weight_decay);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, VALID_LEARNING_RATE);
                    assert_eq!(result.beta1, VALID_BETA1);
                    assert_eq!(result.beta2, VALID_BETA2);
                    assert_eq!(result.epsilon, VALID_EPSILON);
                    assert_eq!(result.weight_decay, weight_decay);
                    assert_eq!(result.decay_product, VALID_LEARNING_RATE * weight_decay);
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_weight_decay(f64::INFINITY, "weight_decay must be finite");
                    assert_invalid_weight_decay(-1.0, "weight_decay must in gte 0.0");
                    assert_valid_weight_decay(VALID_WEIGHT_DECAY);
                    assert_valid_weight_decay(0.0);
                }
            }

            mod decay_product_validation {
                use super::*;

                fn with_rate_and_decay(
                    learning_rate: f64,
                    weight_decay: f64,
                ) -> Result<AdamWConfig, AdamWError> {
                    AdamWConfig::new(
                        learning_rate,
                        VALID_BETA1,
                        VALID_BETA2,
                        VALID_EPSILON,
                        weight_decay,
                    )
                }

                fn assert_invalid_decay_product(
                    learning_rate: f64,
                    weight_decay: f64,
                    message: &str,
                ) {
                    let result = with_rate_and_decay(learning_rate, weight_decay);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidDecayProduct {
                            learning_rate,
                            weight_decay,
                            product: learning_rate * weight_decay,
                        },
                        "{}",
                        message
                    );
                }

                fn assert_valid_decay_product(learning_rate: f64, weight_decay: f64) {
                    let result = with_rate_and_decay(learning_rate, weight_decay);
                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.learning_rate, learning_rate);
                    assert_eq!(result.beta1, VALID_BETA1);
                    assert_eq!(result.beta2, VALID_BETA2);
                    assert_eq!(result.epsilon, VALID_EPSILON);
                    assert_eq!(result.weight_decay, weight_decay);
                    assert_eq!(result.decay_product, learning_rate * weight_decay);
                    assert_eq!(result.shrinkage_factor, 1.0 - result.decay_product);
                }

                #[test]
                fn it_validates_it() {
                    assert_invalid_decay_product(1.0, 1.0, "decay_product must be within (0, 1)");
                    assert_invalid_decay_product(
                        1e-256,
                        1e-256,
                        "decay_product must be within (0, 1)",
                    );
                    assert_invalid_decay_product(
                        f64::MAX,
                        f64::MAX,
                        "decay_product must be finite",
                    );
                    assert_valid_decay_product(VALID_LEARNING_RATE, VALID_WEIGHT_DECAY);
                }
            }
        }

        mod fn_with_learning_rate {
            use super::*;

            #[test]
            fn it_computes_new_config_with_the_given_learning_rate() {
                let config = AdamWConfig::new(0.1, 0.2, 0.3, 4.0, 0.5).unwrap();
                let new_learning_rate = 0.12;
                let result = config.with_learning_rate(new_learning_rate);

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.learning_rate, new_learning_rate);
                assert_eq!(result.beta1, config.beta1);
                assert_eq!(result.beta2, config.beta2);
                assert_eq!(result.epsilon, config.epsilon);
                assert_eq!(result.weight_decay, config.weight_decay);
            }
        }
    }

    mod adam_w_gradient_transform {
        use super::*;

        mod fn_validate {
            use super::*;

            mod uniform_variant {
                use super::*;

                #[test]
                fn it_rejects_infinite_scale() {
                    let scale = f64::INFINITY;
                    let g_transform = AdamWGradientTransform::uniform(scale);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientScale { value: scale }
                    );
                }

                #[test]
                fn it_rejects_negative_scale() {
                    let scale = -0.1;
                    let g_transform = AdamWGradientTransform::uniform(scale);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientScale { value: scale }
                    );
                }

                #[test]
                fn it_rejects_scale_gt_1() {
                    let scale = 1.1;
                    let g_transform = AdamWGradientTransform::uniform(scale);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientScale { value: scale }
                    );
                }

                #[test]
                fn it_accepts_1_0_scale() {
                    let scale = 1.0;
                    let g_transform = AdamWGradientTransform::uniform(scale);
                    let result = g_transform.clone().validate();

                    assert!(result.is_ok(), "{result:?}");
                    assert_eq!(result.unwrap(), g_transform);
                }

                #[test]
                fn it_accepts_lt_1_0_scale() {
                    let scale = 0.5;
                    let g_transform = AdamWGradientTransform::uniform(scale);
                    let result = g_transform.clone().validate();

                    assert!(result.is_ok(), "{result:?}");
                    assert_eq!(result.unwrap(), g_transform);
                }
            }

            mod normalized_variant {
                use super::*;

                #[test]
                fn it_rejects_infinite_divisor() {
                    let divisor = f64::INFINITY;
                    let multiplier = 0.5;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_infinite_multiplier() {
                    let divisor = 0.5;
                    let multiplier = f64::INFINITY;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_0_divisor() {
                    let divisor = 0.0;
                    let multiplier = 0.5;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_0_multiplier() {
                    let divisor = 0.5;
                    let multiplier = 0.0;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_negative_divisor() {
                    let divisor = -0.1;
                    let multiplier = 0.5;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_negative_multiplier() {
                    let divisor = 0.5;
                    let multiplier = -0.1;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_rejects_multiplier_gt_divisor() {
                    let divisor = 0.5;
                    let multiplier = 0.6;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.validate();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientNormalization {
                            divisor,
                            multiplier,
                        }
                    );
                }

                #[test]
                fn it_accepts_multiplier_eq_divisor() {
                    let divisor = 0.5;
                    let multiplier = 0.5;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.clone().validate();

                    assert!(result.is_ok(), "{result:?}");
                    assert_eq!(result.unwrap(), g_transform);
                }

                #[test]
                fn it_accepts_multiplier_lt_divisor() {
                    let divisor = 2.5;
                    let multiplier = 0.2;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);
                    let result = g_transform.clone().validate();

                    assert!(result.is_ok(), "{result:?}");
                    assert_eq!(result.unwrap(), g_transform);
                }
            }
        }

        mod fn_apply {
            use super::*;

            mod uniform_variant {
                use super::*;

                mod when_scale_is_1_0 {
                    use super::*;

                    #[test]
                    fn it_returns_gradient_as_is() {
                        let scale = 1.0;
                        let gradient = 0.1;
                        let g_transform = AdamWGradientTransform::uniform(scale);

                        assert_eq!(g_transform.apply(gradient), gradient);
                    }
                }

                mod when_scale_isnt_1_0 {
                    use super::*;

                    #[test]
                    fn it_returns_their_product() {
                        let scale = 0.2;
                        let gradient = 0.1;
                        let g_transform = AdamWGradientTransform::uniform(scale);

                        assert_eq!(g_transform.apply(gradient), gradient * scale);
                    }
                }
            }

            mod normalized_variant {
                use super::*;

                #[test]
                fn it_scales_the_given_gradient() {
                    let divisor = 2.5;
                    let multiplier = 0.2;
                    let gradient = 10.5;
                    let g_transform = AdamWGradientTransform::normalized(divisor, multiplier);

                    assert_eq!(
                        g_transform.apply(gradient),
                        (gradient / divisor) * multiplier
                    );
                }
            }
        }
    }

    mod adam_w_parameter_groups {
        use super::*;

        mod fn_new {
            use super::*;

            #[test]
            fn it_rejects_empty_strings_in_decay_collection() {
                let decay = vec!["", "foo"];
                let no_decay = vec!["bar"];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWError::EmptyGroupedParameterName {
                        group: AdamWGroup::Decay
                    }
                );
            }

            #[test]
            fn it_rejects_empty_strings_in_no_decay_collection() {
                let decay = vec!["bar", "foo"];
                let no_decay = vec![""];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWError::EmptyGroupedParameterName {
                        group: AdamWGroup::NoDecay
                    }
                );
            }

            #[test]
            fn it_rejects_duplicates_in_decay_collection() {
                let decay = vec!["foo", "foo"];
                let no_decay = vec!["bar"];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWError::DuplicateGroupedParameter {
                        group: AdamWGroup::Decay,
                        name: "foo".to_owned()
                    }
                );
            }

            #[test]
            fn it_rejects_duplicates_in_no_decay_collection() {
                let decay = vec!["foo"];
                let no_decay = vec!["bar", "bar"];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWError::DuplicateGroupedParameter {
                        group: AdamWGroup::NoDecay,
                        name: "bar".to_owned()
                    }
                );
            }

            #[test]
            fn it_rejects_decay_and_no_decay_both_empty() {
                let decay: Vec<&str> = vec![];
                let no_decay: Vec<&str> = vec![];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(result.err().unwrap(), AdamWError::EmptyParameterGroups);
            }

            #[test]
            fn it_rejects_same_document_in_both_collections() {
                let decay: Vec<&str> = vec!["foo", "bar"];
                let no_decay: Vec<&str> = vec!["bar", "baz"];
                let result = AdamWParameterGroups::new(decay, no_decay);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWError::ParameterInMultipleGroups {
                        name: "bar".to_owned(),
                    }
                );
            }

            #[test]
            fn it_accepts_only_no_decay_collection() {
                let decay: Vec<&str> = vec![];
                let no_decay: Vec<&str> = vec!["bar", "baz"];
                let result = AdamWParameterGroups::new(decay.clone(), no_decay.clone());

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.decay.iter().collect::<Vec<_>>(), decay);
                assert_eq!(result.no_decay.iter().collect::<Vec<_>>(), no_decay);
            }

            #[test]
            fn it_accepts_only_decay_collection() {
                let decay: Vec<&str> = vec!["bar", "foo"];
                let no_decay: Vec<&str> = vec![];
                let result = AdamWParameterGroups::new(decay.clone(), no_decay.clone());

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.decay.iter().collect::<Vec<_>>(), decay);
                assert_eq!(result.no_decay.iter().collect::<Vec<_>>(), no_decay);
            }

            #[test]
            fn it_accepts_both_collections() {
                let decay: Vec<&str> = vec!["bar", "foo"];
                let no_decay: Vec<&str> = vec!["baz"];
                let result = AdamWParameterGroups::new(decay.clone(), no_decay.clone());

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.decay.iter().collect::<Vec<_>>(), decay);
                assert_eq!(result.no_decay.iter().collect::<Vec<_>>(), no_decay);
            }
        }

        mod fn_parameter_names {
            use super::*;

            #[test]
            fn it_returns_names_from_both_collections() {
                let decay: Vec<&str> = vec!["foo", "bar"];
                let no_decay: Vec<&str> = vec!["baz"];
                let groups = AdamWParameterGroups::new(decay.clone(), no_decay.clone()).unwrap();

                assert_eq!(groups.parameter_names(), vec!["bar", "baz", "foo"]);
            }
        }

        mod fn_decays {
            use super::*;

            mod when_decay_name_exists {
                use super::*;

                #[test]
                fn it_returns_true() {
                    let decay: Vec<&str> = vec!["foo", "bar"];
                    let no_decay: Vec<&str> = vec!["baz"];
                    let groups =
                        AdamWParameterGroups::new(decay.clone(), no_decay.clone()).unwrap();

                    assert!(groups.decays("bar"));
                }
            }

            mod when_decay_name_does_not_exist {
                use super::*;

                #[test]
                fn it_returns_fale() {
                    let decay: Vec<&str> = vec!["foo", "bar"];
                    let no_decay: Vec<&str> = vec!["baz"];
                    let groups =
                        AdamWParameterGroups::new(decay.clone(), no_decay.clone()).unwrap();

                    assert_eq!(groups.decays("baz"), false);
                }
            }
        }
    }

    mod adam_w_moment_state {
        use super::*;

        mod fn_zeros {
            use super::*;

            #[test]
            fn it_prepares_the_state_with_zero_filled_containers() {
                let shape = vec![2, 3];
                let elements = 6;
                let result = AdamWMomentState::zeros(&shape, elements);

                assert_eq!(result.shape, shape);
                assert_eq!(result.first, vec![0.; elements]);
                assert_eq!(result.second, vec![0.; elements]);
            }
        }
    }

    mod adam_w_state_entry {
        use super::*;

        mod fn_new {
            use super::*;

            #[test]
            fn it_rejects_empty_name() {
                let name = "";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., 6.];
                let second_moment = vec![11., 12., 13., 14., 15., 16.];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(result.err().unwrap(), AdamWStateError::EmptyParameterName);
            }

            #[test]
            fn it_rejects_shape_overflow() {
                let name = "bar";
                let shape = vec![2, usize::MAX];
                let first_moment = vec![1., 2., 3., 4., 5., 6.];
                let second_moment = vec![11., 12., 13., 14., 15., 16.];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::ShapeProductOverflow {
                        name: name.to_owned()
                    }
                );
            }

            #[test]
            fn it_rejects_first_moment_elements_size_not_matching_shape_elements() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., 6., 7.];
                let second_moment = vec![11., 12., 13., 14., 15., 16.];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::MomentLengthMismatch {
                        name: name.to_owned(),
                        shape,
                        expected: 6,
                        first: first_moment.len(),
                        second: second_moment.len(),
                    }
                );
            }

            #[test]
            fn it_rejects_second_moment_elements_size_not_matching_shape_elements() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., 6.];
                let second_moment = vec![11., 12., 13., 14., 15., 16., 17.];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::MomentLengthMismatch {
                        name: name.to_owned(),
                        shape,
                        expected: 6,
                        first: first_moment.len(),
                        second: second_moment.len(),
                    }
                );
            }

            #[test]
            fn it_rejects_infinite_values_in_first_moment() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., f64::INFINITY];
                let second_moment = vec![11., 12., 13., 14., 15., 16.];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::NonFiniteMoment {
                        name: name.to_string(),
                        kind: "first",
                        index: 5,
                        value: f64::INFINITY,
                    }
                );
            }

            #[test]
            fn it_rejects_infinite_values_in_second_moment() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., 6.];
                let second_moment = vec![11., 12., 13., 14., 15., f64::INFINITY];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::NonFiniteMoment {
                        name: name.to_string(),
                        kind: "second",
                        index: 5,
                        value: f64::INFINITY,
                    }
                );
            }

            #[test]
            fn it_rejects_negative_values_in_second_moment() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., 6.];
                let second_moment = vec![11., 12., 13., 14., 15., -16.0];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::NegativeSecondMoment {
                        name: name.to_owned(),
                        index: 5,
                        value: -16.0
                    }
                );
            }

            #[test]
            fn it_accepts_negative_values_in_first_moment() {
                let name = "bar";
                let shape = vec![2, 3];
                let first_moment = vec![1., 2., 3., 4., 5., -6.];
                let second_moment = vec![11., 12., 13., 14., 15., 16.0];
                let result = AdamWStateEntry::new(
                    name,
                    shape.clone(),
                    first_moment.clone(),
                    second_moment.clone(),
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.name, name);
                assert_eq!(
                    result.moments,
                    AdamWMomentState {
                        shape,
                        first: first_moment,
                        second: second_moment,
                    }
                );
            }
        }
    }

    mod adam_w_state {
        use super::*;

        mod fn_new {
            use super::*;

            fn adam_w_config() -> AdamWConfig {
                AdamWConfig::new(0.1, 0.2, 0.3, 3.0, 0.4).unwrap()
            }

            #[test]
            fn it_rejects_non_1_0_beta1_power_at_zero_step() {
                let config = adam_w_config();
                let groups = None;
                let step = 0;
                let beta1_power = 0.5;
                let beta2_power = 1.0;
                let entries = vec![];
                let result = AdamWState::new(
                    config,
                    groups,
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::InvalidBetaPower {
                        name: "beta1",
                        value: 0.5,
                        step
                    }
                )
            }

            #[test]
            fn it_rejects_non_1_0_beta2_power_at_zero_step() {
                let config = adam_w_config();
                let groups = None;
                let step = 0;
                let beta1_power = 1.0;
                let beta2_power = 0.5;
                let entries = vec![];
                let result = AdamWState::new(
                    config,
                    groups,
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::InvalidBetaPower {
                        name: "beta2",
                        value: 0.5,
                        step
                    }
                )
            }

            #[test]
            fn it_rejects_empty_entries_at_non_zero_step() {
                let config = adam_w_config();
                let groups = None;
                let step = 1;
                let beta1_power = 0.5;
                let beta2_power = 0.5;
                let entries = vec![];
                let result = AdamWState::new(
                    config,
                    groups,
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::StatePresence {
                        step,
                        entries: entries.len(),
                    }
                )
            }

            #[test]
            fn it_rejects_entries_duplicates() {
                let config = adam_w_config();
                let groups = None;
                let step = 1;
                let beta1_power = 0.5;
                let beta2_power = 0.5;
                let entries = vec![
                    AdamWStateEntry::new("foo", vec![1], vec![1.], vec![2.]).unwrap(),
                    AdamWStateEntry::new("foo", vec![1], vec![2.], vec![3.]).unwrap(),
                ];
                let result = AdamWState::new(
                    config,
                    groups,
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::DuplicateParameterName {
                        name: "foo".to_owned()
                    }
                )
            }

            #[test]
            fn it_rejects_entries_groups_mismatch_for_passed_groups_for_non_zero_step() {
                let config = adam_w_config();
                let groups = Some(AdamWParameterGroups::new(vec!["foo"], vec!["baz"]).unwrap());
                let step = 1;
                let beta1_power = 0.5;
                let beta2_power = 0.5;
                let entries = vec![
                    AdamWStateEntry::new("foo", vec![1], vec![1.], vec![2.]).unwrap(),
                    AdamWStateEntry::new("bar", vec![1], vec![2.], vec![3.]).unwrap(),
                ];
                let result = AdamWState::new(
                    config,
                    groups,
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    AdamWStateError::ParameterGroupsMismatch {
                        expected: vec!["baz".to_owned(), "foo".to_owned()],
                        actual: vec!["bar".to_owned(), "foo".to_owned()]
                    }
                )
            }

            #[test]
            fn it_accepts_empty_entries_for_zero_step() {
                let config = adam_w_config();
                let groups = None;
                let step = 0;
                let beta1_power = 1.0;
                let beta2_power = 1.0;
                let entries = vec![];
                let result = AdamWState::new(
                    config,
                    groups.clone(),
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.config, config);
                assert_eq!(result.groups, groups);
                assert_eq!(result.step, step);
                assert_eq!(result.beta1_power, beta1_power);
                assert_eq!(result.beta2_power, beta2_power);
                assert_eq!(result.states, BTreeMap::new());
            }

            #[test]
            fn it_computes_adam_w_state() {
                let config = adam_w_config();
                let groups = Some(AdamWParameterGroups::new(vec!["foo"], vec!["bar"]).unwrap());
                let step = 1;
                let beta1_power = 0.5;
                let beta2_power = 0.5;
                let entries = vec![
                    AdamWStateEntry::new("foo", vec![1], vec![1.], vec![2.]).unwrap(),
                    AdamWStateEntry::new("bar", vec![1], vec![2.], vec![3.]).unwrap(),
                ];
                let result = AdamWState::new(
                    config,
                    groups.clone(),
                    step,
                    beta1_power,
                    beta2_power,
                    entries.clone(),
                );

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result.config, config);
                assert_eq!(result.groups, groups);
                assert_eq!(result.step, step);
                assert_eq!(result.beta1_power, beta1_power);
                assert_eq!(result.beta2_power, beta2_power);
                assert_eq!(
                    result.states,
                    BTreeMap::from_iter(
                        entries.iter().map(|e| (e.name.clone(), e.moments.clone()))
                    )
                );
            }
        }
    }

    mod adam_w {
        use super::*;

        fn adam_w_config() -> AdamWConfig {
            AdamWConfig::new(0.1, 0.2, 0.3, 3.0, 0.4).unwrap()
        }

        mod fn_new {
            use super::*;

            #[test]
            fn it_computes_adam_w() {
                let config = adam_w_config();
                let result = AdamW::new(config);

                assert_eq!(result.config, config);
                assert_eq!(result.groups, None);
                assert_eq!(result.step, 0);
                assert_eq!(result.beta1_power, 1.0);
                assert_eq!(result.beta2_power, 1.0);
                assert_eq!(result.states, BTreeMap::new());
            }
        }

        mod fn_with_parameter_groups {
            use super::*;

            #[test]
            fn it_computes_adam_w() {
                let config = adam_w_config();
                let groups = AdamWParameterGroups::new(vec!["foo"], vec!["bar"]).unwrap();
                let result = AdamW::with_parameter_groups(config, groups.clone());

                assert_eq!(result.config, config);
                assert_eq!(result.groups, Some(groups));
                assert_eq!(result.step, 0);
                assert_eq!(result.beta1_power, 1.0);
                assert_eq!(result.beta2_power, 1.0);
                assert_eq!(result.states, BTreeMap::new());
            }
        }

        mod fn_step_with_config {
            use super::*;

            mod validations {
                use super::*;

                fn named_parameter1() -> NamedParameter {
                    let tensor = Tensor::from_vec(vec![1, 3], vec![1., 2., 3.]).unwrap();
                    NamedParameter::from_tensor("foo", tensor).unwrap()
                }

                fn named_parameter2() -> NamedParameter {
                    let tensor = Tensor::from_vec(vec![1, 3], vec![4., 5., 6.]).unwrap();
                    NamedParameter::from_tensor("bar", tensor).unwrap()
                }

                fn g_transform() -> AdamWGradientTransform {
                    AdamWGradientTransform::uniform(1.0)
                }

                fn adam_w() -> AdamW {
                    AdamW::new(adam_w_config())
                }

                fn step_config() -> AdamWConfig {
                    AdamWConfig::new(0.11, 0.21, 0.31, 3.1, 0.41).unwrap()
                }

                #[test]
                fn it_rejects_invalid_gradient_transform() {
                    let mut adam_w = adam_w();
                    let parameters = vec![named_parameter1(), named_parameter2()];
                    let step_config = step_config();
                    let g_transform = AdamWGradientTransform::uniform(2.0);
                    let observer = NoAdamWTrace;
                    let result =
                        adam_w.step_with_config(&parameters, step_config, g_transform, observer);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::InvalidGradientScale { value: 2.0 }
                    );
                }

                #[test]
                fn it_rejects_empty_parameters() {
                    let mut adam_w = adam_w();
                    let parameters = vec![];
                    let step_config = step_config();
                    let g_transform = AdamWGradientTransform::uniform(1.0);
                    let observer = NoAdamWTrace;
                    let result =
                        adam_w.step_with_config(&parameters, step_config, g_transform, observer);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), AdamWError::EmptyParameterSet);
                }

                #[test]
                fn it_rejects_duplicated_parameters() {
                    let mut adam_w = adam_w();
                    let parameters = vec![named_parameter1(), named_parameter1()];
                    let step_config = step_config();
                    let g_transform = AdamWGradientTransform::uniform(1.0);
                    let observer = NoAdamWTrace;
                    let result =
                        adam_w.step_with_config(&parameters, step_config, g_transform, observer);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::DuplicateParameterName {
                            name: "foo".to_owned(),
                            first: 0,
                            repeated: 1,
                        }
                    );
                }

                #[test]
                fn it_rejects_inconsistent_parameter_groups() {
                    let mut adam_w = adam_w();
                    adam_w.groups =
                        Some(AdamWParameterGroups::new(vec!["foo"], vec!["baz"]).unwrap());
                    let parameters = vec![named_parameter1(), named_parameter2()];
                    let step_config = step_config();
                    let g_transform = AdamWGradientTransform::uniform(1.0);
                    let observer = NoAdamWTrace;
                    let result =
                        adam_w.step_with_config(&parameters, step_config, g_transform, observer);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::ParameterSetChanged {
                            expected: vec!["baz".to_string(), "foo".to_string()],
                            actual: vec!["bar".to_string(), "foo".to_string()],
                        }
                    );
                }

                #[test]
                fn it_rejects_inconsistency_between_parameters_and_states_on_non_zero_step() {
                    let mut adam_w = adam_w();
                    adam_w.step = 1;
                    adam_w.states = BTreeMap::from([(
                        "baz".to_owned(),
                        AdamWMomentState::zeros(&vec![2, 3], 6),
                    )]);
                    let parameters = vec![named_parameter1(), named_parameter2()];
                    let step_config = step_config();
                    let g_transform = AdamWGradientTransform::uniform(1.0);
                    let observer = NoAdamWTrace;
                    let result =
                        adam_w.step_with_config(&parameters, step_config, g_transform, observer);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        AdamWError::ParameterSetChanged {
                            expected: vec!["baz".to_string()],
                            actual: vec!["bar".to_string(), "foo".to_string()],
                        }
                    );
                }
            }

            mod parameter_updates_based_on_related_gradient {
                use super::*;

                fn param1() -> NamedParameter {
                    let tensor = Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
                    NamedParameter::from_tensor("foo", tensor).unwrap()
                }

                fn param2() -> NamedParameter {
                    let tensor = Tensor::from_vec(vec![2, 2], vec![1.2, 2.3, 3.4, 4.5]).unwrap();
                    NamedParameter::from_tensor("bar", tensor).unwrap()
                }

                #[test]
                fn it_adjust_tensor_values_based_on_their_gradients() {
                    let param1 = param1();
                    let param2 = param2();
                    let sum = param1.tensor().mul(&param2.tensor()).unwrap();
                    let loss = sum.indexed_mean_nll(1, &[0, 1]).unwrap();
                    loss.backward().unwrap();

                    let mut adam_w = AdamW::new(adam_w_config());
                    let parameters = vec![param1, param2];
                    let g_transform = AdamWGradientTransform::uniform(1.0);

                    let first_step = adam_w.step_with_config(
                        &parameters,
                        adam_w_config(),
                        g_transform,
                        NoAdamWTrace,
                    );
                    assert!(first_step.is_ok(), "{first_step:?}");
                    assert_eq!(
                        parameters[0].tensor().value().as_slice(),
                        vec![
                            0.10646150772499645,
                            0.17370337146718345,
                            0.27288138575110477,
                            0.40307682286307467
                        ]
                    );
                    assert_eq!(
                        parameters[1].tensor().value().as_slice(),
                        vec![
                            1.1529642623230465,
                            2.206089893788435,
                            3.2624527176502847,
                            4.322052457362784
                        ]
                    );

                    let second_step = adam_w.step_with_config(
                        &parameters,
                        adam_w_config(),
                        g_transform,
                        NoAdamWTrace,
                    );
                    assert!(second_step.is_ok(), "{first_step:?}");
                    assert_eq!(
                        parameters[0].tensor().value().as_slice(),
                        vec![
                            0.11266455514099304,
                            0.14845860807567957,
                            0.24684751607216537,
                            0.40603057281162636
                        ]
                    );
                    assert_eq!(
                        parameters[1].tensor().value().as_slice(),
                        vec![
                            1.1078099541531712,
                            2.1159361918253325,
                            3.130407326594558,
                            4.151222816431057
                        ]
                    );
                    // println!("{result2:?}");
                }
            }
        }
    }
}
