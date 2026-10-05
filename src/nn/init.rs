//! Deterministic, width-aware construction of named trainable parameters.

use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::TensorOperation::Parameter;
use crate::autograd::tensor_core::{TensorAutodiffError, TensorValue};
use crate::tensor::storage::{Tensor, TensorError};

const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MIX_ONE: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MIX_TWO: u64 = 0x94d0_49bb_1331_11eb;
const BINARY64_UNIT_SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);

/// A deterministic rejection while constructing or collecting a parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum InitializationError {
    EmptyName,
    EmptyNameSegment {
        index: usize,
    },
    InvalidNameCharacter {
        index: usize,
        byte: u8,
    },
    ZeroFanIn,
    ZeroFanOut,
    FanSumOverflow {
        fan_in: usize,
        fan_out: usize,
    },
    ShapeProductOverflow {
        fan_in: usize,
        fan_out: usize,
    },
    AllocationFailed {
        elements: usize,
    },
    DuplicateName {
        name: String,
        first: usize,
        repeated: usize,
    },
    Tensor(TensorError),
    Autodiff(TensorAutodiffError),
}

impl fmt::Display for InitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("parameter name must not be empty"),
            Self::EmptyNameSegment { index } => write!(
                formatter,
                "parameter name has an empty dot-separated segment at byte {index}"
            ),
            Self::InvalidNameCharacter { index, byte } => write!(
                formatter,
                "parameter name byte {index} must be lowercase ASCII, a digit, underscore, or dot; got 0x{byte:02x}"
            ),
            Self::ZeroFanIn => formatter.write_str("fan-in must be greater than zero"),
            Self::ZeroFanOut => formatter.write_str("fan-out must be greater than zero"),
            Self::FanSumOverflow { fan_in, fan_out } => write!(
                formatter,
                "fan-in {fan_in} plus fan-out {fan_out} does not fit usize"
            ),
            Self::ShapeProductOverflow { fan_in, fan_out } => write!(
                formatter,
                "matrix shape [{fan_in},{fan_out}] does not fit usize"
            ),
            Self::AllocationFailed { elements } => write!(
                formatter,
                "could not reserve storage for {elements} initialized values"
            ),
            Self::DuplicateName {
                name,
                first,
                repeated,
            } => write!(
                formatter,
                "parameter name {name:?} first appears at index {first} and repeats at index {repeated}"
            ),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Autodiff(error) => error.fmt(formatter),
        }
    }
}

impl Error for InitializationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::Autodiff(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for InitializationError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorAutodiffError> for InitializationError {
    fn from(error: TensorAutodiffError) -> Self {
        Self::Autodiff(error)
    }
}

/// A small deterministic generator with an explicit resumable 64-bit state.
///
/// This generator is suitable for reproducible teaching fixtures. It is not a cryptographically
/// secure random-number generator
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Treats `seed` as the raw state before the first increment and draw
    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Resumes directly from a previously recorded raw state
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Returns the raw state that the next draw will advance
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Advances and mixes one exactly specified 64-bit value
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_INCREMENT);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MIX_ONE);
        value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MIX_TWO);
        value ^ (value >> 31)
    }

    /// Maps the high 53 bits of one draw to the binary 64 interval [0,1)
    pub fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * BINARY64_UNIT_SCALE
    }
}

/// The formula-derived scale for one [fan-in, fan-out] matrix.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct XavierScale {
    fan_in: usize,
    fan_out: usize,
    target_variance: f64,
    standard_deviation: f64,
    uniform_limit: f64,
}

impl XavierScale {
    pub fn fan_in(&self) -> usize {
        self.fan_in
    }

    pub fn fan_out(&self) -> usize {
        self.fan_out
    }

    pub fn target_variance(&self) -> f64 {
        self.target_variance
    }

    pub fn standard_deviation(&self) -> f64 {
        self.standard_deviation
    }

    pub fn uniform_limit(&self) -> f64 {
        self.uniform_limit
    }
}

/// Calculates the Xavier variance, standard deviation, and uniform bound
pub fn xavier_scale(fan_in: usize, fan_out: usize) -> Result<XavierScale, InitializationError> {
    if fan_in == 0 {
        return Err(InitializationError::ZeroFanIn);
    }
    if fan_out == 0 {
        return Err(InitializationError::ZeroFanOut);
    }

    let fan_sum = fan_in
        .checked_add(fan_out)
        .ok_or(InitializationError::FanSumOverflow { fan_in, fan_out })?;
    let target_variance = 2.0 / fan_sum as f64;

    Ok(XavierScale {
        fan_in,
        fan_out,
        target_variance,
        standard_deviation: target_variance.sqrt(),
        uniform_limit: (6.0 / fan_sum as f64).sqrt(),
    })
}

fn checked_element_count(fan_in: usize, fan_out: usize) -> Result<usize, InitializationError> {
    fan_in
        .checked_mul(fan_out)
        .ok_or(InitializationError::ShapeProductOverflow { fan_in, fan_out })
}

/// Checks whether name has a valid format - dot-split words. Example:
/// `"block_0.attention2.query_weight"`
pub fn validate_name(name: &str) -> Result<(), InitializationError> {
    if name.is_empty() {
        return Err(InitializationError::EmptyName);
    }

    let mut previous_was_dot = true;
    for (index, byte) in name.bytes().enumerate() {
        if byte == b'.' {
            if previous_was_dot {
                return Err(InitializationError::EmptyNameSegment { index });
            }
            previous_was_dot = true;
            continue;
        }

        if !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
            return Err(InitializationError::InvalidNameCharacter { index, byte });
        }
        previous_was_dot = false;
    }

    if previous_was_dot {
        return Err(InitializationError::EmptyNameSegment { index: name.len() });
    }
    Ok(())
}

fn initialized_values(
    rng: &mut SplitMix64,
    scale: XavierScale,
) -> Result<Vec<f64>, InitializationError> {
    let elements = checked_element_count(scale.fan_in, scale.fan_out)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| InitializationError::AllocationFailed { elements })?;

    for _ in 0..elements {
        let centered = 2.0 * rng.next_unit_f64() - 1.0;
        values.push(scale.uniform_limit * centered);
    }

    Ok(values)
}

/// One immutable external name paired with one trainable tensor-tape leaf
#[derive(Clone, Debug, PartialEq)]
pub struct NamedParameter {
    name: String,
    tensor: TensorValue,
}

impl NamedParameter {
    /// Wraps an already-created tensor as a named trainable leaf
    pub fn from_tensor(
        name: impl Into<String>,
        tensor: Tensor,
    ) -> Result<Self, InitializationError> {
        let name = name.into();
        validate_name(&name)?;

        Ok(Self {
            name,
            tensor: TensorValue::parameter(tensor)?,
        })
    }

    /// Samples one [fan-in, fan-out] trainable matrix transactionally
    pub fn xavier_uniform(
        name: impl Into<String>,
        fan_in: usize,
        fan_out: usize,
        rng: &mut SplitMix64,
    ) -> Result<Self, InitializationError> {
        let name = name.into();
        validate_name(&name)?;

        let scale = xavier_scale(fan_in, fan_out)?;
        let elements = checked_element_count(fan_in, fan_out)?;

        let mut trial = rng.clone();
        let values = initialized_values(&mut trial, scale)?;
        debug_assert_eq!(values.len(), elements);

        let tensor = Tensor::from_vec(vec![fan_in, fan_out], values)?;
        let parameter = Self {
            name,
            tensor: TensorValue::parameter(tensor)?,
        };
        *rng = trial;
        Ok(parameter)
    }

    /// Returns the stable external identity used by layers and checkpoints.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrows the trainable tape leaf without duplicating its tensor storage.
    pub fn tensor(&self) -> &TensorValue {
        &self.tensor
    }

    /// Validates a borrowed name and tensor creating a node or copying values
    pub fn validate_leaf(name: &str, tensor: &Tensor) -> Result<(), InitializationError> {
        validate_name(name)?;

        TensorValue::validate_parameter_value(tensor)?;
        Ok(())
    }
}

/// A duplicate-checked, declaration-ordered set of name parameters
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NamedParameters {
    parameters: Vec<NamedParameter>,
}

impl NamedParameters {
    pub fn try_new(parameters: Vec<NamedParameter>) -> Result<Self, InitializationError> {
        for repeated in 0..parameters.len() {
            if let Some(first) = parameters[..repeated]
                .iter()
                .position(|parameter| parameter.name() == parameters[repeated].name())
            {
                return Err(InitializationError::DuplicateName {
                    name: parameters[repeated].name().to_owned(),
                    first,
                    repeated,
                });
            }
        }
        Ok(Self { parameters })
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    pub fn as_slice(&self) -> &[NamedParameter] {
        &self.parameters
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &NamedParameter> {
        self.parameters.iter()
    }

    pub fn get(&self, name: &str) -> Option<&NamedParameter> {
        self.parameters
            .iter()
            .find(|parameter| parameter.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod split_mix64 {
        use super::*;

        mod fn_next_u64 {
            use super::*;

            #[test]
            fn it_generates_u64_values_sequentially() {
                let mut mix = SplitMix64::from_seed(0);

                assert_eq!(mix.next_u64(), 16294208416658607535);
                assert_eq!(mix.next_u64(), 7960286522194355700);
            }
        }

        mod fn_next_unit_f64 {
            use super::*;

            #[test]
            fn it_generates_f64_values_sequentially() {
                let mut mix = SplitMix64::from_seed(0);

                assert_eq!(mix.next_unit_f64(), 0.8833108082136426);
                assert_eq!(mix.next_unit_f64(), 0.43152799704850997);
            }
        }
    }

    mod fn_xavier_scale {
        use super::*;

        mod when_fan_in_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let fan_in = 0;
                let fan_out = 1;
                let result = xavier_scale(fan_in, fan_out);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(result.err().unwrap(), InitializationError::ZeroFanIn);
            }
        }

        mod when_fan_out_is_0 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let fan_in = 1;
                let fan_out = 0;
                let result = xavier_scale(fan_in, fan_out);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(result.err().unwrap(), InitializationError::ZeroFanOut);
            }
        }

        mod when_sum_of_in_and_out_overflows {
            use super::*;

            #[test]
            fn it_returns_error() {
                let fan_in = usize::MAX;
                let fan_out = usize::MAX;
                let result = xavier_scale(fan_in, fan_out);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    InitializationError::FanSumOverflow { fan_in, fan_out }
                );
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_returns_xavier_scale() {
                let fan_in = 2;
                let fan_out = 10;
                let result = xavier_scale(fan_in, fan_out);

                assert!(result.is_ok(), "{result:?}");
                assert_eq!(
                    result.ok().unwrap(),
                    XavierScale {
                        fan_in,
                        fan_out,
                        target_variance: 2.0 / (fan_in + fan_out) as f64,
                        standard_deviation: (2.0 / (fan_in + fan_out) as f64).sqrt(),
                        uniform_limit: (6.0 / (fan_in + fan_out) as f64).sqrt(),
                    }
                );
            }
        }
    }

    mod fn_validate_name {
        use super::*;

        mod when_name_contains_several_dots_in_a_row {
            use super::*;

            #[test]
            fn it_returns_error() {
                let name = "lol..kek";
                let result = validate_name(name);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    InitializationError::EmptyNameSegment { index: 4 }
                )
            }
        }

        mod when_name_end_with_dot {
            use super::*;

            #[test]
            fn it_returns_error() {
                let name = "lol.";
                let result = validate_name(name);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    InitializationError::EmptyNameSegment { index: 4 }
                )
            }
        }

        mod when_name_contains_restricted_characters {
            use super::*;

            #[test]
            fn it_returns_error() {
                let name = "lol-kek";
                let result = validate_name(name);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    InitializationError::InvalidNameCharacter {
                        index: 3,
                        byte: b'-'
                    }
                )
            }
        }

        mod when_name_contains_non_ascii_characters {
            use super::*;

            #[test]
            fn it_returns_error() {
                let name = "lolя";
                let result = validate_name(name);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err().unwrap(),
                    InitializationError::InvalidNameCharacter {
                        index: 3,
                        byte: 0xd1
                    }
                )
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_returns_ok() {
                let name = "lol.kek";
                let result = validate_name(name);

                assert!(result.is_ok(), "{result:?}");
            }
        }
    }

    mod fn_initialized_values {
        use super::*;

        #[test]
        fn it_generates_pseudo_random_values() {
            let scale = xavier_scale(1, 10).unwrap();
            let mut rng = SplitMix64::from_seed(5);
            let result = initialized_values(&mut rng, scale);

            assert!(result.is_ok(), "{result:?}");
            assert_eq!(
                result.ok().unwrap(),
                vec![
                    -0.1672546805560897,
                    0.3726821611688772,
                    -0.39481472786245236,
                    -0.5918149108375877,
                    -0.46091344637590065,
                    -0.1763523013085437,
                    0.7172248574050568,
                    0.016397985596164098,
                    -0.10863857200786675,
                    0.152791825221136
                ]
            );
        }

        #[test]
        fn it_implements_deterministic_behaviour() {
            let fan_in = 1;
            let fan_out = 10;
            let seed = 5;

            let scale1 = xavier_scale(fan_in, fan_out).unwrap();
            let mut rng1 = SplitMix64::from_seed(seed);
            let result1 = initialized_values(&mut rng1, scale1);

            let scale2 = xavier_scale(fan_in, fan_out).unwrap();
            let mut rng2 = SplitMix64::from_seed(seed);
            let result2 = initialized_values(&mut rng2, scale2);

            assert!(result1.is_ok(), "{result1:?}");
            assert!(result2.is_ok(), "{result2:?}");

            assert_eq!(result1.ok().unwrap(), result2.ok().unwrap());
        }
    }

    mod named_parameter {
        use super::*;

        mod fn_from_tensor {
            use super::*;

            mod when_name_is_valid {
                use super::*;

                #[test]
                fn it_computes_named_parameter() {
                    let name = "foo.bar";
                    let tensor = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
                    let result = NamedParameter::from_tensor(name, tensor.clone());

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.name, name);
                    assert_eq!(
                        result.tensor,
                        TensorValue::parameter(tensor.clone()).unwrap()
                    );
                }
            }

            mod when_name_is_invalid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let name = "foo-bar";
                    let tensor = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
                    let result = NamedParameter::from_tensor(name, tensor.clone());

                    assert!(result.is_err(), "{result:?}");
                }
            }
        }

        mod fn_xavier_uniform {
            use super::*;

            mod when_name_is_invalid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let name = "foo-bar";
                    let fan_in = 1;
                    let fan_out = 1;
                    let mut rng = SplitMix64::from_seed(1);
                    let result = NamedParameter::xavier_uniform(name, fan_in, fan_out, &mut rng);

                    assert!(result.is_err(), "{result:?}");
                }
            }

            mod when_fans_are_valid {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let name = "foo.bar";
                    let fan_in = usize::MAX;
                    let fan_out = usize::MAX;
                    let mut rng = SplitMix64::from_seed(1);
                    let result = NamedParameter::xavier_uniform(name, fan_in, fan_out, &mut rng);

                    assert!(result.is_err(), "{result:?}");
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_named_parameter() {
                    let name = "foo.bar";
                    let fan_in = 1;
                    let fan_out = 2;
                    let mut rng = SplitMix64::from_seed(1);
                    let result = NamedParameter::xavier_uniform(name, fan_in, fan_out, &mut rng);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.name, name);
                    assert_eq!(
                        result.tensor,
                        TensorValue::parameter(
                            Tensor::from_vec(
                                vec![fan_in, fan_out],
                                vec![0.18826456468311187, 0.6951757890096079]
                            )
                            .unwrap()
                        )
                        .unwrap()
                    );
                }

                #[test]
                fn it_changes_rng_state_between_runs() {
                    let name = "foo.bar";
                    let fan_in = 1;
                    let fan_out = 2;

                    let mut rng = SplitMix64::from_seed(1);
                    let result1 = NamedParameter::xavier_uniform(name, fan_in, fan_out, &mut rng);
                    let result2 = NamedParameter::xavier_uniform(name, fan_in, fan_out, &mut rng);

                    assert!(result1.is_ok(), "{result1:?}");
                    assert!(result2.is_ok(), "{result2:?}");

                    assert_ne!(
                        result1.unwrap().tensor.value().clone(),
                        result2.unwrap().tensor.value().clone()
                    );
                }
            }
        }

        mod named_parameters {
            use super::*;

            mod fn_try_new {
                use super::*;

                mod when_there_are_parameters_with_duplicated_names {
                    use super::*;

                    #[test]
                    fn it_returns_error() {
                        let tensor = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                        let param1 = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let param2 = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let result = NamedParameters::try_new(vec![param1, param2]);

                        assert!(result.is_err(), "{result:?}");
                        assert_eq!(
                            result.err().unwrap(),
                            InitializationError::DuplicateName {
                                name: "foo".to_owned(),
                                first: 0,
                                repeated: 1,
                            }
                        );
                    }
                }

                mod when_there_are_no_duplicates {
                    use super::*;

                    #[test]
                    fn it_computes_named_parameters() {
                        let tensor1 = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                        let tensor2 = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                        let param1 = NamedParameter::from_tensor("foo", tensor1.clone()).unwrap();
                        let param2 = NamedParameter::from_tensor("bar", tensor2.clone()).unwrap();
                        let result = NamedParameters::try_new(vec![param1, param2]);

                        assert!(result.is_ok(), "{result:?}");
                        let result = result.as_ref().unwrap();
                        assert_eq!(result.len(), 2);

                        assert_eq!(result.parameters[0].name, "foo");
                        assert_eq!(result.parameters[0].tensor.value().clone(), tensor1);

                        assert_eq!(result.parameters[1].name, "bar");
                        assert_eq!(result.parameters[1].tensor.value().clone(), tensor2);
                    }
                }
            }

            mod fn_get {
                use super::*;

                mod when_parameter_with_the_given_name_exists {
                    use super::*;

                    #[test]
                    fn it_finds_it() {
                        let tensor = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                        let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let params = NamedParameters::try_new(vec![param]).unwrap();
                        let result = params.get("foo");

                        assert!(result.is_some(), "{result:?}");
                        let result = result.unwrap();
                        assert_eq!(result.name, "foo");
                        assert_eq!(result.tensor.value().clone(), tensor);
                    }
                }

                mod when_parameter_with_the_given_name_does_not_exist {
                    use super::*;

                    #[test]
                    fn it_returns_none() {
                        let tensor = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                        let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                        let params = NamedParameters::try_new(vec![param]).unwrap();
                        let result = params.get("bar");

                        assert!(result.is_none(), "{result:?}");
                    }
                }
            }
        }
    }
}
