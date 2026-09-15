//! Stable probability and indexed-loss operations over checked tensor views.

use std::error::Error;
use std::fmt;

use crate::tensor::storage::{Tensor, TensorError, checked_row_major_layout};
use crate::tensor::view::{StridedOffsets, TensorView, TensorViewError};

/// A rejected probability operation, target, output, or converted view operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbabilityError {
    /// An owned output layout violates the tensor storage invariant.
    Tensor(TensorError),
    /// A tensor-view error was converted into the probability error type.
    View(TensorViewError),
    /// The requested class axis does not exist.
    AxisOutOfBounds { axis: usize, rank: usize },
    /// Softmax, log-softmax, and indexed NLL need at least one class.
    EmptyNormalizationAxis { axis: usize },
    /// The checked output shape is valid, but its value buffer cannot be reserved.
    OutputAllocationFailed { elements: usize },
    /// The first rejected logit in group-major, class-minor order is NaN.
    NaNLogit { group: usize, class: usize },
    /// The first rejected logit in group-major, class-minor order is positive infinity.
    PositiveInfinityLogit { group: usize, class: usize },
    /// The first rejected logit in group-major, class-minor order is negative infinity.
    NegativeInfinityLogit { group: usize, class: usize },
    /// There must be one flat target for every class-axis group.
    TargetCountMismatch { expected: usize, actual: usize },
    /// A mean is undefined when there are no target groups.
    EmptyTargets,
    /// One target does not name a class on the selected axis.
    TargetOutOfBounds {
        group: usize,
        target: usize,
        classes: usize,
    },
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(formatter),
            Self::View(error) => error.fmt(formatter),
            Self::AxisOutOfBounds { axis, rank } => {
                write!(
                    formatter,
                    "probability axis {axis} is out of bounds for rank {rank}"
                )
            }
            Self::EmptyNormalizationAxis { axis } => {
                write!(formatter, "probability axis {axis} has no classes")
            }
            Self::OutputAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate probability output for {elements} f64 values"
            ),
            Self::NaNLogit { group, class } => {
                write!(formatter, "logit at group {group}, class {class} is NaN")
            }
            Self::PositiveInfinityLogit { group, class } => write!(
                formatter,
                "logit at group {group}, class {class} is positive infinity"
            ),
            Self::NegativeInfinityLogit { group, class } => write!(
                formatter,
                "logit at group {group}, class {class} is negative infinity"
            ),
            Self::TargetCountMismatch { expected, actual } => write!(
                formatter,
                "indexed mean NLL needs {expected} targets, but received {actual}"
            ),
            Self::EmptyTargets => formatter.write_str("indexed mean NLL needs at least one target"),
            Self::TargetOutOfBounds {
                group,
                target,
                classes,
            } => write!(
                formatter,
                "target {target} at group {group} is out of bounds for {classes} classes"
            ),
        }
    }
}

impl Error for ProbabilityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::View(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for ProbabilityError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorViewError> for ProbabilityError {
    fn from(error: TensorViewError) -> Self {
        Self::View(error)
    }
}

#[derive(Debug, PartialEq)]
struct AxisPlan {
    axis: usize,
    classes: usize,
    group_shape: Vec<usize>,
    group_strides: Vec<usize>,
    groups: usize,
    class_stride: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct RowStats {
    maximum: f64,
    shifted_exponential_sum: f64,
    log_shifted_exponential_sum: f64,
}

#[derive(Copy, Clone, Debug)]
struct FiniteLogits;

#[derive(Copy, Clone, Debug)]
enum LogitFiniteness {
    Check,
    Validated(FiniteLogits),
}

/// Precomputes the traversal layout for operations that process one tensor axis independently for
/// every combination of the remaining axes.
///
/// The selected axis is treated as the class axis. Its extent and stride are stored separately in
/// `classes` and `class_stride`. Removing that axis from the input shape and strides produces
/// `group_shape` and `group_strides`, which describe how to enumerate the base offset of each
/// independent group.
///
/// For an input with shape `[2, 3, 4]` and `axis = 1`, the plan describes `2 * 4 = 8` groups, each
/// containing `3` classes: `input[i, :, k]`. Other words - AxisPlan describes the layout for
/// vertical traversal by the selected axis instead, more common, horizontal traversal. Example:
/// shape   = [2, 3, 2]
/// strides = [6, 2, 1]
/// axis    = 1
/// data:
///   [
///     [
///       [a, b],
///       [c, d],
///       [e, f]
///     ],
///     [
///       [g, h],
///       [i, j],
///       [k, l]
///     ]
/// ]
/// Results in:
/// [
///   [
///     [a, c, e],
///     [b, d, f]
///   ],
///   [
///     [g, i, k],
///     [h, j, l]
///   ]
/// ]
impl AxisPlan {
    /// Builds a checked traversal plan for processing `input` along `axis`.
    ///
    /// The selected axis is separated from the remaining axes:
    ///
    /// - `classes` stores the extent of the selected axis.
    /// - `class_stride` stores the storage stride used to advance between classes
    /// - `group_shape` and `group_strides` describe all remaining axes and are used to enumerate
    ///   one base storage offset per independent group.
    /// - `groups` is the number of such remaining-axis groups.
    ///
    /// `allow_empty_axis` controls whether a selected axis with zero extent is accepted. This is
    /// required by reductions such as log-sum-exp, which have a defined identity for an empty
    /// normalization axis.
    fn new(
        input: &TensorView<'_>,
        axis: usize,
        allow_empty_axis: bool,
    ) -> Result<Self, ProbabilityError> {
        if axis >= input.rank() {
            return Err(ProbabilityError::AxisOutOfBounds {
                axis,
                rank: input.rank(),
            });
        }

        let classes = input.shape()[axis];
        if classes == 0 && !allow_empty_axis {
            return Err(ProbabilityError::EmptyNormalizationAxis { axis });
        }

        let mut group_shape = input.shape().to_vec();
        group_shape.remove(axis);
        let mut group_strides = input.strides().to_vec();
        let class_stride = group_strides.remove(axis);
        let (_, groups) = checked_row_major_layout(&group_shape)?;

        Ok(Self {
            axis,
            classes,
            group_shape,
            group_strides,
            groups,
            class_stride,
        })
    }

    /// Returns the storage base offset of every remaining-axis group in `input`.
    ///
    /// Each returned offset corresponds to the selected class axis being at position zero. Classes
    /// within that group are reached by repeatedly adding `class_stride`.
    ///
    /// Group bases are emitted in logical row-major order over `group_shape`.
    fn group_offsets(&self, input: &TensorView<'_>) -> StridedOffsets {
        input
            .projected_offsets(&self.group_shape, &self.group_strides, self.groups)
            .expect("a checked probability plan retains valid group-base offsets")
    }

    /// Returns the base offset of every group in a row-major output layout.
    ///
    /// The selected class axis is removed from `output_strides` so that traversal visits each
    /// remaining-axis group exactly once. The returned offset denotes class zero of that output
    /// group; subsequent classes are addressed using the output's class-axis stride.
    ///
    /// `output_len` is used to validate that the derived traversal remains within the output
    /// buffer.
    fn output_group_offsets(&self, output_strides: &[usize], output_len: usize) -> StridedOffsets {
        let mut group_strides = output_strides.to_vec();
        group_strides.remove(self.axis);
        StridedOffsets::checked(
            &self.group_shape,
            &group_strides,
            0,
            self.groups,
            output_len,
        )
        .expect("a checked probability output retains valid group-base offsets")
    }

    /// Computes the storage offset of one class within a group.
    ///
    /// `group_base` is the storage offset for class zero of the group, and `target` is the class
    /// index along the selected axis. The target is reached by advancing `target * class_stride`
    /// elements from the group base.
    fn target_offset(&self, group_base: usize, target: usize) -> usize {
        let class_offset = target
            .checked_mul(self.class_stride)
            .expect("a checked probability plan cannot overflow a class offset");
        group_base
            .checked_add(class_offset)
            .expect("a checked probability plan cannot overflow a target offset")
    }

    /// Visits every remaining-axis group after computing its stable row statistics.
    ///
    /// Groups are traversed in logical row-major order. For each group, this method obtains its
    /// base storage offset, computes statistics across the selected class axis, and invokes `visit`
    /// with the flat group index, group base offset, and computed statistics.
    fn for_each_group(
        &self,
        input: &TensorView<'_>,
        finiteness: LogitFiniteness,
        mut visit: impl FnMut(usize, usize, RowStats) -> Result<(), ProbabilityError>,
    ) -> Result<(), ProbabilityError> {
        for (group, group_base) in self.group_offsets(input).enumerate() {
            let stats = row_stats(input, self, finiteness, group, group_base)?;
            visit(group, group_base, stats)?;
        }
        Ok(())
    }
}

fn row_stats(
    input: &TensorView<'_>,
    plan: &AxisPlan,
    finiteness: LogitFiniteness,
    group: usize,
    group_base: usize,
) -> Result<RowStats, ProbabilityError> {
    let mut maximum = f64::NEG_INFINITY;
    let mut input_offset = group_base;
    for class in 0..plan.classes {
        let value = input.value_at_storage_offset(input_offset);
        let value = match finiteness {
            LogitFiniteness::Check => checked_finite_logit(value, group, class)?,
            LogitFiniteness::Validated(_) => value,
        };
        maximum = maximum.max(value);
        if class + 1 < plan.classes {
            input_offset = input_offset
                .checked_add(plan.class_stride)
                .expect("a checked probability plan cannot overflow along the class axis");
        }
    }

    let mut exponential_tail = 0.0;
    let mut max_values_count: f64 = 0.;
    input_offset = group_base;
    for class in 0..plan.classes {
        let value = input.value_at_storage_offset(input_offset);
        let shifted = value - maximum;
        if shifted == 0.0 {
            max_values_count += 1.0;
        } else {
            exponential_tail += shifted.exp();
        }
        if class + 1 < plan.classes {
            input_offset = input_offset
                .checked_add(plan.class_stride)
                .expect("a checked probability plan cannot overflow along the class axis");
        }
    }

    Ok(RowStats {
        maximum,
        shifted_exponential_sum: max_values_count + exponential_tail,
        log_shifted_exponential_sum: max_values_count.ln()
            + (exponential_tail / max_values_count).ln_1p(),
    })
}

fn checked_finite_logit(value: f64, group: usize, class: usize) -> Result<f64, ProbabilityError> {
    if value.is_nan() {
        Err(ProbabilityError::NaNLogit { group, class })
    } else if value == f64::INFINITY {
        Err(ProbabilityError::PositiveInfinityLogit { group, class })
    } else if value == f64::NEG_INFINITY {
        Err(ProbabilityError::NegativeInfinityLogit { group, class })
    } else {
        Ok(value)
    }
}

fn output_buffer(elements: usize) -> Result<Vec<f64>, ProbabilityError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| ProbabilityError::OutputAllocationFailed { elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Requested normalized values emitted by one checked forward traversal.
#[derive(Debug, PartialEq)]
struct NormalizedForward {
    probabilities: Option<Tensor>,
    log_probabilities: Option<Tensor>,
}

/// Log-softmax output and optional probabilities from the same forward traversal.
pub struct LogSoftMaxForward {
    pub value: Tensor,
    pub probabilities: Option<Tensor>,
}

/// Indexed mean NLL and optional probabilities emitted by its forward traversal.
pub struct IndexedMeanNllForward {
    pub loss: f64,
    pub probabilities: Option<Tensor>,
}

/// Describes where and how to write the normalized values of one class group.
///
/// A group contains all values along the selected class axis for one fixed combination of the
/// remaining tensor coordinates.
///
/// `output_group_base` identifies the first class of the group in the output layout, while
/// `output_class_stride` specifies how to advance between classes.
///
/// Depending on the requested operation, normalized probabilities, log-probabilities, or both may
/// be emitted during the same traversal.
struct NormalizedGroupOutput<'a> {
    output_group_base: usize,
    output_class_stride: usize,
    probabilities: Option<&'a mut [f64]>,
    log_probabilities: Option<&'a mut [f64]>,
}

impl NormalizedGroupOutput<'_> {
    /// Writes the normalized values of one class group to the requested outputs.
    ///
    /// The input group is traversed from `input_group_base` using `plan.class_stride`, while
    /// output values are written from `output_group_base` using `output_class_stride`.
    ///
    /// `stats` contains the precomputed maximum and max-shifted normalization terms for the group.
    /// These values are reused to compute softmax probabilities and/or log-softmax values without
    /// another reduction pass.
    ///
    /// At least one of `probabilities` or `log_probabilities` must by present.
    fn emit(
        mut self,
        input: &TensorView<'_>,
        plan: &AxisPlan,
        input_group_base: usize,
        stats: RowStats,
    ) {
        let mut input_offset = input_group_base;
        let mut output_offset = self.output_group_base;
        for class in 0..plan.classes {
            let shifted = input.value_at_storage_offset(input_offset) - stats.maximum;
            if let Some(values) = self.probabilities.as_mut() {
                values[output_offset] =
                    positive_zero(shifted.exp() / stats.shifted_exponential_sum);
            }
            if let Some(values) = self.log_probabilities.as_mut() {
                values[output_offset] = positive_zero(shifted - stats.log_shifted_exponential_sum);
            }

            if class + 1 < plan.classes {
                input_offset = input_offset
                    .checked_add(plan.class_stride)
                    .expect("a checked probability plan cannot overflow along the class axis");
                output_offset = output_offset
                    .checked_add(self.output_class_stride)
                    .expect("a checked probability output cannot overflow along the class axis");
            }
        }
    }
}

/// Reduces one axis with max-shifted log-sum-exp.
///
/// An empty selected axis returns the log-additive identity, negative infinity, once per
/// remaining-axis group. Other non-finite logits are rejected in group-major, class-minor order.
pub fn log_sum_exp(
    input: &TensorView<'_>,
    axis: usize,
    keep_dim: bool,
) -> Result<Tensor, ProbabilityError> {
    let plan = AxisPlan::new(input, axis, true)?;
    let output_shape = if keep_dim {
        let mut shape = input.shape().to_vec();
        shape[axis] = 1;
        shape
    } else {
        plan.group_shape.clone()
    };
    let (_, output_len) = checked_row_major_layout(&output_shape)?;
    let mut values = output_buffer(output_len)?;

    if plan.classes == 0 {
        values.fill(f64::NEG_INFINITY);
    } else {
        plan.for_each_group(
            input,
            LogitFiniteness::Check,
            |group, _group_base, stats| {
                values[group] = stats.maximum + stats.log_shifted_exponential_sum;
                Ok(())
            },
        )?
    }

    Tensor::from_vec(output_shape, values).map_err(Into::into)
}

fn validate_finite_logits(
    input: &TensorView<'_>,
    plan: &AxisPlan,
) -> Result<FiniteLogits, ProbabilityError> {
    for (group, group_base) in plan.group_offsets(input).enumerate() {
        let mut input_offset = group_base;
        for class in 0..plan.classes {
            checked_finite_logit(input.value_at_storage_offset(input_offset), group, class)?;
            if class + 1 < plan.classes {
                input_offset = input_offset
                    .checked_add(plan.class_stride)
                    .expect("a checked probability plan cannot overflow along the class axis");
            }
        }
    }
    Ok(FiniteLogits)
}

fn normalized_forward(
    input: &TensorView<'_>,
    axis: usize,
    emit_probabilities: bool,
    emit_log_probabilities: bool,
) -> Result<NormalizedForward, ProbabilityError> {
    let plan = AxisPlan::new(input, axis, false)?;
    let (output_strides, output_len) = checked_row_major_layout(input.shape())?;
    let mut log_probability_values = emit_log_probabilities
        .then(|| output_buffer(output_len))
        .transpose()?;
    let finiteness = if emit_probabilities && emit_log_probabilities {
        LogitFiniteness::Validated(validate_finite_logits(input, &plan)?)
    } else {
        LogitFiniteness::Check
    };
    let mut probability_values = emit_probabilities
        .then(|| output_buffer(output_len))
        .transpose()?;
    let output_class_stride = output_strides[axis];

    let mut output_group_offsets = plan.output_group_offsets(&output_strides, output_len);
    plan.for_each_group(input, finiteness, |_group, input_group_base, stats| {
        let output_group_base = output_group_offsets
            .next()
            .expect("a checked probability output has one base per input group");
        NormalizedGroupOutput {
            output_group_base,
            output_class_stride,
            probabilities: probability_values.as_deref_mut(),
            log_probabilities: log_probability_values.as_deref_mut(),
        }
        .emit(input, &plan, input_group_base, stats);
        Ok(())
    })?;

    let log_probabilities = log_probability_values
        .map(|values| Tensor::from_vec(input.shape().to_vec(), values))
        .transpose()?;
    let probabilities = probability_values
        .map(|values| Tensor::from_vec(input.shape().to_vec(), values))
        .transpose()?;
    Ok(NormalizedForward {
        probabilities,
        log_probabilities,
    })
}

/// Converts finite logits to normalized probabilities along one explicit axis.
pub fn softmax(input: &TensorView<'_>, axis: usize) -> Result<Tensor, ProbabilityError> {
    let forward = normalized_forward(input, axis, true, false)?;
    Ok(forward
        .probabilities
        .expect("softmax requests a probability output"))
}

/// Converts finite logits to normalized log-probabilities along one explicit axis.
pub fn log_softmax(input: &TensorView<'_>, axis: usize) -> Result<Tensor, ProbabilityError> {
    let forward = log_softmax_forward(input, axis, false)?;
    Ok(forward.value)
}

pub fn log_softmax_forward(
    input: &TensorView<'_>,
    axis: usize,
    emit_probabilities: bool,
) -> Result<LogSoftMaxForward, ProbabilityError> {
    let forward = normalized_forward(input, axis, emit_probabilities, true)?;
    Ok(LogSoftMaxForward {
        value: forward
            .log_probabilities
            .expect("log-softmax forward requests a log-probability output"),
        probabilities: forward.probabilities,
    })
}

/// Computes the mean negative log-likelihood for one target class per class-axis group, optionally
/// emitting softmax probabilities from the same forward traversal.
///
/// `axis` identifies the class dimension of `logits`. Every combination of the remaining axes forms
/// one independent group.
///
/// `targets` contains one ground-truth class index for each group, in row-major group order. Each
/// target is an index along the selected class axis rather than a logit value. Therefore, every
/// target must satisfy `target < logits.shape()[axis]`.
///
/// For example, if `logits` has shape `[2, 3, 4]` and `axis == 1`, the class dimension has size
/// `3`, while the remaining axes form `2 * 4 = 8` groups. In that case, `targets` must contain
/// exactly `8` indices, each in the range `0..3`.
///
/// For each group, the loss is computed in the log domain as
///
/// `logsumexp(logits) - logits[target]`
///
/// using max-shifted statistics to avoid overflow and unnecessary probability underflow. The final
/// mean uses a scaled accumulation fallback when summing otherwise finite per-group losses would
/// overflow.
///
/// When `emit_probabilities` is `true`, softmax probabilities are produced alongside the loss using
/// the already computed group statistics. The probability tensor has the same shape as `logits` and
/// is materialized as a contiguous row-major tensor.
pub fn indexed_mean_nll_forward(
    logits: &TensorView<'_>,
    axis: usize,
    targets: &[usize],
    emit_probabilities: bool,
) -> Result<IndexedMeanNllForward, ProbabilityError> {
    let plan = AxisPlan::new(logits, axis, false)?;
    if targets.len() != plan.groups {
        return Err(ProbabilityError::TargetCountMismatch {
            expected: plan.groups,
            actual: targets.len(),
        });
    }
    if targets.is_empty() {
        return Err(ProbabilityError::EmptyTargets);
    }
    for (group, &target) in targets.iter().enumerate() {
        if target >= plan.classes {
            return Err(ProbabilityError::TargetOutOfBounds {
                group,
                target,
                classes: plan.classes,
            });
        }
    }

    let finiteness = if emit_probabilities {
        LogitFiniteness::Validated(validate_finite_logits(logits, &plan)?)
    } else {
        LogitFiniteness::Check
    };

    let output_layout = emit_probabilities
        .then(|| checked_row_major_layout(logits.shape()))
        .transpose()?;
    let mut probability_values = output_layout
        .as_ref()
        .map(|(_, output_len)| output_buffer(*output_len))
        .transpose()?;
    let mut output_group_offsets = output_layout
        .as_ref()
        .map(|(output_strides, output_len)| plan.output_group_offsets(output_strides, *output_len));
    let output_class_stride = output_layout
        .as_ref()
        .map(|(output_strides, _)| output_strides[axis]);

    let mut total = 0.0;
    let mut scaled_mean = 0.0;
    let mut needs_scaled_fallback = false;
    let target_count = targets.len() as f64;

    plan.for_each_group(logits, finiteness, |group, group_base, stats| {
        let target = targets[group];
        let target_logit = logits.value_at_storage_offset(plan.target_offset(group_base, target));
        let gap = stats.maximum - target_logit;
        let scaled_gap = if gap.is_finite() {
            gap / target_count
        } else {
            stats.maximum / target_count - target_logit / target_count
        };

        scaled_mean += scaled_gap + stats.log_shifted_exponential_sum / target_count;

        let loss = gap + stats.log_shifted_exponential_sum;
        if loss.is_finite() && !needs_scaled_fallback {
            total += loss;
            if !total.is_finite() {
                needs_scaled_fallback = true;
            }
        } else {
            needs_scaled_fallback = true;
        }

        if let Some(values) = probability_values.as_deref_mut() {
            let output_group_base = output_group_offsets
                .as_mut()
                .and_then(Iterator::next)
                .expect("a checked probability output has one base per input group");
            NormalizedGroupOutput {
                output_group_base,
                output_class_stride: output_class_stride
                    .expect("a requested probability output has a class stride"),
                probabilities: Some(values),
                log_probabilities: None,
            }
            .emit(logits, &plan, group_base, stats)
        }
        Ok(())
    })?;

    let loss = positive_zero(if needs_scaled_fallback {
        scaled_mean
    } else {
        total / target_count
    });
    let probabilities = probability_values
        .map(|values| Tensor::from_vec(logits.shape().to_vec(), values))
        .transpose()?;
    Ok(IndexedMeanNllForward {
        loss,
        probabilities,
    })
}

/// Scores one class index per remaining-axis group with fused stable mean NLL.
///
/// Targets follow the row-major group shape obtained by removing `axis` from the logits. Bounds are
/// checked for every target before a logit is read.
pub fn indexed_mean_nll(
    logits: &TensorView<'_>,
    axis: usize,
    targets: &[usize],
) -> Result<f64, ProbabilityError> {
    let forward = indexed_mean_nll_forward(logits, axis, targets, false)?;
    Ok(forward.loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip::macros(vec)]
    fn tensor() -> Tensor {
        Tensor::from_vec(
            vec![2, 3, 4],
            vec![
                1.,  2.,  3.,  4.,
                5.,  6.,  7.,  8.,
                9.,  10., 11., 12.,

                13., 14., 15., 16.,
                17., 18., 19., 20.,
                21., 22., 23., 24.,
            ],
        )
        .unwrap()
    }

    #[test]
    fn t() {
        let tensor = Tensor::from_vec(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        )
        .unwrap();
        let view = tensor.view();
        let plan = AxisPlan::new(&view, 1, false).unwrap();

        println!("{:?}", plan);
        println!("{:?}", plan.group_offsets(&view));
        println!("{:?}", plan.group_offsets(&view).collect::<Vec<_>>());
        plan.for_each_group(
            &view,
            LogitFiniteness::Validated(FiniteLogits),
            |group, group_base, stats| {
                println!(
                    "group: {:?}, group_base: {:?}, stats: {:?}",
                    group, group_base, stats
                );
                Ok(())
            },
        )
        .unwrap();
    }

    mod fn_row_stats {
        use super::*;

        mod when_exp_is_too_low {
            use super::*;

            #[rustfmt::skip::macros(vec)]
            #[test]
            fn it_does_not_lose_it() {
                let tensor = Tensor::from_vec(
                    vec![1, 4, 1],
                    vec![
                        100.0,
                        100.0,
                        64.0,
                        64.0,
                    ],
                )
                .unwrap();
                let view = tensor.view();
                let plan = AxisPlan::new(&view, 1, false).unwrap();
                let result = row_stats(&view, &plan, LogitFiniteness::Check, 0, 0).unwrap();

                assert_eq!(
                    result,
                    RowStats {
                        maximum: 100.0,
                        shifted_exponential_sum: 2.0000000000000004,
                        log_shifted_exponential_sum: 0.6931471805599455
                    }
                );
            }
        }
    }

    mod axis_plan {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_axis_is_out_of_bounce {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();
                    let result = AxisPlan::new(&view, 3, false);

                    assert_eq!(
                        result,
                        Err(ProbabilityError::AxisOutOfBounds { axis: 3, rank: 3 })
                    );
                }
            }

            mod when_empty_axis_is_given {
                use super::*;

                mod when_empty_axis_is_not_allowed {
                    use super::*;

                    #[test]
                    fn it_returns_error() {
                        let tensor = Tensor::from_vec(vec![3, 0, 2], vec![]).unwrap();
                        let view = tensor.view();
                        let result = AxisPlan::new(&view, 1, false);

                        assert_eq!(
                            result,
                            Err(ProbabilityError::EmptyNormalizationAxis { axis: 1 })
                        );
                    }
                }

                mod when_empty_axis_is_allowed {
                    use super::*;

                    #[test]
                    fn it_computes_axis_plan() {
                        let tensor = Tensor::from_vec(vec![3, 0, 2], vec![]).unwrap();
                        let view = tensor.view();
                        let result = AxisPlan::new(&view, 1, true);

                        assert_eq!(
                            result,
                            Ok(AxisPlan {
                                axis: 1,
                                classes: 0,
                                group_shape: vec![3, 2],
                                group_strides: vec![0, 1],
                                groups: 6,
                                class_stride: 2
                            })
                        );
                    }
                }
            }

            mod when_non_empty_axis_is_given {
                use super::*;

                #[test]
                fn it_computes_axis_plan() {
                    let tensor = tensor();
                    let view = tensor.view();
                    let result = AxisPlan::new(&view, 1, false);

                    assert_eq!(
                        result,
                        Ok(AxisPlan {
                            axis: 1,
                            classes: 3,
                            group_shape: vec![2, 4],
                            group_strides: vec![12, 1],
                            groups: 8,
                            class_stride: 4
                        })
                    );
                }
            }
        }

        mod fn_group_offsets {
            use super::*;

            #[test]
            fn it_calculates_group_offsets() {
                let tensor = tensor();
                let view = tensor.view();
                let plan = AxisPlan::new(&view, 1, false).unwrap();

                assert_eq!(
                    plan.group_offsets(&view).collect::<Vec<_>>(),
                    vec![0, 1, 2, 3, 12, 13, 14, 15]
                );
            }
        }

        mod fn_output_group_offsets {
            use super::*;

            #[test]
            fn it_calculates_group_offsets_for_the_output_tensor() {
                let tensor = tensor();
                // shape: [2, 1, 4], strides: [12, 4, 1]
                let view = tensor.view().slice(1, 1..2).unwrap();
                let plan = AxisPlan::new(&view, 1, false).unwrap();

                // strides: [8, 4, 1]
                let (output_strides, output_len) = checked_row_major_layout(view.shape()).unwrap();
                let result = plan
                    .output_group_offsets(&output_strides, output_len)
                    .collect::<Vec<_>>();
                assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }

        mod fn_target_offset {
            use super::*;

            #[test]
            fn it_calculates_target_offset_in_the_given_group() {
                let tensor = tensor();
                let view = tensor.view();
                let plan = AxisPlan::new(&view, 1, false).unwrap();

                assert_eq!(plan.target_offset(15, 2), 23);
            }
        }

        mod fn_for_each_group {
            use super::*;

            #[test]
            fn it_yields_stats_of_each_group() {
                let tensor = tensor();
                let view = tensor.view();
                let plan = AxisPlan::new(&view, 1, false).unwrap();
                let mut result = vec![];

                plan.for_each_group(
                    &view,
                    LogitFiniteness::Validated(FiniteLogits),
                    |group, group_base, stats| {
                        result.push((group, group_base, stats));
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(
                    result,
                    vec![
                        (
                            0,
                            0,
                            RowStats {
                                maximum: 9.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            1,
                            1,
                            RowStats {
                                maximum: 10.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            2,
                            2,
                            RowStats {
                                maximum: 11.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            3,
                            3,
                            RowStats {
                                maximum: 12.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            4,
                            12,
                            RowStats {
                                maximum: 21.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            5,
                            13,
                            RowStats {
                                maximum: 22.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            6,
                            14,
                            RowStats {
                                maximum: 23.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        ),
                        (
                            7,
                            15,
                            RowStats {
                                maximum: 24.0,
                                shifted_exponential_sum: 1.0186511015166366,
                                log_shifted_exponential_sum: 0.018479302594657483
                            }
                        )
                    ]
                );
            }
        }
    }

    mod normalized_group_output {
        use super::*;

        mod fn_emit {
            use super::*;

            #[test]
            fn it_calculates_and_persists_normalized_probabilities() {
                let tensor = tensor();
                let view = tensor.view();
                let mut probabilities = vec![0.0; view.len()];
                let mut log_probabilities = vec![0.0; view.len()];
                let output_group_base = 1;
                let output_class_stride = 2;
                let normalized_group_output = NormalizedGroupOutput {
                    output_group_base,
                    output_class_stride,
                    probabilities: Some(&mut probabilities),
                    log_probabilities: Some(&mut log_probabilities),
                };
                let plan = AxisPlan::new(&view, 1, false).unwrap();
                let maximum = 12.0;
                // Every row has same shifted exp and log shifted exp sums because all values are
                // shifted by the same constant, thus, producing the same shifted value
                let shifted_exponential_sum = 1.0186511015166366;
                let log_shifted_exponential_sum = 0.018479302594657483;
                let row_stats = RowStats {
                    maximum: 12.0,
                    shifted_exponential_sum,
                    log_shifted_exponential_sum,
                };

                normalized_group_output.emit(&view, &plan, 3, row_stats);

                assert_eq!(
                    probabilities[output_group_base + output_class_stride * 0],
                    (4.0_f64 - maximum).exp() / shifted_exponential_sum
                );
                assert_eq!(
                    probabilities[output_group_base + output_class_stride * 1],
                    (8.0_f64 - maximum).exp() / shifted_exponential_sum
                );
                assert_eq!(
                    probabilities[output_group_base + output_class_stride * 2],
                    (12.0_f64 - maximum).exp() / shifted_exponential_sum
                );

                assert_eq!(
                    log_probabilities[output_group_base + output_class_stride * 0],
                    (4.0_f64 - maximum) - log_shifted_exponential_sum
                );
                assert_eq!(
                    log_probabilities[output_group_base + output_class_stride * 1],
                    (8.0_f64 - maximum) - log_shifted_exponential_sum
                );
                assert_eq!(
                    log_probabilities[output_group_base + output_class_stride * 2],
                    (12.0_f64 - maximum) - log_shifted_exponential_sum
                );
            }
        }
    }

    mod fn_log_sum_exp {
        use super::*;

        #[rustfmt::skip::macros(vec)]
        fn output_data() -> Vec<f64> {
            // Every row has same shifted exp and log shifted exp sums because all values are
            // shifted by the same constant, thus, producing the same shifted value
            let log_shifted_exponential_sum = 0.018479302594657483;
            vec![
                9. + log_shifted_exponential_sum,
                10. + log_shifted_exponential_sum,
                11. + log_shifted_exponential_sum,
                12. + log_shifted_exponential_sum,
                21. + log_shifted_exponential_sum,
                22. + log_shifted_exponential_sum,
                23. + log_shifted_exponential_sum,
                24. + log_shifted_exponential_sum,
            ]
        }

        mod when_dimension_is_kept {
            use super::*;

            #[test]
            fn it_calculates_a_sum_of_max_and_log_shifted_exp_sum_per_row_stats() {
                let tensor = tensor();
                let view = tensor.view();
                let result = log_sum_exp(&view, 1, true).unwrap();

                assert_eq!(
                    result,
                    Tensor::from_vec(vec![2, 1, 4], output_data()).unwrap()
                );
            }
        }

        mod when_dimension_is_removed {
            use super::*;

            #[test]
            fn it_calculates_a_sum_of_max_and_log_shifted_exp_sum_per_row_stats() {
                let tensor = tensor();
                let view = tensor.view();
                let result = log_sum_exp(&view, 1, false).unwrap();

                assert_eq!(result, Tensor::from_vec(vec![2, 4], output_data()).unwrap());
            }
        }

        mod when_class_is_empty {
            use super::*;

            #[test]
            fn it_returns_negative_infinite_values() {
                let tensor = Tensor::from_vec(vec![2, 0, 4], vec![]).unwrap();
                let view = tensor.view();
                let result = log_sum_exp(&view, 1, false).unwrap();

                assert_eq!(
                    result,
                    Tensor::from_vec(vec![2, 4], vec![f64::NEG_INFINITY; 8]).unwrap()
                );
            }
        }
    }

    mod fn_normalized_forward {
        use super::*;

        #[rustfmt::skip::macros(vec)]
        #[test]
        fn it_calculates_probabilities_of_the_given_tensor_view() {
            let tensor = Tensor::from_vec(
                vec![3, 2, 3],
                vec![
                    //     ⌄ only this column is calculated because of slice() over the tensor view
                    1.0,  2.1,  3.0,
                    5.0,  3.1,  7.0,

                    8.0,  13.0, 14.0,
                    8.1,  13.4, 15.0,

                    9.0,  13.2, 14.0,
                    18.0, 13.5, 17.0,
                ],
            )
            .unwrap();
            let view = tensor.view().slice(2, 1..2).unwrap();
            let result = normalized_forward(&view, 1, true, true).unwrap();
            let probability_denominator = |v1: f64, v2: f64| {
                let max = v1.max(v2);
                (v1 - max).exp() + (v2 - max).exp()
            };
            let probability_nominator = |v1: f64, v2: f64| {
                let max = v1.max(v2);
                let min = v1.min(v2);
                min - max
            };
            let expected_probabilities_tensor = Tensor::from_vec(
                vec![3, 2, 1],
                vec![
                    probability_nominator(2.1, 3.1).exp() / probability_denominator(2.1, 3.1),
                    probability_nominator(3.1, 3.1).exp() / probability_denominator(2.1, 3.1),

                    probability_nominator(13.0, 13.4).exp() / probability_denominator(13.0, 13.4),
                    probability_nominator(13.4, 13.4).exp() / probability_denominator(13.0, 13.4),

                    probability_nominator(13.2, 13.5).exp() / probability_denominator(13.2, 13.5),
                    probability_nominator(13.5, 13.5).exp() / probability_denominator(13.2, 13.5),
                ],
            )
            .unwrap();

            let expected_log_probabilities_tensor = Tensor::from_vec(
                vec![3, 2, 1],
                vec![
                    probability_nominator(2.1, 3.1) - probability_denominator(2.1, 3.1).ln(),
                    probability_nominator(3.1, 3.1) - probability_denominator(2.1, 3.1).ln(),

                    probability_nominator(13.0, 13.4) - probability_denominator(13.0, 13.4).ln(),
                    probability_nominator(13.4, 13.4) - probability_denominator(13.0, 13.4).ln(),

                    probability_nominator(13.2, 13.5) - probability_denominator(13.2, 13.5).ln(),
                    probability_nominator(13.5, 13.5) - probability_denominator(13.2, 13.5).ln(),
                ],
            )
            .unwrap();

            assert_eq!(result.probabilities, Some(expected_probabilities_tensor));
            assert_eq!(
                result.log_probabilities,
                Some(expected_log_probabilities_tensor)
            );
        }
    }

    mod fn_indexed_mean_nll_forward {
        use super::*;

        #[rustfmt::skip::macros(vec)]
        #[test]
        fn it_calculates_mean_nll_and_probabilities_of_the_given_tensor_view() {
            let tensor = Tensor::from_vec(
                vec![3, 2, 3],
                vec![
                    //     ⌄ only this column is calculated because of slice() over the tensor view
                    1.0,  2.1,  3.0,
                    5.0,  3.1,  7.0,

                    8.0,  13.0, 14.0,
                    8.1,  13.4, 15.0,

                    9.0,  13.2, 14.0,
                    18.0, 13.5, 17.0,
                ],
            )
            .unwrap();
            let view = tensor.view().slice(2, 1..2).unwrap();
            let targets = [1, 0, 1];
            let result = indexed_mean_nll_forward(&view, 1, &targets, true).unwrap();
            let probability_denominator = |v1: f64, v2: f64| {
                let max = v1.max(v2);
                (v1 - max).exp() + (v2 - max).exp()
            };
            let probability_nominator = |v1: f64, v2: f64| {
                let max = v1.max(v2);
                let min = v1.min(v2);
                min - max
            };
            let expected_probabilities_tensor = Tensor::from_vec(
                vec![3, 2, 1],
                vec![
                    probability_nominator(2.1, 3.1).exp() / probability_denominator(2.1, 3.1),
                    probability_nominator(3.1, 3.1).exp() / probability_denominator(2.1, 3.1),

                    probability_nominator(13.0, 13.4).exp() / probability_denominator(13.0, 13.4),
                    probability_nominator(13.4, 13.4).exp() / probability_denominator(13.0, 13.4),

                    probability_nominator(13.2, 13.5).exp() / probability_denominator(13.2, 13.5),
                    probability_nominator(13.5, 13.5).exp() / probability_denominator(13.2, 13.5),
                ],
            )
            .unwrap();
            let row1_log_shifted_exponential_sum = 0.31326168751822286;
            let row2_log_shifted_exponential_sum = 0.5130152523999525;
            let row3_log_shifted_exponential_sum = 0.5543552444685268;
            let target_count = targets.len() as f64;

            let expected_row1_loss = (3.1 - 3.1) + row1_log_shifted_exponential_sum;
            let expected_row2_loss = (13.4 - 13.0) + row2_log_shifted_exponential_sum;
            let expected_row3_loss = (13.5 - 13.5) + row3_log_shifted_exponential_sum;

            assert_eq!(result.probabilities, Some(expected_probabilities_tensor));
            assert_eq!(
                result.loss,
                (expected_row1_loss + expected_row2_loss + expected_row3_loss) / target_count
            );
        }

        #[test]
        fn it_uses_scaled_mean_when_the_loss_sum_overflows() {
            let tensor = Tensor::from_vec(vec![2, 2], vec![f64::MAX, 0.0, f64::MAX, 0.0]).unwrap();
            let targets = [1, 1];

            let result = indexed_mean_nll_forward(&tensor.view(), 1, &targets, false).unwrap();

            assert_eq!(result.loss, f64::MAX);
            assert_eq!(result.probabilities, None);
        }
    }
}
