//! Precomputed rotary-position tables and differentiable feature-pair rotation.
//!
//! Rotary position embedding (RoPE) treats the final feature axis as adjacent pairs. For feature
//! width `d`, pair `j` uses the inverse frequency
//!
//! `ωⱼ = base^(-2j/d)`
//!
//! and absolute position `p` uses angle `θₚⱼ = pωⱼ`. Applying that angle rotates features
//! `(x₂ⱼ, x₂ⱼ₊₁)` while preserving the input shape.
//!
//! [`RotaryEmbedding`] owns three row-major tables:
//!
//! - inverse frequencies with shape `[d / 2]`;
//! - cosines with shape `[max_positions, d / 2]`;
//! - sines with shape `[max_positions, d / 2]`.
//!
//! Construction validation proceeds from cheap structural checks to numerical checks: validate
//! `d`, `max_positions`, and `base`; prove the table element counts are representable; validate
//! every inverse frequency; then ensure all requested position/pair angles produce finite table
//! values. Rotation performs a separate validation for an input shaped
//! `[..., tokens, features]` and selects table rows for the half-open absolute-position interval
//! `[position_offset, position_offset + tokens)`.

use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::tensor::storage::{Tensor, TensorError};
use crate::utils::canonical_zero;
use std::error::Error;
use std::fmt;

/// A rejected rotary configuration, input shape, or position interval.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeError {
    /// The final feature axis has no values and therefore no rotatable pairs.
    ZeroFeatureWidth,
    /// The final feature axis cannot be divided into adjacent two-value pairs.
    OddFeatureWidth {
        /// Rejected number of features.
        width: usize,
    },
    /// No absolute-position row was requested for the precomputed tables.
    ZeroPositionCapacity,
    /// The frequency base is non-positive, NaN, or infinite.
    InvalidBase {
        /// Rejected frequency base.
        base: f64,
    },
    /// Multiplying position rows by feature pairs overflowed `usize`.
    TableSizeOverflow {
        /// Number of position rows participating in the failed product.
        positions: usize,
        /// Number of adjacent feature pairs participating in the failed product.
        pairs: usize,
    },
    /// A table buffer is too large for a `Vec<f64>` or could not be reserved.
    TableAllocationFailed {
        /// Requested number of `f64` values.
        elements: usize,
    },
    /// An inverse frequency, angle, sine, or cosine was not finite.
    NonFiniteTableValue {
        /// First affected absolute-position row.
        position: usize,
        /// Affected adjacent-feature pair.
        pair: usize,
    },
    /// The input does not contain both a token axis and a feature axis.
    InputRank {
        /// Rejected input rank.
        rank: usize,
    },
    /// The input's final axis differs from the width used to build the tables.
    FeatureWidthMismatch {
        /// Feature width owned by the rotary embedding.
        expected: usize,
        /// Feature width read from the input's final axis.
        actual: usize,
    },
    /// Computing the exclusive end of the requested position interval overflowed `usize`.
    PositionOffsetOverflow {
        /// First requested absolute position.
        offset: usize,
        /// Size of the input's penultimate token axis.
        tokens: usize,
    },
    /// The requested position interval extends beyond the precomputed tables.
    PositionRangeExceeded {
        /// First requested absolute position.
        offset: usize,
        /// Size of the input's penultimate token axis.
        tokens: usize,
        /// Number of available absolute-position rows.
        max_positions: usize,
    },
    /// Tensor construction rejected a derived shape or value buffer.
    Tensor(TensorError),
    /// The differentiable pair-rotation primitive rejected the operation.
    Autodiff(TensorAutodiffError),
}

impl fmt::Display for RopeError {
    /// Formats the rejected configuration, shape, interval, or wrapped lower-level error.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFeatureWidth => formatter.write_str("rotary feature width must be nonzero"),
            Self::OddFeatureWidth { width } => {
                write!(formatter, "rotary feature width must be even, got {width}")
            }
            Self::ZeroPositionCapacity => {
                formatter.write_str("rotary position capacity must be nonzero")
            }
            Self::InvalidBase { base } => write!(
                formatter,
                "rotary frequency base must be finite and positive, got {base:?}"
            ),
            Self::TableSizeOverflow { positions, pairs } => write!(
                formatter,
                "rotary table with {positions} positions and {pairs} pairs does not fit usize"
            ),
            Self::TableAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate rotary table storage for {elements} f64 values"
            ),
            Self::NonFiniteTableValue { position, pair } => write!(
                formatter,
                "rotary table value at position {position}, pair {pair} is not finite"
            ),
            Self::InputRank { rank } => write!(
                formatter,
                "rotary input needs at least rank two [..., tokens, features], got rank {rank}"
            ),
            Self::FeatureWidthMismatch { expected, actual } => write!(
                formatter,
                "rotary input feature width must be {expected}, got {actual}"
            ),
            Self::PositionOffsetOverflow { offset, tokens } => write!(
                formatter,
                "rotary position interval overflows: offset {offset} plus {tokens} tokens"
            ),
            Self::PositionRangeExceeded {
                offset,
                tokens,
                max_positions,
            } => write!(
                formatter,
                "rotary position interval [{offset}, {}) exceeds capacity {max_positions}",
                offset.saturating_add(*tokens)
            ),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Autodiff(error) => error.fmt(formatter),
        }
    }
}

impl Error for RopeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::Autodiff(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for RopeError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorAutodiffError> for RopeError {
    fn from(error: TensorAutodiffError) -> Self {
        Self::Autodiff(error)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct RotaryTableLayout {
    /// Number of adjacent pairs in one position row: `feature_width / 2`.
    pairs: usize,
    /// Number of values in either the sine or cosine table: `max_positions * pairs`.
    table_elements: usize,
}

/// Rejects an element count that cannot be addressed safely by a `Vec<f64>`.
///
/// This check establishes the platform-dependent upper bound implied by `isize::MAX` byte
/// offsets. It does not allocate memory and therefore cannot guarantee that a later allocation will
/// succeed.
fn validate_value_capacity(elements: usize) -> Result<(), RopeError> {
    let maximum = (isize::MAX as usize) / size_of::<f64>();
    if elements > maximum {
        return Err(RopeError::TableAllocationFailed { elements });
    }
    Ok(())
}

/// Creates an empty `f64` buffer with capacity for exactly `elements` table values.
///
/// The requested count first passes [`validate_value_capacity`], then `Vec::try_reserve_exact`
/// performs the real allocation attempt. The returned vector has length zero; callers append
/// values in the table's row-major traversal order.
fn reserved_values(elements: usize) -> Result<Vec<f64>, RopeError> {
    validate_value_capacity(elements)?;

    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RopeError::TableAllocationFailed { elements })?;

    Ok(values)
}

/// Computes the inverse angular frequency for one adjacent-feature pair.
///
/// For feature width `d` and pair index `j`, the result is `base^(-2j/d)`. Callers must validate
/// that `d` is nonzero and even, `base` is finite and positive, and `j < d / 2`. This low-level
/// calculation deliberately leaves finiteness checking to [`visit_inverse_frequencies`].
fn inverse_frequency(feature_width: usize, base: f64, pair: usize) -> f64 {
    let exponent = -2.0 * (pair as f64) / (feature_width as f64);
    base.powf(exponent)
}

/// Visits every inverse frequency in increasing pair order.
///
/// The callback receives `(pair, frequency)` for `pair = 0..feature_width / 2`. The caller is
/// expected to have validated the structural table specification first.
fn visit_inverse_frequencies(
    feature_width: usize,
    base: f64,
    mut visit: impl FnMut(usize, f64),
) -> Result<(), RopeError> {
    let pairs = feature_width / 2;
    for pair in 0..pairs {
        let frequency = inverse_frequency(feature_width, base, pair);
        if !frequency.is_finite() {
            return Err(RopeError::NonFiniteTableValue { position: 0, pair });
        }
        visit(pair, frequency);
    }

    Ok(())
}

/// Computes the sine and cosine stored for one absolute-position/pair coordinate.
///
/// The angle is `position * frequency`. A finite result is returned as `(sine, cosine)` with signed
/// zero canonicalized; `None` reports that the angle or either trigonometric result is not finite.
fn table_value(position: usize, frequency: f64) -> Option<(f64, f64)> {
    let angle = (position as f64) * frequency;
    let (sine, cosine) = angle.sin_cos();
    (angle.is_finite() && sine.is_finite() && cosine.is_finite())
        .then(|| (canonical_zero(sine), canonical_zero(cosine)))
}

/// Visits the complete sine/cosine table in row-major order.
///
/// `frequency(pair)` supplies the already validated inverse frequency for a feature pair. The
/// visitor receives `(position, pair, sine, cosine)` for positions first and pairs second, matching
/// tensors shaped `[max_positions, pairs]`.
fn visit_table_values(
    max_positions: usize,
    pairs: usize,
    mut frequency: impl FnMut(usize) -> f64,
    mut visit: impl FnMut(usize, usize, f64, f64),
) -> Result<(), RopeError> {
    for position in 0..max_positions {
        for pair in 0..pairs {
            let (sine, cosine) = table_value(position, frequency(pair))
                .ok_or(RopeError::NonFiniteTableValue { position, pair })?;
            visit(position, pair, sine, cosine)
        }
    }
    Ok(())
}

/// Validates the scalar table specification and derives its two storage dimensions.
///
/// This is the first validation layer used by both [`RotaryEmbedding::new`] and
/// [`RotaryEmbedding::validate_table_specification`]. It maps `feature_width` to
/// `pairs = feature_width / 2` and calculates the number of values required by each position table
/// as `max_positions * pairs`. It does not allocate storage or evaluate any frequencies.
fn validate_table_layout(
    feature_width: usize,
    max_positions: usize,
    base: f64,
) -> Result<RotaryTableLayout, RopeError> {
    if feature_width == 0 {
        return Err(RopeError::ZeroFeatureWidth);
    }
    if !feature_width.is_multiple_of(2) {
        return Err(RopeError::OddFeatureWidth {
            width: feature_width,
        });
    }
    if max_positions == 0 {
        return Err(RopeError::ZeroPositionCapacity);
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(RopeError::InvalidBase { base });
    }

    let pairs = feature_width / 2;
    let table_elements = max_positions
        .checked_mul(pairs)
        .ok_or(RopeError::TableSizeOverflow {
            positions: max_positions,
            pairs,
        })?;

    Ok(RotaryTableLayout {
        pairs,
        table_elements,
    })
}

/// Validates that every requested table coordinate would produce finite trigonometric values.
///
/// This validation-only path avoids constructing the full `[max_positions, pairs]` tables. For
/// each pair it first probes the final requested position. If that value is invalid, it binary
/// searches the monotonic transition from finite to non-finite angles and records the earliest
/// invalid `(position, pair)`. The final error is the first coordinate in row-major order across
/// all pairs.
///
/// `max_positions` must be nonzero, `pairs` must come from a validated feature width, and
/// `frequency(pair)` must return a finite inverse frequency. These preconditions are established by
/// [`validate_table_layout`] and [`visit_inverse_frequencies`] in the public validation pipeline.
fn validate_table_values(
    max_positions: usize,
    pairs: usize,
    mut frequency: impl FnMut(usize) -> f64,
) -> Result<(), RopeError> {
    let last_position = max_positions - 1;
    let mut first_nonfinite = None;
    for pair in 0..pairs {
        let frequency = frequency(pair);
        debug_assert!(frequency.is_finite());

        if table_value(last_position, frequency).is_some() {
            continue;
        }

        let mut last_finite = 0;
        let mut first_invalid = last_position;
        debug_assert!(table_value(last_finite, frequency).is_some());
        while last_finite + 1 < first_invalid {
            let middle = last_finite + (first_invalid - last_finite) / 2;
            if table_value(middle, frequency).is_some() {
                last_finite = middle;
            } else {
                first_invalid = middle;
            }
        }
        let candidate = (first_invalid, pair);
        first_nonfinite = Some(match first_nonfinite {
            Some(current) => std::cmp::min(current, candidate),
            None => candidate,
        })
    }
    match first_nonfinite {
        Some((position, pair)) => Err(RopeError::NonFiniteTableValue { position, pair }),
        None => Ok(()),
    }
}

/// Copies a contiguous table interval into an owned rank-two tensor.
///
/// Rotation selects a flat interval from a precomputed `[max_positions, pairs]` table. This helper
/// gives that interval independent storage and the logical shape `[tokens, pairs]` required by the
/// pair-rotation primitive. `values.len()` must equal `shape[0] * shape[1]`.
///
/// # Errors
///
/// Returns [`RopeError::TableAllocationFailed`] if the copy cannot be reserved, or
/// [`RopeError::Tensor`] if `values` does not match the requested shape.
fn copy_table_slice(values: &[f64], shape: &[usize; 2]) -> Result<Tensor, RopeError> {
    let mut copied = reserved_values(values.len())?;
    copied.extend_from_slice(values);
    Tensor::from_vec(shape.to_vec(), copied).map_err(Into::into)
}

/// Owned inverse-frequency and sine/cosine tables for one feature width.
///
/// The tables are immutable after construction and can serve any input shaped
/// `[..., tokens, feature_width]` whose requested absolute-position interval fits within
/// `max_positions`.
#[derive(Clone, Debug, PartialEq)]
pub struct RotaryEmbedding {
    /// Required extent of an input's final feature axis.
    feature_width: usize,
    /// Number of precomputed absolute-position rows.
    max_positions: usize,
    /// Positive finite base of the geometric inverse-frequency schedule.
    base: f64,
    /// One inverse angular frequency per adjacent feature pair, shape `[feature_width / 2]`.
    inverse_frequencies: Tensor,
    /// Cosines indexed by `[absolute_position, pair]`.
    cosines: Tensor,
    /// Sines indexed by `[absolute_position, pair]`.
    sines: Tensor,
}

impl RotaryEmbedding {
    /// Validates a RoPE specification and precomputes all frequency, sine, and cosine values.
    ///
    /// `feature_width` is the size of the input's final axis, `max_positions` is the exclusive
    /// upper bound for absolute positions, and `base` controls the geometric frequency schedule.
    /// A width `d` creates `d / 2` adjacent-feature pairs. The resulting tensors have shapes:
    ///
    /// - inverse frequencies: `[d / 2]`;
    /// - cosines: `[max_positions, d / 2]`;
    /// - sines: `[max_positions, d / 2]`.
    ///
    /// Position rows are generated in increasing absolute-position order; within a row, pair
    /// values are generated in increasing pair order.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError`] if the scalar specification is invalid, a derived size overflows or
    /// cannot be allocated, a frequency/table value is non-finite, or an owned tensor rejects its
    /// derived shape and buffer.
    pub fn new(feature_width: usize, max_positions: usize, base: f64) -> Result<Self, RopeError> {
        let layout = validate_table_layout(feature_width, max_positions, base)?;
        let mut frequencies = reserved_values(layout.pairs)?;
        visit_inverse_frequencies(feature_width, base, |_, frequency| {
            frequencies.push(frequency);
        })?;

        let mut cosines = reserved_values(layout.table_elements)?;
        let mut sines = reserved_values(layout.table_elements)?;
        visit_table_values(
            max_positions,
            layout.pairs,
            |pair| frequencies[pair],
            |_, _, sine, cosine| {
                cosines.push(cosine);
                sines.push(sine)
            },
        )?;

        Ok(Self {
            feature_width,
            max_positions,
            base,
            inverse_frequencies: Tensor::from_vec(vec![layout.pairs], frequencies)?,
            cosines: Tensor::from_vec(vec![max_positions, layout.pairs], cosines)?,
            sines: Tensor::from_vec(vec![max_positions, layout.pairs], sines)?,
        })
    }

    /// Validates a table specification without allocating or retaining the tables.
    ///
    /// The checks mirror construction in dependency order:
    ///
    /// 1. validate the width, position capacity, base, and `max_positions * pairs` product;
    /// 2. verify that the frequency and position-table buffer sizes are addressable;
    /// 3. compute every inverse frequency and reject non-finite results;
    /// 4. verify that every requested position/pair angle would yield finite sine and cosine values.
    ///
    /// Table-value validation uses endpoint probes and binary search rather than materializing every
    /// value. Passing this method proves that the specification and numerical range are valid, but
    /// it cannot guarantee that a later [`Self::new`] call will obtain memory from the allocator.
    pub fn validate_table_specification(
        feature_width: usize,
        max_positions: usize,
        base: f64,
    ) -> Result<(), RopeError> {
        let layout = validate_table_layout(feature_width, max_positions, base)?;
        validate_value_capacity(layout.pairs)?;
        visit_inverse_frequencies(feature_width, base, |_, _| {})?;
        validate_value_capacity(layout.table_elements)?;
        validate_table_values(max_positions, layout.pairs, |pair| {
            inverse_frequency(feature_width, base, pair)
        })
    }

    /// Rotates the final feature axis at consecutive absolute positions while recording autograd.
    ///
    /// The penultimate axis is the token axis. Any earlier axes are independent lanes, so rank-two,
    /// batched, and batched-head layouts share one rule. For an input shaped
    /// `[..., tokens, feature_width]`, token `t` uses the table row at
    /// `position_offset + t`; the output has the same shape as the input.
    ///
    /// This is equivalent to calling [`Self::rotate_with_context`] with
    /// [`AutogradContext::recording`].
    pub fn rotate(
        &self,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<TensorValue, RopeError> {
        self.rotate_with_context(AutogradContext::recording(), input, position_offset)
    }

    /// Rotates adjacent feature pairs under the caller's explicit graph-recording policy.
    ///
    /// The accepted input layout is `[..., tokens, feature_width]`. Axes before `tokens` are
    /// independent lanes that reuse the same position rows. For each token `t` and pair `j`, this
    /// method selects angle `(position_offset + t) * inverse_frequencies[j]` and applies
    ///
    /// ```text
    /// left′  = left * cos(angle) - right * sin(angle)
    /// right′ = left * sin(angle) + right * cos(angle)
    /// ```
    ///
    /// Validation flows from the input toward the table slice: require token and feature axes;
    /// match the final axis to the configured width; form the half-open interval
    /// `[position_offset, position_offset + tokens)` without overflow; require that interval to fit
    /// `max_positions`; then convert its row range into flat table offsets. The selected cosine and
    /// sine rows are copied into tensors shaped `[tokens, feature_width / 2]` before delegation to
    /// `TensorValue::rotary_pairs_with_context`.
    ///
    /// `context` controls whether the returned value retains the input edge for backward traversal.
    /// In either mode the same shape and numerical validations are performed. The output shape is
    /// identical to `input.shape()`.
    pub fn rotate_with_context(
        &self,
        context: AutogradContext,
        input: &TensorValue,
        position_offset: usize,
    ) -> Result<TensorValue, RopeError> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(RopeError::InputRank { rank: shape.len() });
        }
        let actual_width = shape[shape.len() - 1];
        if actual_width != self.feature_width {
            return Err(RopeError::FeatureWidthMismatch {
                expected: self.feature_width,
                actual: actual_width,
            });
        }
        let tokens = shape[shape.len() - 2];
        let position_end =
            position_offset
                .checked_add(tokens)
                .ok_or(RopeError::PositionOffsetOverflow {
                    offset: position_offset,
                    tokens,
                })?;
        if position_end > self.max_positions {
            return Err(RopeError::PositionRangeExceeded {
                offset: position_offset,
                tokens,
                max_positions: self.max_positions,
            });
        }

        let pairs = self.feature_width / 2;
        let start = position_offset
            .checked_mul(pairs)
            .ok_or(RopeError::TableSizeOverflow {
                positions: position_offset,
                pairs,
            })?;
        let end = position_end
            .checked_mul(pairs)
            .ok_or(RopeError::TableSizeOverflow {
                positions: position_end,
                pairs,
            })?;
        let table_shape = [tokens, pairs];
        let cosines = copy_table_slice(&self.cosines.as_slice()[start..end], &table_shape)?;
        let sines = copy_table_slice(&self.sines.as_slice()[start..end], &table_shape)?;
        input
            .rotary_pairs_with_context(context, &cosines, &sines)
            .map_err(Into::into)
    }

    /// Returns the required size of the final feature axis.
    ///
    /// The value is nonzero and even because it passed construction validation.
    pub fn feature_width(&self) -> usize {
        self.feature_width
    }

    /// Returns the exclusive upper bound for absolute positions.
    ///
    /// Valid rotation intervals must end at or before this value.
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Returns the finite, strictly positive frequency-schedule base.
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Returns the inverse-frequency tensor with shape `[feature_width / 2]`.
    ///
    /// Element `j` is `base^(-2j/feature_width)` and supplies the angular increment per absolute
    /// position for adjacent-feature pair `j`.
    pub fn inverse_frequencies(&self) -> &Tensor {
        &self.inverse_frequencies
    }

    /// Returns the cosine table with shape `[max_positions, feature_width / 2]`.
    ///
    /// Coordinate `[p, j]` stores `cos(p * inverse_frequencies[j])`.
    pub fn cosines(&self) -> &Tensor {
        &self.cosines
    }

    /// Returns the sine table with shape `[max_positions, feature_width / 2]`.
    ///
    /// Coordinate `[p, j]` stores `sin(p * inverse_frequencies[j])`.
    pub fn sines(&self) -> &Tensor {
        &self.sines
    }
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::autograd::tensor_core::{GraphRetention, TensorOperation};

    fn tensor(shape: &[usize], values: &[f64]) -> Tensor {
        Tensor::from_vec(shape.to_vec(), values.to_vec()).unwrap()
    }

    fn parameter(shape: &[usize], values: &[f64]) -> TensorValue {
        TensorValue::parameter(tensor(shape, values)).unwrap()
    }

    fn constant(shape: &[usize], values: &[f64]) -> TensorValue {
        TensorValue::constant(tensor(shape, values)).unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn sum_to_scalar(mut value: TensorValue) -> TensorValue {
        for axis in (0..value.shape().len()).rev() {
            value = value.sum_axis(axis, false).unwrap();
        }
        value
    }

    #[test]
    fn tables_and_known_adjacent_pair_rotations_are_frozen() {
        let rope = RotaryEmbedding::new(4, 6, 100.0).unwrap();
        assert_eq!(rope.feature_width(), 4);
        assert_eq!(rope.max_positions(), 6);
        assert_eq!(rope.base(), 100.0);
        assert_eq!(rope.inverse_frequencies().shape(), [2]);
        assert_close(rope.inverse_frequencies().as_slice(), &[1.0, 0.1], 1e-15);
        assert_eq!(rope.cosines().shape(), [6, 2]);
        assert_eq!(rope.sines().shape(), [6, 2]);

        let repeated = constant(
            &[3, 4],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        );
        let rotated = rope.rotate(&repeated, 0).unwrap();
        assert_eq!(rotated.operation(), TensorOperation::RotaryPairs);
        assert_close(
            rotated.value().as_slice(),
            &[
                1.0,
                0.0,
                1.0,
                0.0,
                1.0_f64.cos(),
                1.0_f64.sin(),
                0.1_f64.cos(),
                0.1_f64.sin(),
                2.0_f64.cos(),
                2.0_f64.sin(),
                0.2_f64.cos(),
                0.2_f64.sin(),
            ],
            1e-15,
        );
    }

    #[test]
    fn relative_dots_and_norms_survive_a_common_position_shift() {
        let rope = RotaryEmbedding::new(4, 6, 100.0).unwrap();
        let repeated_values = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let input = constant(&[3, 4], &repeated_values);
        let original = rope.rotate(&input, 0).unwrap().value_snapshot();
        let shifted = rope.rotate(&input, 3).unwrap().value_snapshot();

        for values in [original.as_slice(), shifted.as_slice()] {
            for row in values.chunks_exact(4) {
                assert!((row.iter().map(|value| value * value).sum::<f64>() - 2.0).abs() < 1e-12);
            }
        }
        for query in 0..3 {
            for key in 0..3 {
                let dot = |values: &[f64]| {
                    values[query * 4..query * 4 + 4]
                        .iter()
                        .zip(&values[key * 4..key * 4 + 4])
                        .map(|(left, right)| left * right)
                        .sum::<f64>()
                };
                assert!((dot(original.as_slice()) - dot(shifted.as_slice())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn transpose_rotation_is_the_exact_local_vjp() {
        let rope = RotaryEmbedding::new(4, 4, 100.0).unwrap();
        let input_values = [0.7, -1.2, 2.0, 0.5];
        let upstream_values = [0.2, -0.7, 1.1, 0.3];
        let input = parameter(&[1, 4], &input_values);
        let output = rope.rotate(&input, 2).unwrap();
        output
            .backward_with_seed(
                &tensor(&[1, 4], &upstream_values).view(),
                GraphRetention::Retain,
            )
            .unwrap();
        let gradient = input.gradient().unwrap();

        let mut expected = [0.0; 4];
        for pair in 0..2 {
            let angle = 2.0 * rope.inverse_frequencies().as_slice()[pair];
            let (sine, cosine) = angle.sin_cos();
            let left = upstream_values[pair * 2];
            let right = upstream_values[pair * 2 + 1];
            expected[pair * 2] = left * cosine + right * sine;
            expected[pair * 2 + 1] = -left * sine + right * cosine;
        }
        assert_close(gradient.as_slice(), &expected, 1e-15);

        let step = 1e-6;
        for coordinate in 0..input_values.len() {
            let mut plus = input_values;
            plus[coordinate] += step;
            let mut minus = input_values;
            minus[coordinate] -= step;
            let loss = |probe: &[f64]| {
                rope.rotate(&constant(&[1, 4], probe), 2)
                    .unwrap()
                    .value()
                    .as_slice()
                    .iter()
                    .zip(upstream_values)
                    .map(|(value, upstream)| value * upstream)
                    .sum::<f64>()
            };
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * step);
            assert!((numerical - gradient.as_slice()[coordinate]).abs() < 2e-9);
        }
    }

    #[test]
    fn rank_two_three_four_and_empty_axes_keep_their_shapes() {
        let rope = RotaryEmbedding::new(4, 4, 100.0).unwrap();
        for shape in [&[2, 4][..], &[2, 2, 4], &[2, 3, 2, 4]] {
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|value| value as f64 / 10.0)
                .collect::<Vec<_>>();
            assert_eq!(
                rope.rotate(&constant(shape, &values), 1).unwrap().shape(),
                shape
            );
        }

        let empty_leading = constant(&[0, 2, 4], &[]);
        assert_eq!(rope.rotate(&empty_leading, 1).unwrap().shape(), [0, 2, 4]);
        let empty_tokens = constant(&[2, 0, 4], &[]);
        assert_eq!(rope.rotate(&empty_tokens, 4).unwrap().shape(), [2, 0, 4]);
    }

    #[test]
    fn configuration_and_shape_errors_have_stable_precedence() {
        assert_eq!(
            RotaryEmbedding::new(0, 0, f64::NAN).unwrap_err(),
            RopeError::ZeroFeatureWidth
        );
        assert_eq!(
            RotaryEmbedding::new(3, 0, f64::NAN).unwrap_err(),
            RopeError::OddFeatureWidth { width: 3 }
        );
        assert_eq!(
            RotaryEmbedding::new(4, 0, f64::NAN).unwrap_err(),
            RopeError::ZeroPositionCapacity
        );
        assert_eq!(
            RotaryEmbedding::new(4, 2, 0.0).unwrap_err(),
            RopeError::InvalidBase { base: 0.0 }
        );

        let rope = RotaryEmbedding::new(4, 3, 100.0).unwrap();
        assert_eq!(
            rope.rotate(&constant(&[4], &[0.0; 4]), usize::MAX)
                .unwrap_err(),
            RopeError::InputRank { rank: 1 }
        );
        assert_eq!(
            rope.rotate(&constant(&[1, 2], &[0.0; 2]), usize::MAX)
                .unwrap_err(),
            RopeError::FeatureWidthMismatch {
                expected: 4,
                actual: 2
            }
        );
        assert_eq!(
            rope.rotate(&constant(&[1, 4], &[0.0; 4]), usize::MAX)
                .unwrap_err(),
            RopeError::PositionOffsetOverflow {
                offset: usize::MAX,
                tokens: 1
            }
        );
        assert_eq!(
            rope.rotate(&constant(&[2, 4], &[0.0; 8]), 2).unwrap_err(),
            RopeError::PositionRangeExceeded {
                offset: 2,
                tokens: 2,
                max_positions: 3
            }
        );
    }

    #[test]
    fn table_storage_free_specification_matches_constructor_validation() {
        RotaryEmbedding::validate_table_specification(4, 6, 100.0).unwrap();

        for (feature_width, max_positions, base) in [
            (0, 0, f64::NAN),
            (3, 0, f64::NAN),
            (4, 0, f64::NAN),
            (4, 2, 0.0),
            (4, usize::MAX, 100.0),
            (1024, 1, f64::from_bits(1)),
        ] {
            assert_eq!(
                RotaryEmbedding::validate_table_specification(feature_width, max_positions, base,)
                    .unwrap_err(),
                RotaryEmbedding::new(feature_width, max_positions, base).unwrap_err()
            );
        }
    }

    #[test]
    fn table_specification_rejects_unrepresentable_vec_capacities_before_traversal() {
        let elements = (isize::MAX as usize) / std::mem::size_of::<f64>() + 1;
        let expected = RopeError::TableAllocationFailed { elements };
        assert_eq!(
            RotaryEmbedding::validate_table_specification(2, elements, 100.0),
            Err(expected.clone())
        );
        assert_eq!(
            RotaryEmbedding::new(2, elements, 100.0),
            Err(expected.clone())
        );

        let feature_width = elements * 2;
        assert_eq!(
            RotaryEmbedding::validate_table_specification(feature_width, 1, 100.0),
            Err(expected.clone())
        );
        assert_eq!(RotaryEmbedding::new(feature_width, 1, 100.0), Err(expected));
    }

    #[test]
    fn inverse_frequency_error_precedes_later_table_capacity_failure() {
        let maximum = (isize::MAX as usize) / std::mem::size_of::<f64>();
        let max_positions = maximum / (1024 / 2) + 1;
        let constructor_error =
            RotaryEmbedding::new(1024, max_positions, f64::from_bits(1)).unwrap_err();
        assert!(matches!(
            constructor_error,
            RopeError::NonFiniteTableValue { position: 0, .. }
        ));
        assert_eq!(
            RotaryEmbedding::validate_table_specification(1024, max_positions, f64::from_bits(1),),
            Err(constructor_error)
        );
    }

    #[test]
    fn positive_position_table_error_matches_row_major_precedence() {
        let expected = RopeError::NonFiniteTableValue {
            position: 72,
            pair: 511,
        };
        assert_eq!(
            RotaryEmbedding::validate_table_specification(1024, 300, 1e-307),
            Err(expected.clone())
        );
        assert_eq!(RotaryEmbedding::new(1024, 300, 1e-307), Err(expected));
    }

    #[test]
    fn representable_huge_table_specification_uses_bounded_validation_work() {
        RotaryEmbedding::validate_table_specification(2, 1_000_000_000_000, 100.0).unwrap();
    }

    #[test]
    fn released_operands_fail_only_after_public_shape_checks() {
        let rope = RotaryEmbedding::new(4, 2, 100.0).unwrap();
        let parameter = parameter(&[1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let released = parameter
            .add(&constant(&[], &[0.0]))
            .expect("broadcast scalar is valid");
        sum_to_scalar(released.clone())
            .backward_with_seed(&tensor(&[], &[1.0]).view(), GraphRetention::Release)
            .unwrap();
        assert_eq!(
            rope.rotate(&released, 0).unwrap_err(),
            RopeError::Autodiff(TensorAutodiffError::ReleasedOperand {
                operation: TensorOperation::RotaryPairs,
                operand: 0,
            })
        );
    }
}
