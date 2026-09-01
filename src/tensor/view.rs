//! Borrowed tensor views with checked axis transforms and explicit materialization.

use super::storage::{
    DEFAULT_STRIDE, Tensor, TensorError, checked_offset, checked_row_major_layout,
};
use log::log;
use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

/// A rejected tensor-view transform, slice, or coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorViewError {
    /// The shared tensor layout or coordinate rules rejected an operation.
    Tensor(TensorError),
    /// An operation names an axis that the view does not have.
    AxisOutOfBounds { axis: usize, rank: usize },
    /// A permutation must name exactly one source axis per output axis.
    PermutationLengthMismatch { expected: usize, actual: usize },
    /// A permutation names one source axis more than once.
    DuplicateAxis { axis: usize },
    /// A half-open slice has its start after its end.
    SliceStartAfterEnd {
        axis: usize,
        start: usize,
        end: usize,
    },
    /// A half-open slice ends beyond its source-axis extent.
    SliceEndOutOfBounds {
        axis: usize,
        end: usize,
        dimension: usize,
    },
    /// A reshape requests a different number of logical elements.
    ReshapeElementCountMismatch { current: usize, requested: usize },
    /// This chapter only reshapes views whose logical order is row-major contiguous.
    NonContiguousReshape,
}

impl fmt::Display for TensorViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(formatter),
            Self::AxisOutOfBounds { axis, rank } => {
                write!(formatter, "axis {axis} is out of bounds for rank {rank}")
            }
            Self::PermutationLengthMismatch { expected, actual } => write!(
                formatter,
                "permutation length {actual} does not match tensor rank {expected}"
            ),
            Self::DuplicateAxis { axis } => {
                write!(formatter, "permutation axis {axis} appears more than once")
            }
            Self::SliceStartAfterEnd { axis, start, end } => write!(
                formatter,
                "slice start {start} is after end {end} on axis {axis}"
            ),
            Self::SliceEndOutOfBounds {
                axis,
                end,
                dimension,
            } => write!(
                formatter,
                "slice end {end} is out of bounds for axis {axis} with size {dimension}"
            ),
            Self::ReshapeElementCountMismatch { current, requested } => write!(
                formatter,
                "cannot reshape {current} elements into {requested} elements"
            ),
            Self::NonContiguousReshape => formatter.write_str(
                "cannot reshape a non-row-major-contiguous view without materializing it",
            ),
        }
    }
}

impl Error for TensorViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for TensorViewError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

/// Logical row-major traversal over a layout whose metadata was already checked.
///
/// The cursor owns one `O(rank)` asix-state vector and updates it in place. It keeps no tensor
/// borrow, so in later kernel versions, the same mechanism can be used to read from the source or
/// write to the target storage. It remains create-private so arbitrary external coordinates still
/// enter thorugh [`TensorView::storage_offset`] or [`TensorView::get`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct OffsetAxis {
    extent: usize,
    stride: usize,
    // Position of a cursor inside axis
    position: usize,
    // If stride is a step which we use to move forward by axis, rewind is a range between first and
    // last elements on axis. Example:
    // Shape: [3, 2], strides: [2, 1], data: [[10, 20], [30, 40], [50, 60]]. Rewind for external
    // axis is 4:
    // [10,20] -> [30,40] -> [50,60]
    //    0          2          4
    // |------------------------|
    //            rewind = 4
    // For internal - 1:
    // 10   ->   20
    //  0         1
    //  |---------|
    //   rewind = 1
    // The main purpose of rewind is to reset the current offset of 1D buffer before adding a stripe
    // of outer axis to move to the next element. Example:
    // Flat buffer is [10, 20, 30, 40, 50, 60].
    // Offset: 0 -> 10
    // + internal stride (1)
    // Offset: 1 -> 20
    // We hit the end, and we want to advance to the next element of outer axis. If we simply add an
    // outer stride at this point, we end up having Offset == 3, thus skipping 30 value. Which is
    // why we need to rewind the offset first and then add outer stride:
    // Offset = Offset - rewind + outer stride => 2 -> 30
    rewind: usize,
}

#[derive(Debug, PartialEq)]
pub struct StrideOffsets {
    axes: Vec<OffsetAxis>,
    next_offset: usize,
    remaining: usize,
}

impl StrideOffsets {
    /// Checks one internal traversal plan and then owns its reusable axis state.
    pub fn checked(
        shape: &[usize],
        strides: &[usize],
        base_offset: usize,
        logical_len: usize, // Iteration length
        backing_len: usize, // Actual data length
    ) -> Option<Self> {
        if shape.len() != strides.len() {
            return None;
        }

        let expected_len = if shape.contains(&0) {
            0
        } else {
            // Safely multiply all shape sizes
            shape
                .iter()
                .try_fold(DEFAULT_STRIDE, |count, &extent| count.checked_mul(extent))?
        };
        if logical_len != expected_len {
            return None;
        }

        if logical_len == 0 {
            return Some(Self {
                axes: shape
                    .iter()
                    .zip(strides)
                    .map(|(&extent, &stride)| OffsetAxis {
                        extent,
                        stride,
                        position: 0,
                        rewind: 0,
                    })
                    .collect(),
                next_offset: base_offset,
                remaining: 0,
            });
        }

        let mut maximum_offset = base_offset;
        let mut axes = Vec::with_capacity(shape.len());
        for (&extent, &stride) in shape.iter().zip(strides) {
            let rewind = (extent - DEFAULT_STRIDE).checked_mul(stride)?;
            maximum_offset = maximum_offset.checked_add(rewind)?;
            axes.push(OffsetAxis {
                extent,
                stride,
                position: 0,
                rewind,
            });
        }

        if maximum_offset >= backing_len {
            return None;
        }

        Some(Self {
            axes,
            next_offset: base_offset,
            remaining: logical_len,
        })
    }

    fn advance(&mut self) {
        for axis in self.axes.iter_mut().rev() {
            let next_position = axis.position + DEFAULT_STRIDE;
            if next_position < axis.extent {
                axis.position = next_position;
                self.next_offset = self
                    .next_offset
                    .checked_add(axis.stride)
                    .expect("a checked traversal cannot overflow while advancing");
                return;
            }

            axis.position = 0;
            self.next_offset = self
                .next_offset
                .checked_sub(axis.rewind)
                .expect("a checked traversal cannot underflow while carrying");
        }

        unreachable!("a checked nonempty traversal advances within its logical shape");
    }
}

impl Iterator for StrideOffsets {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let offset = self.next_offset;
        self.remaining -= DEFAULT_STRIDE;
        if self.remaining != 0 {
            self.advance();
        }
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StrideOffsets {}
impl FusedIterator for StrideOffsets {}

/// An immutable n-dimensional interpretation of storage owned by a [`Tensor`].
///
/// The view copies only shape and stride metadata. Rust keeps the source tensor borrowed for the
/// view's lifetime, so safe code cannot mutate the owner while a sbsequently used view still
/// exists.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorView<'a> {
    source: &'a Tensor,
    shape: Vec<usize>,
    strides: Vec<usize>,
    base_offset: usize,
    len: usize,
}

impl Tensor {
    /// Borrows this tensor as a row-major-contiguous view without copying values.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            source: self,
            shape: self.shape().to_vec(),
            strides: self.strides().to_vec(),
            base_offset: 0,
            len: self.len(),
        }
    }
}

impl<'a> TensorView<'a> {
    /// Returns the number of logical axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the extent of every logical axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the source-storage movement for each logical axis.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the source-storage offset of the view's logical origin.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Returns the number of logical values in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reports whether the view has no logical values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reports whether logical row-major iteration visits one dense storage span. Or, in other
    /// words the method describes whether the shape and the stripes result in contiguous data
    /// sequence.
    ///
    /// Singleton axes may carry any stride because they never advance. Scalars and empty views are
    /// contiguous by this implementation's convention.
    pub fn is_contiguous(&self) -> bool {
        if self.is_empty() {
            return true;
        }

        let mut expected_stride = DEFAULT_STRIDE;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension > 1 && stride != expected_stride {
                return false;
            }
            expected_stride = expected_stride.checked_mul(dimension).expect(&format!(
                "It seems {:?} strides and {:?} shape are not consistent/ckecked.",
                self.strides, self.shape
            ));
        }
        true
    }

    /// Maps one checked logical coordinate to the owner's flat storage offset.
    pub fn storage_offset(&self, coordinate: &[usize]) -> Result<usize, TensorViewError> {
        checked_offset(&self.shape, &self.strides, self.base_offset, coordinate).map_err(Into::into)
    }

    /// Borrows the source value selected by one checked logical coordinate.
    pub fn get(&self, coordinate: &[usize]) -> Result<&'a f64, TensorViewError> {
        let offset = self.storage_offset(coordinate)?;
        let source: &'a [f64] = self.source.as_slice();

        Ok(&source[offset])
    }

    /// Traverses this already validated view in logical row-major order.
    pub fn logical_offsets(&self) -> StrideOffsets {
        self.projected_offsets(&self.shape, &self.strides, self.len)
            .expect("a TensorView retains checked traversal metadata")
    }

    /// Checks effective strides for another logical traversal of this source.
    pub fn projected_offsets(
        &self,
        iteration_shape: &[usize],
        effective_strides: &[usize],
        logical_len: usize,
    ) -> Option<StrideOffsets> {
        StrideOffsets::checked(
            iteration_shape,
            effective_strides,
            self.base_offset,
            logical_len,
            self.source.len(),
        )
    }

    /// Copis one source scalar selected by an offset from a checked plan.
    pub fn value_at_storage_offset(&self, offset: usize) -> f64 {
        self.source.as_slice()[offset]
    }

    /// Reinterprets a raw-major-contiguous view with a compatible shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorViewError> {
        let (strides, requested) = checked_row_major_layout(shape)?;
        if requested != self.len {
            return Err(TensorViewError::ReshapeElementCountMismatch {
                current: self.len,
                requested,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorViewError::NonContiguousReshape);
        }
        Ok(Self {
            source: self.source,
            shape: shape.to_vec(),
            strides,
            base_offset: self.base_offset,
            len: self.len,
        })
    }

    /// Swaps two logical axes without moving source values.
    pub fn transpose(&self, first: usize, second: usize) -> Result<Self, TensorViewError> {
        self.chek_axis(first)?;
        self.chek_axis(second)?;

        let mut axes = (0..self.rank()).collect::<Vec<_>>();
        axes.swap(first, second);
        self.permute(&axes)
    }

    /// Reorders axes so output axis `k` uses source axis `axes[k]`
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorViewError> {
        if axes.len() != self.rank() {
            return Err(TensorViewError::PermutationLengthMismatch {
                expected: self.rank(),
                actual: axes.len(),
            });
        }

        let mut seen = vec![false; self.rank()];
        for &axis in axes {
            self.chek_axis(axis)?;
            if seen[axis] {
                return Err(TensorViewError::DuplicateAxis { axis });
            }
            seen[axis] = true;
        }

        Ok(Self {
            source: self.source,
            shape: axes.iter().map(|&axis| self.shape[axis]).collect(),
            strides: axes.iter().map(|&axis| self.strides[axis]).collect(),
            base_offset: self.base_offset,
            len: self.len,
        })
    }

    /// Checks whether the given axis exists
    fn chek_axis(&self, axis: usize) -> Result<(), TensorViewError> {
        if axis >= self.rank() {
            return Err(TensorViewError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }
        Ok(())
    }

    /// Selects a half-open, unit-step range on one axis without copying values.
    pub fn slice(&self, axis: usize, range: Range<usize>) -> Result<Self, TensorViewError> {
        self.chek_axis(axis)?;
        if range.start > range.end {
            return Err(TensorViewError::SliceStartAfterEnd {
                axis,
                start: range.start,
                end: range.end,
            });
        }

        let dimension = self.shape[axis];
        if range.end > dimension {
            return Err(TensorViewError::SliceEndOutOfBounds {
                axis,
                end: range.end,
                dimension,
            });
        }

        let start_offset = range
            .start
            .checked_mul(self.strides[axis])
            .ok_or(TensorViewError::Tensor(TensorError::ShapeOverflow))?;
        let base_offset = self
            .base_offset
            .checked_add(start_offset)
            .ok_or(TensorViewError::Tensor(TensorError::ShapeOverflow))?;
        let mut shape = self.shape.clone();
        shape[axis] = range.end - range.start;
        let (_, len) = checked_row_major_layout(&shape)?;

        Ok(Self {
            source: self.source,
            shape,
            strides: self.strides.clone(),
            base_offset,
            len,
        })
    }

    /// Copies logical row-major values into a new owned, contiguous tensor.
    pub fn materialize(&self) -> Result<Tensor, TensorViewError> {
        let mut values = Vec::with_capacity(self.len);
        for storage_offset in self.logical_offsets() {
            values.push(self.value_at_storage_offset(storage_offset));
        }
        Tensor::from_vec(self.shape.clone(), values).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> Tensor {
        Tensor::from_vec(vec![2, 3], vec![10.0, 11.0, 12.0, 21.0, 22.0, 23.0]).unwrap()
    }

    mod tensor {
        use super::*;

        mod fn_view {
            use super::*;

            #[test]
            fn it_computes_tensor_view() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(view.source, &tensor);
                assert_eq!(view.shape, vec![2, 3]);
                assert_eq!(view.strides, vec![3, 1]);
                assert_eq!(view.base_offset, 0);
                assert_eq!(view.len, 6);
            }
        }
    }

    mod tensor_view {
        use super::*;

        mod fn_is_contiguous {
            use super::*;

            mod when_view_is_empty {
                use super::*;

                #[test]
                fn it_returns_true() {
                    let tensor = Tensor::from_vec(vec![0], vec![]).unwrap();
                    let view = tensor.view();

                    assert_eq!(view.is_contiguous(), true);
                }
            }

            mod when_view_is_not_contiguous {
                use super::*;

                #[test]
                fn it_returns_false() {
                    let tensor = tensor();
                    let mut view = tensor.view();
                    view.shape = vec![2, 2];

                    assert_eq!(view.is_contiguous(), false);
                }
            }

            mod when_view_is_contiguous {
                use super::*;

                #[test]
                fn it_returns_true() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(view.is_contiguous(), true);
                }
            }
        }

        mod fn_storage_offset {
            use super::*;

            #[test]
            fn it_calculates_offset_by_the_given_coordinate() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(view.storage_offset(&[1, 2]), Ok(5));
            }
        }

        mod fn_get {
            use super::*;

            #[test]
            fn it_returns_data_value_by_the_given_coordinate() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(view.get(&[1, 2]), Ok(&23.0));
            }
        }

        mod fn_logical_offsets {
            use super::*;

            #[test]
            fn it_calculates_stride_offsets() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(
                    view.logical_offsets(),
                    StrideOffsets {
                        axes: vec![
                            OffsetAxis {
                                extent: 2,
                                stride: 3,
                                position: 0,
                                rewind: 3
                            },
                            OffsetAxis {
                                extent: 3,
                                stride: 1,
                                position: 0,
                                rewind: 2
                            }
                        ],
                        next_offset: 0,
                        remaining: 6
                    }
                );
            }
        }

        mod fn_projected_offsets {
            use super::*;

            #[test]
            fn it_returns_checked_stride_offsets_by_the_given_input() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(
                    view.projected_offsets(&[2, 2], &[3, 1], 4),
                    Some(StrideOffsets {
                        axes: vec![
                            OffsetAxis {
                                extent: 2,
                                stride: 3,
                                position: 0,
                                rewind: 3
                            },
                            OffsetAxis {
                                extent: 2,
                                stride: 1,
                                position: 0,
                                rewind: 1
                            }
                        ],
                        next_offset: 0,
                        remaining: 4
                    })
                );
            }
        }

        mod fn_value_at_storage_offset {
            use super::*;

            #[test]
            fn it_returns_data_value_at_the_given_offset() {
                let tensor = tensor();
                let view = tensor.view();

                assert_eq!(view.value_at_storage_offset(3), 21.0);
            }
        }

        mod fn_reshape {
            use super::*;

            mod when_new_shape_calculated_length_does_not_match_current_data_length {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.reshape(&[3, 3]),
                        Err(TensorViewError::ReshapeElementCountMismatch {
                            current: view.len,
                            requested: 9,
                        })
                    );
                }
            }

            mod when_reshaping_non_contiguous_view {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view().transpose(0, 1).unwrap();

                    assert_eq!(
                        view.reshape(&[3, 2]),
                        Err(TensorViewError::NonContiguousReshape)
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_returns_reshaped_view() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.reshape(&[3, 2]),
                        Ok(TensorView {
                            source: &tensor,
                            shape: vec![3, 2],
                            strides: vec![2, 1],
                            base_offset: view.base_offset,
                            len: view.len
                        })
                    );
                }
            }
        }

        mod fn_transpose {
            use super::*;

            mod when_first_axis_to_swap_is_out_of_bounds {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.transpose(2, 1),
                        Err(TensorViewError::AxisOutOfBounds { axis: 2, rank: 2 })
                    );
                }
            }

            mod when_second_axis_to_swap_is_out_of_bounds {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.transpose(0, 2),
                        Err(TensorViewError::AxisOutOfBounds { axis: 2, rank: 2 })
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_swaps_places_of_given_axes() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.transpose(0, 1),
                        Ok(TensorView {
                            source: &tensor,
                            shape: vec![3, 2],
                            strides: vec![1, 3],
                            base_offset: view.base_offset,
                            len: view.len
                        })
                    );
                }
            }
        }

        mod fn_permute {
            use super::*;

            mod when_given_axes_length_does_not_match_current_axes_length {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.permute(&[1, 2, 3]),
                        Err(TensorViewError::PermutationLengthMismatch {
                            expected: 2,
                            actual: 3,
                        })
                    );
                }
            }

            mod when_some_axis_is_duplicated {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.permute(&[0, 0]),
                        Err(TensorViewError::DuplicateAxis { axis: 0 })
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_aligns_shape_and_strides_according_to_the_new_axis_order() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.permute(&[1, 0]),
                        Ok(TensorView {
                            source: view.source,
                            shape: vec![3, 2],
                            strides: vec![1, 3],
                            base_offset: view.base_offset,
                            len: view.len
                        })
                    );
                }
            }
        }

        mod fn_slice {
            use super::*;

            mod when_axis_is_out_of_bounds {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.slice(2, 0..2),
                        Err(TensorViewError::AxisOutOfBounds { axis: 2, rank: 2 })
                    );
                }
            }

            mod when_range_is_descending {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.slice(1, 2..0),
                        Err(TensorViewError::SliceStartAfterEnd {
                            axis: 1,
                            start: 2,
                            end: 0,
                        })
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_sliced_tensor_view() {
                    let tensor = tensor();
                    let view = tensor.view();

                    assert_eq!(
                        view.slice(0, 1..2),
                        Ok(TensorView {
                            source: view.source,
                            shape: vec![1, 3],
                            strides: vec![3, 1],
                            base_offset: 3,
                            len: 3,
                        })
                    );
                }
            }
        }

        mod fn_materialize {
            use super::*;

            #[test]
            fn it_computes_new_contiguous_tensor() {
                let tensor = tensor();
                let view = tensor.view().slice(1, 1..3).unwrap();
                let result = view.materialize();

                assert_eq!(
                    result.as_ref().unwrap(),
                    &Tensor::from_vec(vec![2, 2], vec![11.0, 12.0, 22.0, 23.0]).unwrap()
                );
                assert!(result.as_ref().unwrap().view().is_contiguous());
            }
        }
    }

    mod stride_offsets {
        use super::*;

        mod fn_checked {
            use super::*;

            mod when_shape_length_does_not_match_strides_length {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let result = StrideOffsets::checked(&[2, 3], &[3, 2, 1], 0, 6, 6);

                    assert_eq!(result, None);
                }
            }

            mod when_logical_length_does_not_match_expected_length {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let result = StrideOffsets::checked(&[2, 3], &[3, 1], 0, 5, 6);

                    assert_eq!(result, None);
                }
            }

            mod when_logical_length_is_zero {
                use super::*;

                #[test]
                fn it_returns_zero_remaining_stride_offsets() {
                    let shape = [2, 0];
                    let strides = [3, 1];
                    let base_offsets = 0;
                    let logical_length = 0;
                    let backing_length = 6;
                    let result = StrideOffsets::checked(
                        &shape,
                        &strides,
                        base_offsets,
                        logical_length,
                        backing_length,
                    );

                    assert_eq!(
                        result,
                        Some(StrideOffsets {
                            axes: vec![
                                OffsetAxis {
                                    extent: 2,
                                    stride: 3,
                                    position: 0,
                                    rewind: 0
                                },
                                OffsetAxis {
                                    extent: 0,
                                    stride: 1,
                                    position: 0,
                                    rewind: 0
                                }
                            ],
                            next_offset: base_offsets,
                            remaining: logical_length
                        })
                    );
                }
            }

            mod when_max_offset_is_gte_backing_length {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let shape = [2, 2];
                    let strides = [3, 1];
                    let base_offsets = 2;
                    let logical_length = 4;
                    let backing_length = 6;
                    let result = StrideOffsets::checked(
                        &shape,
                        &strides,
                        base_offsets,
                        logical_length,
                        backing_length,
                    );

                    assert_eq!(result, None);
                }
            }

            mod when_offsets_can_fit_into_boundaries {
                use super::*;

                #[test]
                fn it_returns_offsets() {
                    let shape = [2, 2];
                    let strides = [3, 1];
                    let base_offsets = 1;
                    let logical_length = 4;
                    let backing_length = 6;
                    let result = StrideOffsets::checked(
                        &shape,
                        &strides,
                        base_offsets,
                        logical_length,
                        backing_length,
                    );

                    assert_eq!(
                        result,
                        Some(StrideOffsets {
                            axes: vec![
                                OffsetAxis {
                                    extent: 2,
                                    stride: 3,
                                    position: 0,
                                    rewind: 3
                                },
                                OffsetAxis {
                                    extent: 2,
                                    stride: 1,
                                    position: 0,
                                    rewind: 1
                                }
                            ],
                            next_offset: base_offsets,
                            remaining: logical_length
                        })
                    );
                }
            }
        }

        mod fn_next {
            use super::*;

            mod when_there_are_remaining_items {
                use super::*;

                #[test]
                fn it_calculates_offsets() {
                    let shape = [2, 2];
                    let strides = [3, 1];
                    let base_offsets = 1;
                    let logical_length = 4;
                    let backing_length = 6;
                    let stride_offsets = StrideOffsets::checked(
                        &shape,
                        &strides,
                        base_offsets,
                        logical_length,
                        backing_length,
                    )
                    .unwrap();

                    assert_eq!(stride_offsets.collect::<Vec<_>>(), vec![1, 2, 4, 5]);
                }
            }

            mod when_there_are_no_remaining_items {
                use super::*;

                #[test]
                fn it_stops_iteration() {
                    let shape = [0, 2];
                    let strides = [3, 1];
                    let base_offsets = 1;
                    let logical_length = 0;
                    let backing_length = 6;
                    let stride_offsets = StrideOffsets::checked(
                        &shape,
                        &strides,
                        base_offsets,
                        logical_length,
                        backing_length,
                    )
                    .unwrap();

                    assert_eq!(stride_offsets.collect::<Vec<_>>(), Vec::<usize>::new());
                }
            }
        }
    }
}
