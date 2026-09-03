//! Checked elementwise tensor maps, trailing-axis broadcasting, and axis reductions.

use std::error::Error;
use std::fmt;

use super::storage::{Tensor, TensorError, checked_row_major_layout};
use super::view::{TensorView, TensorViewError};

/// A rejected elementwise operation, broadcast plan, or axis reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOpError {
    /// An owned output layout or buffer violates the tensor storage invariant.
    Tensor(TensorError),
    /// A checked logical read from an input view failed.
    View(TensorViewError),
    /// Two trailing-aligned dimensions are neither equal nor singleton.
    IncompatibleBroadcast {
        axis: usize,
        left_dimension: usize,
        right_dimension: usize,
    },
    /// A reduction names an axis that the input does not have.
    ReductionAxisOutOfBounds { axis: usize, rank: usize },
    /// Mean has no value when the selected axis contains no elements.
    EmptyMeanAxis { axis: usize },
    /// Maximum has no value when the selected axis contains no elements.
    EmptyMaxAxis { axis: usize },
    /// The checked output shape is valid, but its value buffer cannot be reserved.
    OutputAllocationFailed { elements: usize },
}

impl fmt::Display for TensorOpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(formatter),
            Self::View(error) => error.fmt(formatter),
            Self::IncompatibleBroadcast {
                axis,
                left_dimension,
                right_dimension,
            } => write!(
                formatter,
                "cannot broadcast output axis {axis}: left size {left_dimension}, right size {right_dimension}"
            ),
            Self::ReductionAxisOutOfBounds { axis, rank } => {
                write!(
                    formatter,
                    "reduction axis {axis} is out of bounds for rank {rank}"
                )
            }
            Self::EmptyMeanAxis { axis } => {
                write!(formatter, "cannot compute mean over empty axis {axis}")
            }
            Self::EmptyMaxAxis { axis } => {
                write!(formatter, "cannot compute max over empty axis {axis}")
            }
            Self::OutputAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate output buffer for {elements} f64 values"
            ),
        }
    }
}

impl Error for TensorOpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::View(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for TensorOpError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorViewError> for TensorOpError {
    fn from(error: TensorViewError) -> Self {
        Self::View(error)
    }
}

/// Computes the checked output shape for trailing-axis broadcasting.
///
/// Missing leading dimensions act as size one. Aligned dimensions are compatible when they are
/// equal or either one is size one. Compatibility is reported from the leftmost aligned output axis
/// before layout overflow.
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorOpError> {
    let output_rank = left.len().max(right.len());
    let left_padding = output_rank - left.len();
    let right_padding = output_rank - right.len();
    let mut output = Vec::with_capacity(output_rank);

    for axis in 0..output_rank {
        let left_dimension = left
            .get(axis.wrapping_sub(left_padding))
            .copied()
            .unwrap_or(1);
        let right_dimension = right
            .get(axis.wrapping_sub(right_padding))
            .copied()
            .unwrap_or(1);
        let dimension = if left_dimension == right_dimension {
            left_dimension
        } else if left_dimension == 1 {
            right_dimension
        } else if right_dimension == 1 {
            left_dimension
        } else {
            return Err(TensorOpError::IncompatibleBroadcast {
                axis,
                left_dimension,
                right_dimension,
            });
        };
        output.push(dimension);
    }

    checked_row_major_layout(&output)?;
    Ok(output)
}

/// Tries to allocate Vec of `elements` size of `f64` values. Returns newly created vector in case
/// of success.
fn output_buffer(elements: usize) -> Result<Vec<f64>, TensorOpError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| TensorOpError::OutputAllocationFailed { elements })?;
    Ok(values)
}

/// Applies one scalar function in logical row-major order and owns the result.
pub fn map_unary<F>(input: &TensorView<'_>, mut operation: F) -> Result<Tensor, TensorOpError>
where
    F: FnMut(f64) -> f64,
{
    let mut values = output_buffer(input.len())?;
    for input_offset in input.logical_offsets() {
        values.push(operation(input.value_at_storage_offset(input_offset)));
    }
    Tensor::from_vec(input.shape().to_vec(), values).map_err(Into::into)
}

fn broadcast_effective_strides(input: &TensorView<'_>, output_rank: usize) -> Vec<usize> {
    let padding = output_rank - input.rank();
    (0..output_rank)
        .map(|output_axis| {
            if output_axis < padding {
                return 0;
            }

            let input_axis = output_axis - padding;
            if input.shape()[input_axis] == 1 {
                0
            } else {
                input.strides()[input_axis]
            }
        })
        .collect()
}

/// Applies one scalar function across two trailing-axis-compatible views.
pub fn map_binary<F>(
    left: &TensorView<'_>,
    right: &TensorView<'_>,
    mut operation: F,
) -> Result<Tensor, TensorOpError>
where
    F: FnMut(f64, f64) -> f64,
{
    let output_shape = broadcast_shape(left.shape(), right.shape())?;
    let (_, output_len) = checked_row_major_layout(&output_shape)?;
    let mut values = output_buffer(output_len)?;

    let left_strides = broadcast_effective_strides(left, output_shape.len());
    let right_strides = broadcast_effective_strides(right, output_shape.len());
    let left_offsets = left
        .projected_offsets(&output_shape, &left_strides, output_len)
        .expect("a compatible broadcast retains a valid left traversal plan");
    let right_offsets = right
        .projected_offsets(&output_shape, &right_strides, output_len)
        .expect("a compatible broadcast retains a valid right traversal plan");

    for (left_offset, right_offset) in left_offsets.zip(right_offsets) {
        let left_value = left.value_at_storage_offset(left_offset);
        let right_value = right.value_at_storage_offset(right_offset);
        values.push(operation(left_value, right_value));
    }

    Tensor::from_vec(output_shape, values).map_err(Into::into)
}

#[derive(Copy, Clone)]
enum Reduction {
    Sum,
    Mean,
    Max,
}

/// Removes the reduced axis or retains it with extent 1 in the output shape.
fn reduction_shape(input_shape: &[usize], axis: usize, keep_dim: bool) -> Vec<usize> {
    if keep_dim {
        let mut output = input_shape.to_vec();
        output[axis] = 1;
        return output;
    }

    input_shape
        .iter()
        .enumerate()
        .filter_map(|(input_axis, &dimension)| (input_axis != axis).then_some(dimension))
        .collect()
}

/// Maps output-group coordinates to source bases by removing or zeroing the reduced axis.
fn reduction_group_strides(input_strides: &[usize], axis: usize, keep_dim: bool) -> Vec<usize> {
    if keep_dim {
        let mut group_strides = input_strides.to_vec();
        group_strides[axis] = 0;
        return group_strides;
    }

    input_strides
        .iter()
        .enumerate()
        .filter_map(|(input_axis, &stride)| (input_axis != axis).then_some(stride))
        .collect()
}

/// Executes one checksum, mean or maximum reduction and owns its output.
fn reduce_axis(
    input: &TensorView<'_>,
    axis: usize,
    keep_dim: bool,
    reduction: Reduction,
) -> Result<Tensor, TensorOpError> {
    if axis >= input.rank() {
        return Err(TensorOpError::ReductionAxisOutOfBounds {
            axis,
            rank: input.rank(),
        });
    }

    let axis_len = input.shape()[axis];
    match reduction {
        Reduction::Mean if axis_len == 0 => {
            return Err(TensorOpError::EmptyMeanAxis { axis });
        }
        Reduction::Max if axis_len == 0 => {
            return Err(TensorOpError::EmptyMaxAxis { axis });
        }
        _ => {}
    }

    let output_shape = reduction_shape(input.shape(), axis, keep_dim);
    let (_, output_len) = checked_row_major_layout(&output_shape)?;
    let mut values = output_buffer(output_len)?;

    if axis_len == 0 {
        values.resize(output_len, 0.0);
        return Tensor::from_vec(output_shape, values).map_err(Into::into);
    }

    let group_strides = reduction_group_strides(input.strides(), axis, keep_dim);
    let group_offsets = input
        .projected_offsets(&output_shape, &group_strides, output_len)
        .expect("a checked reduction retains a valid group traversal plan");
    let axis_stride = input.strides()[axis];

    for group_offset in group_offsets {
        let value = match reduction {
            Reduction::Sum | Reduction::Mean => {
                let mut total = 0.0;
                let mut input_offset = group_offset;
                for index in 0..axis_len {
                    total += input.value_at_storage_offset(input_offset);
                    if index + 1 < axis_len {
                        input_offset = input_offset
                            .checked_add(axis_stride)
                            .expect("a checked view cannot overflow along a reduction axis");
                    }
                }
                if matches!(reduction, Reduction::Mean) {
                    total / axis_len as f64
                } else {
                    total
                }
            }
            Reduction::Max => {
                let mut input_offset = group_offset;
                let mut maximum = input.value_at_storage_offset(input_offset);
                for _ in 1..axis_len {
                    // NaN value persists into final result, and should not be ignored. Also, no
                    // point to continue after the first found NaN
                    if maximum.is_nan() {
                        break;
                    }

                    input_offset = input_offset
                        .checked_add(axis_stride)
                        .expect("a checked view cannot overflow along a reduction axis");
                    let candidate = input.value_at_storage_offset(input_offset);
                    if !maximum.is_nan() && (candidate.is_nan() || candidate > maximum) {
                        maximum = candidate
                    }
                }
                maximum
            }
        };
        values.push(value);
    }

    Tensor::from_vec(output_shape, values).map_err(Into::into)
}

/// Sums one explicit axis in ascending index order.
///
/// An empty selected axis uses the additive identity, so every output group is `0.0`. `keep_dim`
/// replaces the selected extent with one instead of removing the axis.
pub fn sum_axis(
    input: &TensorView<'_>,
    axis: usize,
    keep_dim: bool,
) -> Result<Tensor, TensorOpError> {
    reduce_axis(input, axis, keep_dim, Reduction::Sum)
}

/// Averages one explicit nonempty axis in ascending index order.
pub fn mean_axis(
    input: &TensorView<'_>,
    axis: usize,
    keep_dim: bool,
) -> Result<Tensor, TensorOpError> {
    reduce_axis(input, axis, keep_dim, Reduction::Mean)
}

/// Selects the maximum over one explicit nonempty axis.
///
/// The fold propagates the first NaN and keeps the earlier value on equal comparisons, including
/// the earlier signed-zero bit pattern.
pub fn max_axis(
    input: &TensorView<'_>,
    axis: usize,
    keep_dim: bool,
) -> Result<Tensor, TensorOpError> {
    reduce_axis(input, axis, keep_dim, Reduction::Max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> Tensor {
        Tensor::from_vec(vec![2, 3], vec![10.0, 11.0, 12.0, 21.0, 22.0, 23.0]).unwrap()
    }

    mod fn_sum_axis {
        use super::*;

        #[test]
        fn it_calculates_axis_sum() {
            let tensor = tensor();
            let tensor_view = tensor.view();
            let keep_dimension = false;
            let axis = 1;

            assert_eq!(
                sum_axis(&tensor_view, axis, keep_dimension),
                Ok(Tensor::from_vec(vec![2], vec![33.0, 66.0]).unwrap())
            )
        }
    }

    mod fn_mean_axis {
        use super::*;

        #[test]
        fn it_calculates_axis_mean() {
            let tensor = tensor();
            let tensor_view = tensor.view();
            let keep_dimension = false;
            let axis = 1;

            assert_eq!(
                mean_axis(&tensor_view, axis, keep_dimension),
                Ok(Tensor::from_vec(vec![2], vec![11.0, 22.0]).unwrap())
            )
        }
    }

    mod fn_max_axis {
        use super::*;

        #[test]
        fn it_calculates_axis_mean() {
            let tensor = tensor();
            let tensor_view = tensor.view();
            let keep_dimension = false;
            let axis = 1;

            assert_eq!(
                max_axis(&tensor_view, axis, keep_dimension),
                Ok(Tensor::from_vec(vec![2], vec![12.0, 23.0]).unwrap())
            )
        }
    }

    mod fn_reduce_axis {
        use super::*;

        mod when_axis_number_is_out_of_bounds {
            use super::*;

            #[test]
            fn it_returns_error() {
                let tensor = tensor();
                let tensor_view = tensor.view();
                let axis = 2;
                let keep_dimension = false;

                assert_eq!(
                    reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Max),
                    Err(TensorOpError::ReductionAxisOutOfBounds { axis: 2, rank: 2 })
                );
            }
        }

        mod when_axis_len_is_0 {
            use super::*;

            fn tensor() -> Tensor {
                Tensor::from_vec(vec![2, 3, 0], vec![]).unwrap()
            }

            mod when_reduction_is_mean {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let tensor_view = tensor.view();
                    let keep_dimension = false;
                    let axis = 2;

                    assert_eq!(
                        reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Mean),
                        Err(TensorOpError::EmptyMeanAxis { axis })
                    );
                }
            }

            mod when_reduction_is_max {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = tensor();
                    let tensor_view = tensor.view();
                    let keep_dimension = false;
                    let axis = 2;

                    assert_eq!(
                        reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Max),
                        Err(TensorOpError::EmptyMaxAxis { axis })
                    );
                }
            }

            mod when_reduction_is_sum {
                use super::*;

                mod when_dimension_is_kept {
                    use super::*;

                    #[test]
                    fn it_reduces_axis_and_adjusts_0_dimension_to_1() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = true;
                        let axis = 2;

                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Sum),
                            Ok(Tensor::from_vec(vec![2, 3, 1], vec![0.0; 6]).unwrap())
                        );
                    }
                }

                mod when_dimension_is_removed {
                    use super::*;

                    #[test]
                    fn it_reduces_axis_and_removes_0_dimension() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = false;
                        let axis = 2;

                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Sum),
                            Ok(Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap())
                        );
                    }
                }
            }
        }

        mod when_axis_len_is_gt_0 {
            use super::*;

            mod when_dimension_is_kept {
                use super::*;

                mod when_reduction_is_sum {
                    use super::*;

                    #[test]
                    fn it_calculates_the_sum_of_given_axis_and_reduces_dimension_to_1() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = true;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Sum),
                            Ok(Tensor::from_vec(vec![1, 3], vec![31.0, 33.0, 35.0]).unwrap())
                        )
                    }
                }

                mod when_reduction_is_mean {
                    use super::*;

                    #[test]
                    fn it_calculates_the_mean_of_given_axis_and_reduces_dimension_to_1() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = true;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Mean),
                            Ok(Tensor::from_vec(vec![1, 3], vec![15.5, 16.5, 17.5]).unwrap())
                        )
                    }
                }

                mod when_reduction_is_max {
                    use super::*;

                    #[test]
                    fn it_calculates_the_max_of_given_axis_and_reduces_dimension_to_1() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = true;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Max),
                            Ok(Tensor::from_vec(vec![1, 3], vec![21.0, 22.0, 23.0]).unwrap())
                        )
                    }
                }
            }

            mod when_dimension_is_removed {
                use super::*;

                mod when_reduction_is_sum {
                    use super::*;

                    #[test]
                    fn it_calculates_the_sum_of_given_axis_and_removes_the_dimension() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = false;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Sum),
                            Ok(Tensor::from_vec(vec![3], vec![31.0, 33.0, 35.0]).unwrap())
                        )
                    }
                }

                mod when_reduction_is_mean {
                    use super::*;

                    #[test]
                    fn it_calculates_the_mean_of_given_axis_and_removes_the_dimension() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = false;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Mean),
                            Ok(Tensor::from_vec(vec![3], vec![15.5, 16.5, 17.5]).unwrap())
                        )
                    }
                }

                mod when_reduction_is_max {
                    use super::*;

                    #[test]
                    fn it_calculates_the_max_of_given_axis_and_removes_the_dimension() {
                        let tensor = tensor();
                        let tensor_view = tensor.view();
                        let keep_dimension = false;
                        let axis = 0;
                        assert_eq!(
                            reduce_axis(&tensor_view, axis, keep_dimension, Reduction::Max),
                            Ok(Tensor::from_vec(vec![3], vec![21.0, 22.0, 23.0]).unwrap())
                        )
                    }
                }
            }
        }
    }

    mod fn_reduction_shape {
        use super::*;

        mod when_dimension_is_kept {
            use super::*;

            #[test]
            fn it_reduces_the_given_axis_dimension_to_1() {
                let keep_dimension = true;

                assert_eq!(reduction_shape(&[2, 3, 4], 1, keep_dimension), [2, 1, 4]);
            }
        }

        mod when_dimension_is_removed {
            use super::*;

            #[test]
            fn it_removes_the_given_axis_dimension() {
                let keep_dimension = false;

                assert_eq!(reduction_shape(&[2, 3, 4], 1, keep_dimension), [2, 4]);
            }
        }
    }

    mod fn_reduction_group_strides {
        use super::*;

        mod when_dimension_is_kept {
            use super::*;

            #[test]
            fn it_reduces_the_given_axis_stride_to_0() {
                let keep_dimension = true;

                assert_eq!(
                    reduction_group_strides(&[2, 3, 4], 1, keep_dimension),
                    [2, 0, 4]
                );
            }
        }

        mod when_dimension_is_removed {
            use super::*;

            #[test]
            fn it_removes_the_given_axis_stride() {
                let keep_dimension = false;

                assert_eq!(
                    reduction_group_strides(&[2, 3, 4], 1, keep_dimension),
                    [2, 4]
                );
            }
        }
    }

    mod fn_map_binary {
        use super::*;

        #[test]
        fn it_computes_new_tensor_from_given_tensor_views_transformed_by_given_function() {
            let tensor1 = Tensor::from_vec(vec![3], vec![30.0, 31.0, 32.0]).unwrap();
            let tensor2 = tensor();
            let view1 = tensor1.view();
            let view2 = tensor2.view();
            let result = map_binary(&view1, &view2, |left, right| left + right);

            assert_eq!(
                result,
                Ok(Tensor::from_vec(vec![2, 3], vec![40.0, 42.0, 44.0, 51.0, 53.0, 55.0]).unwrap())
            );
        }
    }

    mod fn_broadcast_shape {
        use super::*;

        mod when_shapes_are_not_compatible {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape1 = [1, 2, 3];
                let shape2 = [1, 3, 2];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(
                    result,
                    Err(TensorOpError::IncompatibleBroadcast {
                        axis: 1,
                        left_dimension: 2,
                        right_dimension: 3,
                    })
                );
            }
        }

        mod when_shapes_of_same_size_are_compatible_because_of_one_sized_axis_of_right_shape {
            use super::*;

            #[test]
            fn it_returns_adjusted_shape() {
                let shape1 = [1, 2, 3];
                // shape2 is compatible even though position 2 contains value which differs from
                // value of shape1 at same position. Value 1 is an exception and it is acceptable
                let shape2 = [1, 2, 1];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_shapes_of_same_size_are_compatible_because_of_one_sized_axis_of_left_shape {
            use super::*;

            #[test]
            fn it_returns_adjusted_shape() {
                // shape1 is compatible even though position 2 contains value which differs from
                // value of shape2 at same position. Value 1 is an exception and it is acceptable
                let shape1 = [1, 2, 1];
                let shape2 = [1, 2, 3];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_left_shape_size_is_less_than_right_shape_size {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [2, 3];
                let shape2 = [1, 2, 3];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_right_shape_size_is_less_than_left_shape_size {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [1, 2, 3];
                let shape2 = [2, 3];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_shapes_are_equal {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [2, 2, 3];
                let shape2 = [2, 2, 3];
                let result = broadcast_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![2, 2, 3]));
            }
        }
    }

    mod fn_map_unary {
        use super::*;

        #[test]
        fn it_transforms_data_of_the_view_using_given_function() {
            let tensor = tensor();
            let view = tensor.view();
            let result = map_unary(&view, |v| v - 1.0);

            assert_eq!(
                result.unwrap(),
                Tensor::from_vec(
                    view.shape().iter().copied().collect::<Vec<_>>(),
                    vec![9.0, 10.0, 11.0, 20.0, 21.0, 22.0]
                )
                .unwrap(),
            );
        }
    }

    mod fn_broadcast_effective_strides {
        use super::*;

        mod when_axis_contains_single_element {
            use super::*;

            #[test]
            fn it_adjusts_strides_of_the_view_to_the_given_rank() {
                let tensor =
                    Tensor::from_vec(vec![1, 6], vec![10.0, 11.0, 12.0, 21.0, 22.0, 23.0]).unwrap();
                let view = tensor.view();
                let result = broadcast_effective_strides(&view, 3);

                assert_eq!(result, vec![0, 0, 1]);
            }
        }

        mod when_axes_rank_does_not_match_the_given_rank {
            use super::*;

            #[test]
            fn it_adjusts_strides_of_the_view_to_the_given_rank() {
                let tensor = tensor();
                let view = tensor.view();
                let result = broadcast_effective_strides(&view, 3);

                assert_eq!(result, vec![0, 3, 1]);
            }
        }

        mod when_axes_rank_matches_the_given_rank {
            use super::*;

            #[test]
            fn it_returns_same_strides() {
                let tensor = tensor();
                let view = tensor.view();
                let result = broadcast_effective_strides(&view, 2);

                assert_eq!(result, view.strides().iter().copied().collect::<Vec<_>>());
            }
        }
    }
}
