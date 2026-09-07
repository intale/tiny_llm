//! Checked matrix multiplication over owner or strided tensor views.

use std::error::Error;
use std::fmt;

use super::storage::{Tensor, TensorError, checked_row_major_layout};
use super::view::{TensorView, TensorViewError};

/// A rejected matrix product, output layout, allocation, or converted view operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulError {
    /// An owned output layout violates the tensor storage invariant.
    Tensor(TensorError),
    /// A tensor-view error was converted into the matrix-multiplication error type.
    View(TensorViewError),
    /// Matrix multiplication does not promote a left vector in this chapter.
    LeftRankTooSmall { rank: usize },
    /// Matrix multiplication does not promote a right vector in this chapter.
    RightRankTooSmall { rank: usize },
    /// The effective final left axis and penultimate right axis differ.
    InnerDimensionMismatch { left: usize, right: usize },
    /// Two trailing-aligned batch dimensions are neither equal nor singleton.
    IncompatibleBatch {
        axis: usize,
        left_dimension: usize,
        right_dimension: usize,
    },
    /// The checked output shape is valid, but its value buffer cannot be reserved.
    OutputAllocationFailed { elements: usize },
}

impl fmt::Display for MatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(formatter),
            Self::View(error) => error.fmt(formatter),
            Self::LeftRankTooSmall { rank } => {
                write!(
                    formatter,
                    "left matmul input must have rank at least 2, got {rank}"
                )
            }
            Self::RightRankTooSmall { rank } => {
                write!(
                    formatter,
                    "right matmul input must have rank at least 2, got {rank}"
                )
            }
            Self::InnerDimensionMismatch { left, right } => write!(
                formatter,
                "matmul inner dimensions do not match: left size {left}, right size {right}"
            ),
            Self::IncompatibleBatch {
                axis,
                left_dimension,
                right_dimension,
            } => write!(
                formatter,
                "cannot broadcast batch axis {axis}: left size {left_dimension}, right size {right_dimension}"
            ),
            Self::OutputAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate output buffer for {elements} f64 values"
            ),
        }
    }
}

impl Error for MatmulError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::View(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for MatmulError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorViewError> for MatmulError {
    fn from(error: TensorViewError) -> Self {
        Self::View(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EffectiveMatrixLayout {
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}

/// Reads the final two axes as rows and columns, swapping their extents and strides when
/// transposed.
fn effective_matrix_layout(input: &TensorView<'_>, transposed: bool) -> EffectiveMatrixLayout {
    // Calculate offset to matrix axes(last two axes of the input)
    let matrix_axis_offset = input.rank() - 2;
    let stored = EffectiveMatrixLayout {
        rows: input.shape()[matrix_axis_offset],
        columns: input.shape()[matrix_axis_offset + 1],
        row_stride: input.strides()[matrix_axis_offset],
        column_stride: input.strides()[matrix_axis_offset + 1],
    };
    if transposed {
        EffectiveMatrixLayout {
            rows: stored.columns,
            columns: stored.rows,
            row_stride: stored.column_stride,
            column_stride: stored.row_stride,
        }
    } else {
        stored
    }
}

/// Right-aligns two batch shapes and returns their checked singleton-broadcast output shape.
fn broadcast_batch_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, MatmulError> {
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
            return Err(MatmulError::IncompatibleBatch {
                axis,
                left_dimension,
                right_dimension,
            });
        };
        output.push(dimension);
    }

    Ok(output)
}

/// Projects input strides onto output batch axes, using zero for missing or singleton axes.
fn batch_effective_strides(input: &TensorView<'_>, output_batch_rank: usize) -> Vec<usize> {
    // We cut off last 2 axes as they are matrix axes. All axes that come before are batch axes.
    // Thus, we need to:
    // - cut non-batch axes by reducing input's rank by 2
    // - align strides of batch axes to output_batch_rank rank
    // So, to make it clear - the method does the same thing as broadcast_effective_strides() from
    // opts.rs, but it does it only for batch axes.
    let input_batch_rank = input.rank() - 2;
    let padding = output_batch_rank - input_batch_rank;

    (0..output_batch_rank)
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

/// Safely reserves an empty `f64` buffer with capacity of the requested size
fn output_buffer(elements: usize) -> Result<Vec<f64>, MatmulError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| MatmulError::OutputAllocationFailed { elements })?;
    Ok(values)
}

#[derive(Debug, PartialEq)]
struct MatmulPlan {
    /// The owned result's logical row-major shape and element count.
    output_shape: Vec<usize>,
    output_len: usize,
    /// The number of products accumulated into each output cell.
    inner: usize,
    /// One source-storage movement per output batch, row, and column axis.
    left_cell_strides: Vec<usize>,
    right_cell_strides: Vec<usize>,
    /// Source-storage movement when the contracted index increases by one.
    left_inner_stride: usize,
    right_inner_stride: usize,
}

impl MatmulPlan {
    fn new(
        left: &TensorView<'_>,
        right: &TensorView<'_>,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<Self, MatmulError> {
        if left.rank() < 2 {
            return Err(MatmulError::LeftRankTooSmall { rank: left.rank() });
        }
        if right.rank() < 2 {
            return Err(MatmulError::RightRankTooSmall { rank: right.rank() });
        }

        let left_matrix = effective_matrix_layout(left, transpose_left);
        let right_matrix = effective_matrix_layout(right, transpose_right);
        let rows = left_matrix.rows;
        let inner = left_matrix.columns;
        let right_inner = right_matrix.rows;
        let columns = right_matrix.columns;

        if inner != right_inner {
            return Err(MatmulError::InnerDimensionMismatch {
                left: inner,
                right: right_inner,
            });
        }

        let left_batch_shape = &left.shape()[..left.rank() - 2];
        let right_batch_shape = &right.shape()[..right.rank() - 2];
        let batch_shape = broadcast_batch_shape(left_batch_shape, right_batch_shape)?;
        let mut output_shape = batch_shape.clone();
        output_shape.extend([rows, columns]); // [batch..., M, N]
        let (_, output_len) = checked_row_major_layout(&output_shape)?;

        let batch_rank = batch_shape.len();
        let mut left_cell_strides = batch_effective_strides(left, batch_rank);
        left_cell_strides.extend([left_matrix.row_stride, 0]); // Strides for M, K
        let mut right_cell_strides = batch_effective_strides(right, batch_rank);
        right_cell_strides.extend([0, right_matrix.column_stride]); // Strides for K, N

        Ok(Self {
            output_shape,
            output_len,
            inner, // K
            left_cell_strides,
            right_cell_strides,
            left_inner_stride: left_matrix.column_stride,
            right_inner_stride: right_matrix.row_stride,
        })
    }
}

/// Multiplies two rank-two or batched tensor views as stored.
pub fn matmul(left: &TensorView<'_>, right: &TensorView<'_>) -> Result<Tensor, MatmulError> {
    matmul_with_transpose(left, right, false, false)
}

/// Multiplies two tensor views after optional logical final-axis transposes.
///
/// Inputs must have rank at least two. Only axes before the final two matrix axes broadcast, using
/// trailing alignment. The effective inner dimensions must match exactly. The scalar contraction
/// visits `k` in ascending order and reads through offsets established by one checked strided plan
/// per operand.
pub fn matmul_with_transpose(
    left: &TensorView<'_>,
    right: &TensorView<'_>,
    transpose_left: bool,
    transpose_right: bool,
) -> Result<Tensor, MatmulError> {
    let plan = MatmulPlan::new(left, right, transpose_left, transpose_right)?;
    let mut values = output_buffer(plan.output_len)?;
    if plan.inner == 0 {
        values.resize(plan.output_len, 0.0);
        return Tensor::from_vec(plan.output_shape, values).map_err(Into::into);
    }

    let left_cell_offsets = left
        .projected_offsets(&plan.output_shape, &plan.left_cell_strides, plan.output_len)
        .expect("a checked matmul plan retains valid left cell offsets");
    let right_cell_offsets = right
        .projected_offsets(
            &plan.output_shape,
            &plan.right_cell_strides,
            plan.output_len,
        )
        .expect("a checked matmul plan retains valid right cell offsets");

    for (left_cell_offset, right_cell_offset) in left_cell_offsets.zip(right_cell_offsets) {
        let mut sum = 0.0;
        let mut left_offset = left_cell_offset;
        let mut right_offset = right_cell_offset;
        for inner_index in 0..plan.inner {
            let left_value = left.value_at_storage_offset(left_offset);
            let right_value = right.value_at_storage_offset(right_offset);

            sum += left_value * right_value;

            if inner_index + 1 < plan.inner {
                left_offset = left_offset
                    .checked_add(plan.left_inner_stride)
                    .expect("a checked matmul plan cannot overflow along the left inner axis");
                right_offset = right_offset
                    .checked_add(plan.right_inner_stride)
                    .expect("a checked matmul plan cannot overflow along the right inner axis");
            }
        }
        values.push(sum);
    }

    Tensor::from_vec(plan.output_shape, values).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fn_broadcast_batch_shape {
        use super::*;

        mod when_shapes_are_not_compatible {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape1 = [1, 2, 3];
                let shape2 = [1, 3, 2];
                let result = broadcast_batch_shape(&shape1, &shape2);

                assert_eq!(
                    result,
                    Err(MatmulError::IncompatibleBatch {
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
                let result = broadcast_batch_shape(&shape1, &shape2);

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
                let result = broadcast_batch_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_left_shape_size_is_less_than_right_shape_size {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [2, 3];
                let shape2 = [1, 2, 3];
                let result = broadcast_batch_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_right_shape_size_is_less_than_left_shape_size {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [1, 2, 3];
                let shape2 = [2, 3];
                let result = broadcast_batch_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![1, 2, 3]));
            }
        }

        mod when_shapes_are_equal {
            use super::*;

            #[test]
            fn it_computes_adjusted_shape() {
                let shape1 = [2, 2, 3];
                let shape2 = [2, 2, 3];
                let result = broadcast_batch_shape(&shape1, &shape2);

                assert_eq!(result, Ok(vec![2, 2, 3]));
            }
        }
    }

    mod fn_batch_effective_strides {
        use super::*;

        #[test]
        fn it_aligns_strides_of_batch_axes() {
            // Strides: [2, 1, 1, 1, 1]
            let tensor = Tensor::from_vec(
                vec![3, 2, 1, 1, 1],
                vec![10.0, 11.0, 12.0, 21.0, 22.0, 23.0],
            )
            .unwrap();
            let view = tensor.view();

            assert_eq!(batch_effective_strides(&view, 5), vec![0, 0, 2, 1, 0]);
        }
    }

    mod fn_effective_matrix_layout {
        use super::*;

        mod computing_transposed_matrix {
            use super::*;

            #[test]
            fn it_computes_transposed_effective_matrix_layout_from_the_input() {
                // Strides: [6, 2, 1]
                let tensor = Tensor::from_vec(vec![4, 3, 2], vec![0.0; 24]).unwrap();
                let view = tensor.view();

                assert_eq!(
                    effective_matrix_layout(&view, true),
                    EffectiveMatrixLayout {
                        rows: 2,
                        columns: 3,
                        row_stride: 1,
                        column_stride: 2
                    }
                );
            }
        }

        mod computing_original_matrix {
            use super::*;

            #[test]
            fn it_computes_effective_matrix_layout_from_the_input() {
                // Strides: [6, 2, 1]
                let tensor = Tensor::from_vec(vec![4, 3, 2], vec![0.0; 24]).unwrap();
                let view = tensor.view();

                assert_eq!(
                    effective_matrix_layout(&view, false),
                    EffectiveMatrixLayout {
                        rows: 3,
                        columns: 2,
                        row_stride: 2,
                        column_stride: 1
                    }
                );
            }
        }
    }

    mod matmul_plan {
        use super::*;

        mod fn_new {
            use super::*;

            #[test]
            fn it_builds_matmul_plan() {
                // Strides: [6, 2, 1, 1]
                let tensor1 = Tensor::from_vec(vec![4, 3, 2, 1], vec![0.0; 24]).unwrap();
                // Strides: [2, 2, 1]
                let tensor2 = Tensor::from_vec(vec![3, 1, 2], vec![0.0; 6]).unwrap();
                let view1 = tensor1.view();
                let view2 = tensor2.view();

                assert_eq!(
                    MatmulPlan::new(&view1, &view2, false, false).unwrap(),
                    MatmulPlan {
                        output_shape: vec![4, 3, 2, 2],
                        output_len: 48,
                        inner: 1,
                        left_cell_strides: vec![6, 2, 1, 0],
                        right_cell_strides: vec![0, 2, 0, 1],
                        left_inner_stride: 1,
                        right_inner_stride: 2
                    }
                );
            }
        }
    }

    mod fn_matmul_with_transpose {
        use super::*;

        mod without_transpose {
            use super::*;

            #[test]
            fn it_computes_new_tensor_which_is_a_product_of_two_given_tensor_views() {
                // Strides: [2, 1, 1]
                let tensor1 =
                    Tensor::from_vec(vec![3, 2, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
                // Strides: [3, 3, 1]
                let tensor2 = Tensor::from_vec(
                    vec![3, 1, 3],
                    vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                )
                    .unwrap();
                let view1 = tensor1.view();
                let view2 = tensor2.view();

                // MatmulPlan:
                // output_shape: 3, 2, 3
                // output_len: 18
                // inner: 1
                // left_cell_strides: 2, 1, 0
                // right_cell_strides: 3, 0, 1
                // left_inner_stride: 1
                // right_inner_stride: 3
                let result = matmul_with_transpose(&view1, &view2, false, false).unwrap();

                // left_cell_offsets:  [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                // right_cell_offsets: [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8]
                assert_eq!(
                    result,
                    Tensor::from_vec(
                        vec![3, 2, 3],
                        vec![
                            1.0 * 10.0, 1.0 * 20.0, 1.0 * 30.0,
                            2.0 * 10.0, 2.0 * 20.0, 2.0 * 30.0,
                            3.0 * 40.0, 3.0 * 50.0, 3.0 * 60.0,
                            4.0 * 40.0, 4.0 * 50.0, 4.0 * 60.0,
                            5.0 * 70.0, 5.0 * 80.0, 5.0 * 90.0,
                            6.0 * 70.0, 6.0 * 80.0, 6.0 * 90.0
                        ]
                    ).unwrap()
                );
            }
        }

        mod with_transpose {
            use super::*;

            #[test]
            fn it_computes_new_tensor_which_is_a_product_of_two_given_tensor_views() {
                // Strides: [12, 4, 1]
                let tensor1 =
                    Tensor::from_vec(
                        vec![2, 3, 4], // 2 batches 3 x 4
                        vec![
                            // batch 0
                            1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0, 12.0,

                            // batch 1
                            13.0, 14.0, 15.0, 16.0,
                            17.0, 18.0, 19.0, 20.0,
                            21.0, 22.0, 23.0, 24.0,
                        ]
                    ).unwrap();
                // Strides: [20, 4, 1]
                let tensor2 = Tensor::from_vec(
                    vec![2, 5, 4], // 2 batches 5 x 4
                    vec![
                        // batch 0
                        1.0,  2.0,  3.0,  4.0,
                        5.0,  6.0,  7.0,  8.0,
                        9.0, 10.0, 11.0, 12.0,
                        13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0,

                        // batch 1
                        21.0, 22.0, 23.0, 24.0,
                        25.0, 26.0, 27.0, 28.0,
                        29.0, 30.0, 31.0, 32.0,
                        33.0, 34.0, 35.0, 36.0,
                        37.0, 38.0, 39.0, 40.0,
                    ],
                )
                    .unwrap();
                let view1 = tensor1.view();
                let view2 = tensor2.view();

                // MatmulPlan:
                // output_shape: 2, 3, 5
                // output_len: 30
                // inner: 4
                // left_cell_strides: 12, 4, 0
                // right_cell_strides: 20, 0, 4
                // left_inner_stride: 1
                // right_inner_stride: 1
                let result = matmul_with_transpose(&view1, &view2, false, true).unwrap();

                // left_cell_offsets: [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20],
                // right_cell_offsets: [0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 20, 24, 28, 32, 36, 20, 24, 28, 32, 36]
                assert_eq!(
                    result,
                    Tensor::from_vec(
                        vec![2, 3, 5],
                        vec![
                            30.0,  70.0,  110.0, 150.0, 190.0,
                            70.0,  174.0, 278.0, 382.0, 486.0,
                            110.0, 278.0, 446.0, 614.0, 782.0,

                            1310.0, 1542.0, 1774.0, 2006.0, 2238.0,
                            1670.0, 1966.0, 2262.0, 2558.0, 2854.0,
                            2030.0, 2390.0, 2750.0, 3110.0, 3470.0
                        ]
                    ).unwrap()
                );
            }
        }


    }
}
