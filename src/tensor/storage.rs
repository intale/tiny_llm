//! Contiguous row-major tensor storage and checked coordinate lookup.

use std::error::Error;
use std::fmt;

/// A rejected tensor layout, buffer, or coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// A suffix product needed by the row-major layout exceeds `usize`.
    ShapeOverflow,
    /// The flat buffer length differs from the shape's element count.
    DataLengthMismatch { expected: usize, actual: usize },
    /// A coordinate supplies a different number of axes from the tensor.
    RankMismatch { expected: usize, actual: usize },
    /// One coordinate index lies outside its axis.
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dimension: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => {
                formatter.write_str("shape does not fit a row-major usize layout")
            }
            Self::DataLengthMismatch { expected, actual } => write!(
                formatter,
                "shape needs {expected} values, but data has {actual}"
            ),
            Self::RankMismatch { expected, actual } => write!(
                formatter,
                "coordinate rank {actual} does not match tensor rank {expected}"
            ),
            Self::IndexOutOfBounds {
                axis,
                index,
                dimension,
            } => write!(
                formatter,
                "index {index} is out of bounds for axis {axis} with size {dimension}"
            ),
        }
    }
}

impl Error for TensorError {}

/// Takes a tensor shape, computes its contiguous row-major strides and element count, and checks
/// that every required product fits in usize.
pub fn checked_row_major_layout(shape: &[usize]) -> Result<(Vec<usize>, usize), TensorError> {
    if shape.is_empty() {
        return Ok((Vec::new(), 1));
    }

    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = shape[axis + 1]
            .checked_mul(strides[axis + 1])
            .ok_or(TensorError::ShapeOverflow)?;
    }

    let element_count = shape[0]
        .checked_mul(strides[0])
        .ok_or(TensorError::ShapeOverflow)?;
    Ok((strides, element_count))
}

/// Converts N-dimension coordinate into index of 1D buffer
pub fn checked_offset(
    shape: &[usize],
    strides: &[usize],
    base_offset: usize,
    coordinate: &[usize],
) -> Result<usize, TensorError> {
    if coordinate.len() != shape.len() {
        return Err(TensorError::RankMismatch {
            expected: shape.len(),
            actual: coordinate.len(),
        });
    }

    let mut offset = base_offset;
    for (axis, ((&axis_index, &dimension), &stride)) in
        coordinate.iter().zip(shape).zip(strides).enumerate()
    {
        // index at certain coordinate is greater than a shape size at the given axis
        if axis_index >= dimension {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: axis_index,
                dimension,
            });
        }

        let contribution = axis_index
            .checked_mul(stride)
            .ok_or(TensorError::ShapeOverflow)?;
        offset = offset
            .checked_add(contribution)
            .ok_or(TensorError::ShapeOverflow)?;
    }

    Ok(offset)
}

/// A tensor whose logical dimensions map onto one contiguous value buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    /// Builds a tensor after checking its row-major layout and buffer length.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, TensorError> {
        let (strides, expected) = checked_row_major_layout(&shape)?;
        let actual = data.len();

        if actual != expected {
            return Err(TensorError::DataLengthMismatch { expected, actual });
        }

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Returns the number of logical axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the extent of every axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of stored values.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the number of stored values.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Reports whether the flat buffer stores no values.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrows the contiguous value buffer.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutably borrows the contiguous value buffer without changing its length.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the tensor and returns its contiguous value buffer.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Maps one in-bounds coordinate to its row-major flat-buffer offset.
    pub fn offset(&self, coordinate: &[usize]) -> Result<usize, TensorError> {
        checked_offset(&self.shape, &self.strides, 0, coordinate)
    }

    /// Borrows the value at one checked coordinate.
    pub fn get(&self, coordinate: &[usize]) -> Result<&f64, TensorError> {
        let offset = self.offset(coordinate)?;
        Ok(&self.data[offset])
    }

    /// Mutably borrows the value at one checked coordinate.
    pub fn get_mut(&mut self, coordinate: &[usize]) -> Result<&mut f64, TensorError> {
        let offset = self.offset(coordinate)?;
        Ok(&mut self.data[offset])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fn_checked_row_major_layout {
        use super::*;

        mod when_some_of_axis_steps_is_out_of_bounds {
            use super::*;

            #[test]
            fn it_returns_error() {
                let res = checked_row_major_layout(&[2, 2, usize::MAX]);

                assert_eq!(res, Err(TensorError::ShapeOverflow));
            }
        }

        mod when_total_number_of_elements_is_out_of_bounds {
            use super::*;

            #[test]
            fn it_returns_error() {
                let res = checked_row_major_layout(&[usize::MAX, 2, 3]);

                assert_eq!(res, Err(TensorError::ShapeOverflow));
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_calculates_axis_steps_and_total_elements_number() {
                let res = checked_row_major_layout(&[2, 2, 3]);

                assert_eq!(res, Ok((vec![6, 3, 1], 12)));
            }
        }
    }

    mod fn_checked_offset {
        use super::*;

        mod when_coordinate_size_does_not_match_shape_size {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape = [2, 2, 3];
                let strides = [6, 3, 1];
                let base_offset = 0;
                let coordinate = [1, 2];
                let res = checked_offset(&shape, &strides, base_offset, &coordinate);

                assert_eq!(
                    res,
                    Err(TensorError::RankMismatch {
                        expected: shape.len(),
                        actual: coordinate.len()
                    })
                );
            }
        }

        mod when_axis_index_is_greater_than_shape_size_at_the_same_axis {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape = [1, 2, 3];
                let strides = [6, 3, 1];
                let base_offset = 0;
                let coordinate = [0, 2, 1];
                let res = checked_offset(&shape, &strides, base_offset, &coordinate);

                assert_eq!(
                    res,
                    Err(TensorError::IndexOutOfBounds {
                        axis: 1,
                        index: 2,
                        dimension: 2
                    })
                );
            }
        }

        mod when_offset_is_out_of_usize_bounds {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape = [usize::MAX, 2, 3];
                let strides = [6, 3, 1];
                let base_offset = 0;
                let coordinate = [usize::MAX - 1, 1, 1];
                let res = checked_offset(&shape, &strides, base_offset, &coordinate);

                assert_eq!(res, Err(TensorError::ShapeOverflow));
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_calculates_offset_in_1d_buffer() {
                let shape = [1, 2, 3];
                let strides = [6, 3, 1];
                let base_offset = 0;
                let coordinate = [0, 1, 1];
                let res = checked_offset(&shape, &strides, base_offset, &coordinate);

                assert_eq!(res, Ok(4));
            }
        }
    }

    mod tensor {
        use super::*;

        mod fn_from_vec {
            use super::*;

            mod when_data_length_from_the_shape_does_not_match_the_given_data_length {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let shape = vec![2, 2, 3];
                    let data = vec![0.5; 13];
                    let result = Tensor::from_vec(shape, data);

                    assert_eq!(
                        result,
                        Err(TensorError::DataLengthMismatch {
                            expected: 12,
                            actual: 13
                        })
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_tensor() {
                    let shape = vec![2, 2, 3];
                    let data = vec![0.5; 12];
                    let result = Tensor::from_vec(shape.clone(), data.clone());

                    assert_eq!(
                        result,
                        Ok(Tensor {
                            data,
                            shape,
                            strides: vec![6, 3, 1]
                        })
                    );
                }
            }
        }

        mod fn_offset {
            use super::*;

            #[test]
            fn it_calculates_offset_by_the_given_coordinate() {
                let shape = vec![2, 2, 3];
                let data = vec![0.5; 12];
                let tensor = Tensor::from_vec(shape, data).unwrap();
                let result = tensor.offset(&[1, 1, 2]);

                assert_eq!(result, Ok(11))
            }
        }

        mod fn_get {
            use super::*;

            #[test]
            fn it_takes_the_value_by_the_given_coordinate() {
                let shape = vec![2, 2, 1];
                let data = vec![0.1, 0.2, 0.3, 0.4];
                let tensor = Tensor::from_vec(shape, data).unwrap();
                let result = tensor.get(&[1, 1, 0]);

                assert_eq!(result, Ok(&0.4))
            }
        }

        mod fn_get_mut {
            use super::*;

            #[test]
            fn it_takes_mutable_value_by_the_given_coordinate() {
                let shape = vec![2, 2, 1];
                let data = vec![0.1, 0.2, 0.3, 0.4];
                let mut tensor = Tensor::from_vec(shape, data).unwrap();
                let result = tensor.get_mut(&[1, 1, 0]);

                assert_eq!(result, Ok(&mut 0.4))
            }
        }
    }
}
