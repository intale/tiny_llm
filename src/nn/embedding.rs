//! A trainable token table backed by differentiable row gathering.

use std::error::Error;
use std::fmt;

use crate::autograd::model_ops::RowGatherPlan;
use crate::autograd::tensor_core::{AutogradContext, TensorAutodiffError, TensorValue};
use crate::nn::init::{InitializationError, NamedParameter, SplitMix64};
use crate::tensor::storage::{TensorError, checked_row_major_layout};

/// A rejected embedding table, token layout, selector, or delegated operation.
#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddingError {
    Initialization(InitializationError),
    Autodiff(TensorAutodiffError),
    TableRank {
        rank: usize,
    },
    EmptyVocabulary,
    ZeroEmbeddingWidth,
    TokenShape(TensorError),
    TokenCountMismatch {
        expected: usize,
        actual: usize,
    },
    TokenIdOutOfBounds {
        position: usize,
        id: u32,
        vocabulary_size: usize,
    },
    IndexAllocationFailed {
        elements: usize,
    },
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization(error) => error.fmt(formatter),
            Self::Autodiff(error) => error.fmt(formatter),
            Self::TableRank { rank } => {
                write!(
                    formatter,
                    "embedding table must have rank two, got rank {rank}"
                )
            }
            Self::EmptyVocabulary => {
                formatter.write_str("embedding vocabulary must contain at least one row")
            }
            Self::ZeroEmbeddingWidth => {
                formatter.write_str("embedding width must be greater than zero")
            }
            Self::TokenShape(error) => write!(formatter, "invalid token-ID shape: {error}"),
            Self::TokenCountMismatch { expected, actual } => write!(
                formatter,
                "token-ID shape needs {expected} IDs, but received {actual}"
            ),
            Self::TokenIdOutOfBounds {
                position,
                id,
                vocabulary_size,
            } => write!(
                formatter,
                "token ID {id} at flat position {position} is out of bounds for vocabulary size {vocabulary_size}"
            ),
            Self::IndexAllocationFailed { elements } => write!(
                formatter,
                "could not reserve {elements} converted embedding indices"
            ),
        }
    }
}

impl Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Initialization(error) => Some(error),
            Self::Autodiff(error) => Some(error),
            Self::TokenShape(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InitializationError> for EmbeddingError {
    fn from(error: InitializationError) -> Self {
        Self::Initialization(error)
    }
}

impl From<TensorAutodiffError> for EmbeddingError {
    fn from(error: TensorAutodiffError) -> Self {
        Self::Autodiff(error)
    }
}

/// One named trainable `[vocabulary_size, embedding_width]` token table
#[derive(Debug, Clone)]
pub struct Embedding {
    table: NamedParameter,
    vocabulary_size: usize,
    embedding_width: usize,
}

impl Embedding {
    /// Initializes one table.
    ///
    /// The complete parameter name is used as supplied. Validation checks the name before the
    /// vocabulary and width, and every error preserves `rng`.
    pub fn new(
        parameter_name: impl Into<String>,
        vocabulary_size: usize,
        embedding_width: usize,
        rng: &mut SplitMix64,
    ) -> Result<Self, EmbeddingError> {
        let mut trial = rng.clone();
        let table = NamedParameter::xavier_uniform(
            parameter_name,
            vocabulary_size,
            embedding_width,
            &mut trial,
        )
        .map_err(|error| match error {
            InitializationError::ZeroFanIn => EmbeddingError::EmptyVocabulary,
            InitializationError::ZeroFanOut => EmbeddingError::ZeroEmbeddingWidth,
            other => EmbeddingError::Initialization(other),
        })?;
        let embedding = Self::from_parameter(table)?;
        *rng = trial;
        Ok(embedding)
    }

    /// Gives embedding semantics to an existing named trainable rank-two table
    pub fn from_parameter(table: NamedParameter) -> Result<Self, EmbeddingError> {
        let shape = table.tensor().shape();
        if shape.len() != 2 {
            return Err(EmbeddingError::TableRank { rank: shape.len() });
        }
        if shape[0] == 0 {
            return Err(EmbeddingError::EmptyVocabulary);
        }
        if shape[1] == 0 {
            return Err(EmbeddingError::ZeroEmbeddingWidth);
        }

        Ok(Self {
            table,
            vocabulary_size: shape[0],
            embedding_width: shape[1],
        })
    }

    /// Selects one table row per `u32` token ID and appends the feature axis
    ///
    /// Token IDs remain integer selectors rather than differentiable operands. After this boundary
    /// validates and converts them, the shared gather kernel consumes the sealed facts without
    /// scanning the selectors again
    pub fn forward(
        &self,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<TensorValue, EmbeddingError> {
        self.forward_with_context(AutogradContext::recording(), token_ids, token_shape)
    }

    /// Selects embedding rows under the caller's explicit recording policy
    pub fn forward_with_context(
        &self,
        context: AutogradContext,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<TensorValue, EmbeddingError> {
        let (_, expected) =
            checked_row_major_layout(token_shape).map_err(EmbeddingError::TokenShape)?;
        if token_ids.len() != expected {
            return Err(EmbeddingError::TokenCountMismatch {
                expected,
                actual: token_ids.len(),
            });
        }

        for (position, &id) in token_ids.iter().enumerate() {
            let valid = usize::try_from(id)
                .ok()
                .is_some_and(|index| index < self.vocabulary_size);
            if !valid {
                return Err(EmbeddingError::TokenIdOutOfBounds {
                    position,
                    id,
                    vocabulary_size: self.vocabulary_size,
                });
            }
        }

        let mut indices: Vec<usize> = Vec::new();
        indices
            .try_reserve_exact(expected)
            .map_err(|_| EmbeddingError::IndexAllocationFailed { elements: expected })?;
        indices.extend(
            token_ids
                .iter()
                .map(|&token_id| token_id as usize)
                .collect::<Vec<_>>(),
        );
        self.table
            .tensor()
            .gather_rows_with_plan_and_context(context, move |table| {
                RowGatherPlan::from_validated_indices(table, indices, token_shape.to_vec())
            })
            .map_err(EmbeddingError::Autodiff)
    }

    /// Returns the table with its stable external name and trainable leaf.
    pub fn table(&self) -> &NamedParameter {
        &self.table
    }

    /// Returns the one-element parameter slice without duplicating the leaf.
    pub fn parameters(&self) -> &[NamedParameter] {
        std::slice::from_ref(&self.table)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub fn embedding_width(&self) -> usize {
        self.embedding_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::storage::Tensor;

    const ZERO_SEED_NEXT_U64: u64 = 16294208416658607535;
    const ZERO_SEED_NEXT_U64_AFTER_2X1_TENSOR: u64 = 487617019471545679;

    mod embedding {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_error_raises {
                use super::*;

                #[test]
                fn it_does_not_change_rng_state() {
                    let param_name = "foo";
                    let vocabulary_size = 0;
                    let embedding_width = 2;
                    let mut rng = SplitMix64::from_seed(0);
                    let result =
                        Embedding::new(param_name, vocabulary_size, embedding_width, &mut rng);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_embedding_and_persists_new_rng_state() {
                    let param_name = "foo";
                    let vocabulary_size = 2;
                    let embedding_width = 1;
                    let mut rng = SplitMix64::from_seed(0);
                    let result =
                        Embedding::new(param_name, vocabulary_size, embedding_width, &mut rng);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.vocabulary_size, vocabulary_size);
                    assert_eq!(result.embedding_width, embedding_width);
                    assert_eq!(
                        result.table.tensor().value().clone(),
                        Tensor::from_vec(vec![2, 1], vec![1.0841666871598514, -0.1936680704336956])
                            .unwrap()
                    );
                    assert_eq!(rng.next_u64(), ZERO_SEED_NEXT_U64_AFTER_2X1_TENSOR);
                }
            }
        }

        mod fn_from_parameter {
            use super::*;

            mod when_shape_is_not_2 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 2, 2], vec![0.; 4]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let result = Embedding::from_parameter(param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), EmbeddingError::TableRank { rank: 3 });
                }
            }

            mod when_vocabulary_size_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![0, 2], vec![]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let result = Embedding::from_parameter(param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), EmbeddingError::EmptyVocabulary);
                }
            }

            mod when_embedding_width_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2, 0], vec![]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor).unwrap();
                    let result = Embedding::from_parameter(param);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), EmbeddingError::ZeroEmbeddingWidth);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_embedding() {
                    let tensor =
                        Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let result = Embedding::from_parameter(param);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.vocabulary_size, 2);
                    assert_eq!(result.embedding_width, 3);
                    assert_eq!(result.table.tensor().value().clone(), tensor.clone());
                }
            }
        }

        mod fn_forward_with_context {
            use super::*;

            mod when_token_ids_size_does_not_match_token_shape_inferred_size {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor =
                        Tensor::from_vec(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let embedding = Embedding::from_parameter(param).unwrap();
                    let context = AutogradContext::recording();
                    let token_ids = [0, 0, 2];
                    let token_shape = [2];
                    let result = embedding.forward_with_context(context, &token_ids, &token_shape);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        EmbeddingError::TokenCountMismatch {
                            expected: 2,
                            actual: 3
                        }
                    );
                }
            }

            mod when_some_token_is_out_of_bounds {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor =
                        Tensor::from_vec(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let embedding = Embedding::from_parameter(param).unwrap();
                    let context = AutogradContext::recording();
                    let token_ids = [0, 0, 3];
                    let token_shape = [3];
                    let result = embedding.forward_with_context(context, &token_ids, &token_shape);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        EmbeddingError::TokenIdOutOfBounds {
                            position: 2,
                            id: 3,
                            vocabulary_size: 3,
                        }
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;
                use crate::autograd::model_ops::ModelSavedContext;
                use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

                #[test]
                fn it_performs_gather_rows_forward_pass() {
                    let tensor =
                        Tensor::from_vec(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                    let param = NamedParameter::from_tensor("foo", tensor.clone()).unwrap();
                    let embedding = Embedding::from_parameter(param).unwrap();
                    let context = AutogradContext::recording();
                    let token_ids = [0, 0];
                    let token_shape = [2];
                    let result = embedding.forward_with_context(context, &token_ids, &token_shape);

                    assert!(result.is_ok(), "{result:?}");
                    let resulting_tensor_value = result.unwrap();
                    assert_eq!(
                        resulting_tensor_value.value().clone(),
                        Tensor::from_vec(vec![2, 2], vec![1., 2., 1., 2.]).unwrap()
                    );
                    assert_eq!(resulting_tensor_value.parents().len(), 1);

                    assert_eq!(
                        resulting_tensor_value.parents()[0].parent.value().clone(),
                        tensor.clone()
                    );
                    assert_eq!(
                        resulting_tensor_value.parents()[0].saved,
                        TensorSavedContext::Model(ModelSavedContext::GatherRows {
                            indices: vec![0, 0],
                            index_shape: vec![2],
                            input_shape: vec![3, 2],
                            output_shape: vec![2, 2],
                        })
                    );
                }
            }
        }
    }
}
