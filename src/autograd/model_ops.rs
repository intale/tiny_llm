//! Model-critical tensor operations and their local reverse-mode rules.

use super::tensor_core::{
    AutogradContext, TensorAutodiffError, TensorOperation, TensorValue, accumulate_unbroadcast,
};
use crate::nn::probability::{indexed_mean_nll_forward, log_softmax_forward};
use crate::tensor::matmul::{MatmulError, matmul, matmul_with_transpose};
use crate::tensor::ops::{map_binary, map_unary, sum_axis as tensor_sum_axis};
use crate::tensor::storage::{Tensor, checked_row_major_layout};
use crate::utils::canonical_zero;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;

/// A rejected model-specific shape, selector, or allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelOpError {
    GatherTableRank {
        rank: usize,
    },
    GatherIndexCountMismatch {
        expected: usize,
        actual: usize,
    },
    GatherIndexOutOfBounds {
        position: usize,
        index: usize,
        rows: usize,
    },
    CausalSoftmaxRank {
        rank: usize,
    },
    CausalSoftmaxNonSquare {
        queries: usize,
        keys: usize,
    },
    CausalSoftmaxEmptyTokens,
    OutputAllocationFailed {
        elements: usize,
    },
}

impl fmt::Display for ModelOpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GatherTableRank { rank } => {
                write!(
                    formatter,
                    "row gather needs a rank-two table, got rank {rank}"
                )
            }
            Self::GatherIndexCountMismatch { expected, actual } => write!(
                formatter,
                "gather index shape needs {expected} IDs, but received {actual}"
            ),
            Self::GatherIndexOutOfBounds {
                position,
                index,
                rows,
            } => write!(
                formatter,
                "gather ID {index} at flat position {position} is out of bounds for {rows} rows"
            ),
            Self::CausalSoftmaxRank { rank } => write!(
                formatter,
                "causal softmax needs at least rank-two [..., queries, keys] scores, got rank {rank}"
            ),
            Self::CausalSoftmaxNonSquare { queries, keys } => write!(
                formatter,
                "causal softmax needs a square query-key grid, got {queries} queries and {keys} keys"
            ),
            Self::CausalSoftmaxEmptyTokens => formatter.write_str(
                "causal softmax needs at least one token so every row has an allowed key",
            ),
            Self::OutputAllocationFailed { elements } => write!(
                formatter,
                "cannot allocate model-operation output for {elements} f64 values"
            ),
        }
    }
}

impl Error for ModelOpError {}

/// Immutable forward evidence used by one model-operation parent edge.
#[derive(Clone, PartialEq, Debug)]
pub enum ModelSavedContext {
    /// Persisted context needed to calculate the gradient with respect to the
    /// left operand of a matrix multiplication during backward traversal.
    ///
    /// The right operand is required to compute `upstream @ right^T`.
    /// The input and output shapes preserve the broadcast relationship so the
    /// resulting gradient can be reduced back to the original left-operand shape.
    MatmulLeft {
        right: Tensor,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Persisted context needed to calculate the gradient with respect to the
    /// right operand of a matrix multiplication during backward traversal.
    ///
    /// The left operand is required to compute `left^T @ upstream`.
    /// The input and output shapes preserve the broadcast relationship so the
    /// resulting gradient can be reduced back to the original right-operand shape.
    MatmulRight {
        left: Tensor,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Persisted context needed to calculate the gradient of a row-gather
    /// operation during backward traversal.
    ///
    /// The selected row indices determine where upstream rows are scattered
    /// back into the source table. Repeated indices accumulate their gradient
    /// contributions into the same source row.
    GatherRows {
        indices: Vec<usize>,
        index_shape: Vec<usize>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Persisted context needed to calculate the gradient of an `exp()`
    /// operation during backward traversal.
    ///
    /// The forward output is retained because `d exp(x) / dx = exp(x)`,
    /// allowing the VJP to reuse the already computed exponential values.
    Exp { output: Tensor },

    /// Persisted context needed to calculate the gradient of a `log()`
    /// operation during backward traversal.
    ///
    /// The forward input is retained because `d log(x) / dx = 1 / x`.
    Log { input: Tensor },

    /// Persisted context needed to calculate the gradient of a SiLU operation
    /// during backward traversal.
    ///
    /// Both the original input and its sigmoid are retained so the local
    /// derivative can be evaluated without recomputing the sigmoid.
    Silu { input: Tensor, sigmoid: Tensor },

    /// Persisted context needed to calculate the gradient of a log-softmax
    /// operation during backward traversal.
    ///
    /// The forward softmax probabilities are retained for the log-softmax VJP,
    /// while the class axis and input shape describe the reduction groups and
    /// validate the incoming gradient layout.
    LogSoftmax {
        probabilities: Tensor,
        axis: usize,
        input_shape: Vec<usize>,
    },

    /// Persisted context needed to calculate the gradient of a causal softmax
    /// operation during backward traversal.
    ///
    /// The forward probabilities are retained for the softmax VJP. The input
    /// shape and token count describe the square causal attention grids and
    /// preserve the restriction that only keys visible to each query
    /// participate in the gradient calculation.
    CausalSoftmax {
        probabilities: Tensor,
        input_shape: Vec<usize>,
        tokens: usize,
    },

    /// Persisted context needed to calculate the gradient of a rotary-pair
    /// operation during backward traversal.
    ///
    /// The cosine and sine tables are retained so the upstream gradient can be
    /// transformed by the inverse pair rotation. The input shape validates that
    /// the gradient has the same layout as the original operand.
    RotaryPairs {
        cosines: Tensor,
        sines: Tensor,
        input_shape: Vec<usize>,
    },

    /// Persisted context needed to calculate the gradient of indexed mean
    /// negative log-likelihood during backward traversal.
    ///
    /// The forward probabilities and target class indices are retained to form
    /// the per-logit gradient `probability - target_indicator`. The class axis,
    /// input shape, and group count describe how targets map onto independent
    /// class groups and how the final mean scaling is applied.
    IndexedMeanNll {
        probabilities: Tensor,
        targets: Vec<usize>,
        axis: usize,
        input_shape: Vec<usize>,
        groups: usize,
    },
}

/// Owned row-gather facts established before materialization begins
#[derive(Debug, Clone)]
pub struct RowGatherPlan {
    indices: Vec<usize>,
    index_shape: Vec<usize>,
    input_shape: [usize; 2],
    output_shape: Vec<usize>,
    output_len: usize,
}

impl RowGatherPlan {
    /// Seals indices whose shape, count, and bounds were established by a crate-owned caller for
    /// this exact rank-two table
    pub fn from_validated_indices(
        table: &Tensor,
        indices: Vec<usize>,
        index_shape: Vec<usize>,
    ) -> Result<Self, TensorAutodiffError> {
        let input_shape = [table.shape()[0], table.shape()[1]];
        let width = input_shape[1];
        let mut output_shape = index_shape.clone();
        output_shape
            .try_reserve_exact(1)
            .map_err(|_| ModelOpError::OutputAllocationFailed {
                elements: indices.len().saturating_mul(width),
            })?;
        output_shape.push(width);
        let (_, output_len) = checked_row_major_layout(&output_shape)?;
        Ok(Self {
            indices,
            index_shape,
            input_shape,
            output_shape,
            output_len,
        })
    }

    fn checked(
        table: &Tensor,
        indices: &[usize],
        index_shape: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        if table.rank() != 2 {
            return Err(ModelOpError::GatherTableRank { rank: table.rank() }.into());
        }

        let (_, expected) = checked_row_major_layout(index_shape)?;
        if indices.len() != expected {
            return Err(ModelOpError::GatherIndexCountMismatch {
                expected,
                actual: indices.len(),
            }
            .into());
        }

        let rows = table.shape()[0];
        for (position, &index) in indices.iter().enumerate() {
            if index >= rows {
                return Err(ModelOpError::GatherIndexOutOfBounds {
                    position,
                    index,
                    rows,
                }
                .into());
            }
        }

        Self::from_validated_indices(table, indices.to_vec(), index_shape.to_vec())
    }

    fn into_saved_context(self) -> ModelSavedContext {
        let Self {
            indices,
            index_shape,
            input_shape,
            output_shape,
            ..
        } = self;
        ModelSavedContext::GatherRows {
            indices,
            index_shape,
            input_shape: input_shape.to_vec(),
            output_shape,
        }
    }
}

impl TensorValue {
    /// Multiplies rank-two or batched tensors and records both matrix pullbacks
    pub fn matmul(&self, right: &Self) -> Result<Self, TensorAutodiffError> {
        self.matmul_with_context(AutogradContext::recording(), right)
    }

    /// Multiplies tensors under the caller's explicit graph-recording policy
    pub fn matmul_with_context(
        &self,
        context: AutogradContext,
        right: &Self,
    ) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(
            context,
            TensorOperation::MatMul,
            [self, right],
            |primals| {
                let left = primals[0];
                let right = primals[1];
                let value = matmul(&left.view(), &right.view())?;
                let output_shape = value.shape().to_vec();
                Ok((
                    value,
                    [
                        ModelSavedContext::MatmulLeft {
                            right: right.clone(),
                            input_shape: left.shape().to_vec(),
                            output_shape: output_shape.clone(),
                        },
                        ModelSavedContext::MatmulRight {
                            left: left.clone(),
                            input_shape: right.shape().to_vec(),
                            output_shape: output_shape.clone(),
                        },
                    ],
                ))
            },
        )
    }

    /// Selects rows from a rank-two table into `index_shape + [width]`
    ///
    /// IDs are integer selectors and are deliberately not tape operands
    pub fn gather_rows(
        &self,
        indices: &[usize],
        index_shape: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        self.gather_rows_with_context(AutogradContext::recording(), indices, index_shape)
    }

    /// Selects rows under the caller's explicit graph-recording policy
    pub fn gather_rows_with_context(
        &self,
        context: AutogradContext,
        indices: &[usize],
        index_shape: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        self.gather_rows_with_plan_and_context(context, |table| {
            RowGatherPlan::checked(table, indices, index_shape)
        })
    }

    /// Builds one row-gather plan under an explicit graph-recording policy
    pub fn gather_rows_with_plan_and_context(
        &self,
        context: AutogradContext,
        build_plan: impl FnOnce(&Tensor) -> Result<RowGatherPlan, TensorAutodiffError>,
    ) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(
            context,
            TensorOperation::GatherRows,
            [self],
            |primals| {
                let table = primals[0];
                let plan = build_plan(table)?;
                let value = gather_rows_forward(table, &plan)?;
                Ok((value, [plan.into_saved_context()]))
            },
        )
    }

    /// Applies the elementwise exponential and saves its output for reversal
    pub fn exp(&self) -> Result<Self, TensorAutodiffError> {
        self.exp_with_context(AutogradContext::recording())
    }

    /// Applies the elementwise exponential under an explicit recording policy
    pub fn exp_with_context(&self, context: AutogradContext) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(context, TensorOperation::Exp, [self], |primals| {
            let value = map_unary(&primals[0].view(), f64::exp)?;
            Ok((value.clone(), [ModelSavedContext::Exp { output: value }]))
        })
    }

    /// Applies the natural logarithm; zero and negative inputs are rejected by the tape's
    /// finite-forward invariant
    pub fn log(&self) -> Result<Self, TensorAutodiffError> {
        self.log_with_context(AutogradContext::recording())
    }

    /// Applies the natural logarithm under an explicit recording policy.
    pub fn log_with_context(&self, context: AutogradContext) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(context, TensorOperation::Log, [self], |primals| {
            let input = primals[0];
            let value = map_unary(&input.view(), f64::ln)?;

            Ok((
                value,
                [ModelSavedContext::Log {
                    input: input.clone(),
                }],
            ))
        })
    }

    /// Applies SiLU, `x * sigmoid(x)`, with a branchwise stable sigmoid. SiLU function behavior:
    ///      x << 0       x = 0       x >> 0
    ///      y -> 0-      y = 0       y ~ x
    pub fn silu(&self) -> Result<Self, TensorAutodiffError> {
        self.silu_with_context(AutogradContext::recording())
    }

    /// Applies SiLU under the caller's explicit graph-recording policy.
    /// x * sigmiod(x) graph https://chatgpt.com/s/w_6ab50b85289c819180e782fb1cac43f6
    pub fn silu_with_context(&self, context: AutogradContext) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(context, TensorOperation::Silu, [self], |primals| {
            let input = primals[0];
            let sigmoid = map_unary(&input.view(), stable_sigmoid)?;
            let value = map_binary(&input.view(), &sigmoid.view(), |x, probability| {
                canonical_zero(x * probability)
            })?;

            Ok((
                value,
                [ModelSavedContext::Silu {
                    input: input.clone(),
                    sigmoid,
                }],
            ))
        })
    }

    /// Applies stable log-softmax along one explicit class axis
    pub fn log_softmax(&self, axis: usize) -> Result<Self, TensorAutodiffError> {
        self.log_softmax_with_context(AutogradContext::recording(), axis)
    }

    /// Applies stable log-softmax under an explicit recording policy
    pub fn log_softmax_with_context(
        &self,
        context: AutogradContext,
        axis: usize,
    ) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(
            context,
            TensorOperation::LogSoftmax,
            [self],
            |primals| {
                let input = primals[0];
                let forward = log_softmax_forward(&input.view(), axis, true)?;
                let probabilities = forward
                    .probabilities
                    .expect("the autodiff log-softmax forward requests saved probabilities");

                Ok((
                    forward.value,
                    [ModelSavedContext::LogSoftmax {
                        probabilities,
                        axis,
                        input_shape: input.shape().to_vec(),
                    }],
                ))
            },
        )
    }

    /// Normalizes each square score row over its inclusive prefix of keys
    ///
    /// Future-key probabilities are exactly zero. The implementation applies the additive
    /// negative-infinity mask as a branch, so non-finite values never enter the operation tape
    pub fn causal_softmax(&self) -> Result<Self, TensorAutodiffError> {
        self.causal_softmax_with_context(AutogradContext::recording())
    }

    /// Applies causal softmax under an explicit graph-recording policy.
    pub fn causal_softmax_with_context(
        &self,
        context: AutogradContext,
    ) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(
            context,
            TensorOperation::CausalSoftmax,
            [self],
            |primals| {
                let input = primals[0];
                let probabilities = causal_softmax_forward(input)?;
                let tokens = input.shape()[input.rank() - 1];

                Ok((
                    probabilities.clone(),
                    [ModelSavedContext::CausalSoftmax {
                        probabilities,
                        input_shape: input.shape().to_vec(),
                        tokens,
                    }],
                ))
            },
        )
    }

    pub fn rotary_pairs(
        &self,
        cosines: &Tensor,
        sines: &Tensor,
    ) -> Result<Self, TensorAutodiffError> {
        self.rotary_pairs_with_context(AutogradContext::recording(), cosines, sines)
    }

    /// Rotates adjacent features pairs with one precomputed angle row per token
    ///
    /// Shape and position-range validation belongs to the rotary-embedding owner. Keeping this
    /// primitive crate-private prevents callers from constructing inconsistent sine and cosine
    /// tables.
    /// Rotates feature pairs under an explicit graph-recording policy
    pub fn rotary_pairs_with_context(
        &self,
        context: AutogradContext,
        cosines: &Tensor,
        sines: &Tensor,
    ) -> Result<Self, TensorAutodiffError> {
        let cosines = cosines.clone();
        let sines = sines.clone();
        Self::model_operation_with_context(
            context,
            TensorOperation::RotaryPairs,
            [self],
            move |primals| {
                let input = primals[0];
                let value = rotary_pairs_forward(input, &cosines, &sines, false)?;

                Ok((
                    value,
                    [ModelSavedContext::RotaryPairs {
                        cosines,
                        sines,
                        input_shape: input.shape().to_vec(),
                    }],
                ))
            },
        )
    }

    /// Computes one stable rank-zero mean NLL from flat group-major targets
    pub fn indexed_mean_nll(
        &self,
        axis: usize,
        targets: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        self.indexed_mean_nll_with_context(AutogradContext::recording(), axis, targets)
    }

    /// Computes indexed mean NLL under an explicit graph-recording policy
    pub fn indexed_mean_nll_with_context(
        &self,
        context: AutogradContext,
        axis: usize,
        targets: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        Self::model_operation_with_context(
            context,
            TensorOperation::IndexedMeanNll,
            [self],
            |primals| {
                let logits = primals[0];
                let forward = indexed_mean_nll_forward(&logits.view(), axis, targets, true)?;
                let probabilities = forward
                    .probabilities
                    .expect("the autodiff indexed-NLL forward requests saved probabilities");
                let value = Tensor::from_vec(Vec::new(), vec![forward.loss])?;
                Ok((
                    value,
                    [ModelSavedContext::IndexedMeanNll {
                        probabilities,
                        targets: targets.to_vec(),
                        axis,
                        input_shape: logits.shape().to_vec(),
                        groups: targets.len(),
                    }],
                ))
            },
        )
    }
}

fn stable_sigmoid(value: f64) -> f64 {
    let result = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        // Mathematically this is the same as the computation for positive value, but without
        // potential f64 overflow
        // 1 / (1 + e^(-x)) == e^x / (1 + e^x)
        // We simply multiplied the original equation by e^x/e^x. Otherwise f64 would overflow
        // for on relatively small x values. Example: x = -1000 would require to calculate e^1000
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    canonical_zero(result)
}

fn output_buffer(elements: usize) -> Result<Vec<f64>, TensorAutodiffError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| ModelOpError::OutputAllocationFailed { elements })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn zeros(shape: &[usize]) -> Result<Tensor, TensorAutodiffError> {
    let (_, elements) = checked_row_major_layout(shape)?;
    Tensor::from_vec(shape.to_vec(), output_buffer(elements)?).map_err(Into::into)
}

fn gather_rows_forward(
    table: &Tensor,
    plan: &RowGatherPlan,
) -> Result<Tensor, TensorAutodiffError> {
    debug_assert_eq!(table.shape(), plan.input_shape);

    let width = plan.input_shape[1];
    let mut values = output_buffer(plan.output_len)?;

    for (position, &index) in plan.indices.iter().enumerate() {
        let source = index * width;
        let destination = position * width;
        values[destination..destination + width]
            .copy_from_slice(&table.as_slice()[source..source + width]);
    }

    Tensor::from_vec(plan.output_shape.clone(), values).map_err(Into::into)
}

/// Applies a numerically stable causal softmax to the query-key grids stored
/// in the final two dimensions of `input`.
///
/// The input must have shape `[..., queries, keys]`. Every leading coordinate
/// identifies an independent attention-score grid. Causal softmax requires
/// each grid to be square, so `queries` must equal `keys`.
///
/// For each query row `q`, softmax normalization is restricted to keys
/// `0..=q`. This allows the query to attend only to itself and earlier keys.
/// All positions with `key > q` remain exactly zero.
///
/// Conceptually, for a grid with `queries = 4` and `keys = 4`, the output has
/// the following structure:
///
/// ```text
/// q = 0: [p00,   0,   0,   0]
/// q = 1: [p10, p11,   0,   0]
/// q = 2: [p20, p21, p22,   0]
/// q = 3: [p30, p31, p32, p33]
/// ```
///
/// Each allowed row prefix is independently normalized to sum to one. The
/// output has the same shape as the input.
///
/// Each allowed prefix is normalized using a maximum shift before
/// exponentiation, avoiding overflow for large positive scores. One maximum
/// contribution is represented explicitly as `1.0` in the denominator to
/// improve numerical stability.
///
/// Leading dimensions are treated as independent query-key grids and are
/// flattened into a single grid index during traversal.
///
/// Future keys are excluded from normalization by restricting each query row
/// to the inclusive prefix `0..=q`, rather than by materializing
/// negative-infinity mask values.
///
/// This lower-triangular structure enforces autoregressive causality. During
/// next-token training, the representation at query position `q` must depend
/// only on tokens at positions `0..=q`; allowing it to attend to keys at
/// positions `q + 1..` would expose future tokens that the model is supposed
/// to predict.
///
/// For example, when training on a sequence such as
///
/// ```text
/// The cat sat down
/// ```
///
/// the representation of `cat` may attend to `The` and `cat`, but it must not
/// attend to `sat` or `down` if it is later used to predict `sat`. Otherwise
/// the model could obtain information about the target token directly from
/// the future context.
///
/// Applying this restriction to every query position produces a
/// lower-triangular attention matrix:
///
/// ```text
///         keys
///         0   1   2   3
/// q = 0   ✓   ·   ·   ·
/// q = 1   ✓   ✓   ·   ·
/// q = 2   ✓   ✓   ✓   ·
/// q = 3   ✓   ✓   ✓   ✓
/// ```
///
/// This allows all sequence positions to be processed in parallel during
/// training while preserving the same causal constraint that exists during
/// autoregressive generation, where future tokens are not yet available.
fn causal_softmax_forward(input: &Tensor) -> Result<Tensor, TensorAutodiffError> {
    if input.rank() < 2 {
        return Err(ModelOpError::CausalSoftmaxRank { rank: input.rank() }.into());
    }
    let queries = input.shape()[input.rank() - 2];
    let keys = input.shape()[input.rank() - 1];
    if queries != keys {
        return Err(ModelOpError::CausalSoftmaxNonSquare { queries, keys }.into());
    }
    if queries == 0 {
        return Err(ModelOpError::CausalSoftmaxEmptyTokens.into());
    }

    let mut probabilities = zeros(input.shape())?;
    let grids = input.len() / (queries * keys);
    for grid in 0..grids {
        for query in 0..queries {
            let row_start = (grid * queries + query) * keys;
            let allowed = &input.as_slice()[row_start..=row_start + query];
            let maximum = allowed.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let mut exponential_tail = 0.0;
            let mut max_values_count: f64 = 0.;
            for &score in allowed {
                let shifted = score - maximum;
                if shifted == 0.0 {
                    max_values_count += 1.;
                } else {
                    exponential_tail += shifted.exp();
                }
            }

            let denominator = max_values_count + exponential_tail;
            for (key, &score) in allowed.iter().enumerate() {
                let probability = (score - maximum).exp() / denominator;
                probabilities.as_mut_slice()[row_start + key] = canonical_zero(probability);
            }
        }
    }
    Ok(probabilities)
}

/// Applies rotary positional encoding to adjacent feature pairs.
///
/// This function injects token-position information into the input tensor by
/// rotating every adjacent pair of feature values with a position- and
/// pair-specific angle. The rotation is defined by precomputed cosine and sine
/// tables.
///
/// For a feature pair `(x0, x1)` and an angle `φ`, the forward transformation is:
///
/// ```text
/// x0' = x0 * cos(φ) - x1 * sin(φ)
/// x1' = x0 * sin(φ) + x1 * cos(φ)
/// ```
///
/// Conceptually, this applies the two-dimensional rotation matrix
///
/// ```text
/// [ cos(φ)  -sin(φ) ]
/// [ sin(φ)   cos(φ) ]
/// ```
///
/// to every adjacent feature pair.
///
/// The purpose of this transformation is to encode token position directly into
/// query and key features used by self-attention. Queries and keys at different
/// sequence positions are rotated by different angles. When their dot product is
/// later computed, the absolute rotations combine into a dependence on the
/// relative position between the two tokens. This allows attention to distinguish
/// nearby and distant tokens without storing the position as a separate scalar
/// feature.
///
/// Different feature pairs use different rotation frequencies. As a result, the
/// relative token distance is represented by a collection of periodic signals at
/// different scales rather than by a single angle.
///
/// `input` is expected to have shape `[..., tokens, width]`, where `width` is
/// even. The final feature dimension is interpreted as `width / 2` adjacent
/// pairs:
///
/// ```text
/// [x0, x1] [x2, x3] [x4, x5] ...
/// ```
///
/// `cosines` and `sines` must both have shape `[tokens, width / 2]`. Each
/// `[token, pair]` entry contains the precomputed trigonometric coefficient for
/// the rotation applied to that feature pair at that token position.
///
/// When `inverse` is `false`, the forward rotation is applied. When `inverse`
/// is `true`, the transpose of the rotation matrix is applied:
///
/// ```text
/// x0' =  x0 * cos(φ) + x1 * sin(φ)
/// x1' = -x0 * sin(φ) + x1 * cos(φ)
/// ```
///
/// Since rotation matrices are orthogonal, this inverse rotation is also the
/// VJP required during backward propagation.
///
/// The returned tensor has the same shape as `input`.
fn rotary_pairs_forward(
    input: &Tensor,
    cosines: &Tensor,
    sines: &Tensor,
    inverse: bool,
) -> Result<Tensor, TensorAutodiffError> {
    debug_assert!(input.rank() >= 2);
    debug_assert_eq!(cosines.shape(), sines.shape());
    debug_assert_eq!(cosines.rank(), 2);

    let width = input.shape()[input.rank() - 1];
    let tokens = input.shape()[input.rank() - 2];
    let pairs = width / 2;
    debug_assert_eq!(cosines.shape(), [tokens, pairs]);

    let mut output = zeros(input.shape())?;
    if input.is_empty() {
        return Ok(output);
    }

    let rows = input.len() / width;
    for row in 0..rows {
        let token = row % tokens;
        for pair in 0..pairs {
            let feature = row * width + pair * 2;
            let table = token * pairs + pair;
            let left = input.as_slice()[feature];
            let right = input.as_slice()[feature + 1];

            let cosine = cosines.as_slice()[table];
            let sine = sines.as_slice()[table];
            let (rotate_left, rotated_right) = rotate(left, right, sine, cosine, inverse);
            output.as_mut_slice()[feature] = canonical_zero(rotate_left);
            output.as_mut_slice()[feature + 1] = canonical_zero(rotated_right);
        }
    }

    Ok(output)
}

fn rotate(left: f64, right: f64, sine: f64, cosine: f64, inverse: bool) -> (f64, f64) {
    if inverse {
        (left * cosine + right * sine, -left * sine + right * cosine)
    } else {
        (left * cosine - right * sine, left * sine + right * cosine)
    }
}

fn unbroadcast(upstream: &Tensor, input_shape: &[usize]) -> Result<Tensor, TensorAutodiffError> {
    let mut result = zeros(input_shape)?;
    accumulate_unbroadcast(upstream, &mut result);
    Ok(result)
}

/// Applies the model-specific vector-Jacobian product for one parent edge.
///
/// `upstream` is the adjoint of the operation output:
///
/// ```text
/// upstream = dL / dy
/// ```
///
/// where `y` is the result of the forward operation. `saved` contains the
/// forward values and structural information required to apply that
/// operation's local derivative with respect to one parent `x`.
///
/// The returned tensor is the gradient contribution for that parent:
///
/// ```text
/// dL / dx = J_f(x)^T * upstream
/// ```
///
/// where `y = f(x)` and `J_f(x)` is the Jacobian of the forward operation.
/// The Jacobian is never materialized explicitly; every match arm computes
/// its VJP directly from the saved forward context.
pub fn apply_model_vjp(
    upstream: &Tensor,
    saved: &ModelSavedContext,
) -> Result<Tensor, TensorAutodiffError> {
    match saved {
        ModelSavedContext::MatmulLeft {
            right,
            input_shape,
            output_shape,
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            // Forward:
            //
            //     Y = Left @ Right
            //
            // For the left operand, the matrix-product VJP is:
            //
            //     dL/dLeft = dL/dY @ Right^T
            //
            // The matmul may have broadcast batch dimensions, so this first produces the gradient
            // in the expanded output batch shape.
            let expanded = matmul_with_transpose(&upstream.view(), &right.view(), false, true)?;
            // If Left was broadcast during the forward matmul, several output batch positions
            // depended on the same Left element. Their gradient contributions therefore have to be
            // summed back into the original Left shape.
            unbroadcast(&expanded, input_shape)
        }
        ModelSavedContext::MatmulRight {
            left,
            input_shape,
            output_shape,
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);

            // Forward:
            //
            //     Y = Left @ Right
            //
            // For the right operand, the matrix-product VJP is:
            //
            //     dL/dRight = Left^T @ dL/dY
            //
            // As with the left operand, the result initially follows the broadcasted batch shape
            // of the forward output.
            let expanded = matmul_with_transpose(&left.view(), &upstream.view(), true, false)?;
            // Sum contributions across any batch axes along which Right was broadcast so that the
            // returned gradient exactly matches the original Right shape.
            unbroadcast(&expanded, input_shape)
        }
        ModelSavedContext::GatherRows {
            indices,
            input_shape,
            output_shape,
            ..
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);

            let width = input_shape[1];
            // Forward gathers selected table rows:
            //
            //     output[position, :] = table[indices[position], :]
            //
            // Therefore the VJP performs the opposite routing operation: every output-row gradient
            // is scattered back into the table row from which that output row originated.
            let mut table_gradient = zeros(input_shape)?;

            for (position, &index) in indices.iter().enumerate() {
                let source = position * width;
                let destination = index * width;
                for feature in 0..width {
                    // The same table row may have been gathered more than once. Every use is an
                    // independent path from the table element to the loss, so repeated selections
                    // must accumulate their gradient contributions rather than overwrite each
                    // other.
                    table_gradient.as_mut_slice()[destination + feature] +=
                        upstream.as_slice()[source + feature];
                }
            }
            Ok(table_gradient)
        }
        ModelSavedContext::Exp { output } => {
            // Forward:
            //
            //     y = exp(x)
            //
            // Since:
            //
            //     dy/dx = exp(x) = y
            //
            // and the forward output was saved, the chain rule gives:
            //
            //     dL/dx = dL/dy * y
            //
            // Reusing the saved output avoids recomputing exp(x).
            map_binary(&upstream.view(), &output.view(), |gradient, value| {
                gradient * value
            })
            .map_err(Into::into)
        }
        ModelSavedContext::Log { input } => {
            // Forward:
            //
            //     y = log(x)
            //
            // Its local derivative is:
            //
            //     dy/dx = 1 / x
            //
            // so the VJP is:
            //
            //     dL/dx = dL/dy * 1 / x
            //
            // The original input was saved because the derivative depends on x rather than on
            // log(x).
            map_binary(&upstream.view(), &input.view(), |gradient, value| {
                gradient / value
            })
            .map_err(Into::into)
        }
        ModelSavedContext::Silu { input, sigmoid } => {
            // Forward:
            //
            //     y = x * sigmoid(x)
            //
            // Let s = sigmoid(x). Using the product rule and
            //
            //     ds/dx = s * (1 - s)
            //
            // gives:
            //
            //     dy/dx
            //       = s + x * s * (1 - s)
            //       = s * (1 + x * (1 - s))
            //
            // Both x and s were saved during the forward pass so this local derivative can be
            // evaluated without recomputing sigmoid(x).
            let derivative = map_binary(&input.view(), &sigmoid.view(), |value, probability| {
                probability * (1.0 + value * (1.0 - probability))
            })?;
            map_binary(&upstream.view(), &derivative.view(), |gradient, local| {
                gradient * local
            })
            .map_err(Into::into)
        }
        ModelSavedContext::LogSoftmax {
            probabilities,
            axis,
            input_shape,
        } => {
            debug_assert_eq!(upstream.shape(), input_shape);

            // For:
            //
            //     y_i = log_softmax(x)_i
            //
            // the Jacobian entries are:
            //
            //     dy_i/dx_j = δ_ij - softmax(x)_j
            //
            // Applying its transpose to upstream g gives:
            //
            //     dL/dx_j = g_j - p_j * sum_i(g_i)
            //
            // where p is the softmax probability saved during the forward pass.
            //
            // First compute sum_i(g_i) independently for every class-axis group while retaining the
            // reduced axis for broadcasting.
            let row_sum = tensor_sum_axis(&upstream.view(), *axis, true)?;
            // Compute:
            //
            //     p_j * sum_i(g_i)
            //
            // for every class position.
            let correction = map_binary(
                &probabilities.view(),
                &row_sum.view(),
                |probability, sum| probability * sum,
            )?;
            // Finish:
            //
            //     dL/dx = upstream - correction
            map_binary(&upstream.view(), &correction.view(), |gradient, term| {
                gradient - term
            })
            .map_err(Into::into)
        }
        ModelSavedContext::CausalSoftmax {
            probabilities,
            input_shape,
            tokens,
        } => {
            debug_assert_eq!(upstream.shape(), input_shape);

            let tokens = *tokens;
            let mut result = zeros(input_shape)?;
            let grids = upstream.len() / (tokens * tokens);
            // Each query row contains a separate softmax, but causal masking restricts that softmax
            // to keys 0..=query. Future keys did not participate in the forward operation and
            // therefore receive zero gradient here as well.
            for grid in 0..grids {
                for query in 0..tokens {
                    let row_start = (grid * tokens + query) * tokens;
                    // For an ordinary softmax with probabilities p and upstream g, the VJP for one
                    // element is:
                    //
                    //     dL/dx_i = p_i * (g_i - sum_j(g_j * p_j))
                    //
                    // Compute the shared dot product:
                    //
                    //     sum_j(g_j * p_j)
                    //
                    // only over the causally visible prefix 0..=query.
                    let mut weighted_upstream = 0.0;
                    for key in 0..=query {
                        weighted_upstream += upstream.as_slice()[row_start + key]
                            * probabilities.as_slice()[row_start + key];
                    }
                    for key in 0..=query {
                        let offset = row_start + key;
                        // Apply the softmax VJP to each allowed key:
                        //
                        //     p_i * (g_i - <g, p>)
                        //
                        // Masked future positions are never visited, leaving their gradients at the
                        // zero value initialized above.
                        let gradient = probabilities.as_slice()[offset]
                            * (upstream.as_slice()[offset] - weighted_upstream);
                        result.as_mut_slice()[offset] = canonical_zero(gradient);
                    }
                }
            }
            Ok(result)
        }
        ModelSavedContext::RotaryPairs {
            cosines,
            sines,
            input_shape,
        } => {
            debug_assert_eq!(upstream.shape(), input_shape);

            // Forward RoPE applies a two-dimensional rotation to every adjacent feature pair:
            //
            //     y = R(φ) * x
            //
            // Its Jacobian with respect to x is therefore R(φ). The VJP uses the transpose:
            //
            //     dL/dx = R(φ)^T * dL/dy
            //
            // Rotation matrices are orthogonal, so:
            //
            //     R(φ)^T = R(φ)^-1 = R(-φ)
            //
            // `inverse = true` applies exactly this inverse/transposed rotation using the same
            // saved sine and cosine tables.
            rotary_pairs_forward(upstream, cosines, sines, true)
        }
        ModelSavedContext::IndexedMeanNll {
            probabilities,
            targets,
            axis,
            input_shape,
            groups,
        } => {
            // Upstream must be scalar
            debug_assert_eq!(upstream.shape(), &[] as &[usize]);
            debug_assert_eq!(probabilities.shape(), input_shape);
            debug_assert_eq!(targets.len(), *groups);
            // Every target must not be out of bounds of its axis
            debug_assert!(targets.iter().all(|&target| target < input_shape[*axis]));

            // For one target class t, negative log-likelihood has the familiar derivative with
            // respect to the logits:
            //
            //     dNLL/dlogit_i = p_i - 1[i = t]
            //
            // where p is the softmax probability saved during forward.
            //
            // Start from p for every class...
            let mut result = probabilities.clone();

            let mut group_shape = input_shape.to_vec();
            let mut group_strides = result.strides().to_vec();
            group_shape.remove(*axis);
            let class_stride = group_strides.remove(*axis);
            let group_offsets = result
                .view()
                .projected_offsets(&group_shape, &group_strides, *groups)
                .expect("a checked indexed-NLL VJP retains valid group offsets");

            for (group_offset, &target) in group_offsets.zip(targets) {
                let target_offset = target
                    .checked_mul(class_stride)
                    .and_then(|class_offset| group_offset.checked_add(class_offset))
                    .expect("a checked indexed-NLL target retains a valid storage offset");

                // ...and subtract one at the ground-truth class, producing:
                //
                //     p - one_hot(target)
                //
                // independently for every class-axis group.
                result.as_mut_slice()[target_offset] -= 1.0;
            }

            // The forward operation returns the mean NLL over all groups, so every per-group
            // derivative carries a factor 1 / groups.
            //
            // `upstream` is scalar because IndexedMeanNll itself returns a rank-zero tensor.
            // Multiplying by it applies the outer chain rule:
            //
            //     dL/dlogits
            //       = dL/d(mean_nll)
            //         * (p - one_hot(target)) / groups
            let scale = upstream.as_slice()[0] / (*groups as f64);
            for value in result.as_mut_slice() {
                *value *= scale;
                *value = canonical_zero(*value);
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::tensor_core::GraphRetention;
    use crate::support::gradcheck::sampled_tensor_gradient_check;

    fn backward_with_test_seed(output: &TensorValue) -> Tensor {
        let elements = output.value().len();
        let values = (1..=elements)
            .map(|index| index as f64 / elements as f64)
            .collect();
        let seed = Tensor::from_vec(output.shape(), values).unwrap();
        output
            .backward_with_seed(&seed.view(), GraphRetention::Retain)
            .unwrap();
        seed
    }

    fn assert_sampled_gradient(
        parameter: &TensorValue,
        parameter_value: &Tensor,
        seed: &Tensor,
        mut objective: impl FnMut(&Tensor) -> TensorValue,
    ) {
        let analytic = parameter.gradient_snapshot().unwrap();
        let mut checked_parameter = parameter_value.clone();
        let check = sampled_tensor_gradient_check(
            &mut checked_parameter,
            &analytic.view(),
            1.0e-5,
            1.0e-6,
            parameter_value.len(),
            |candidate| {
                let output = objective(candidate);
                output
                    .value()
                    .as_slice()
                    .iter()
                    .zip(seed.as_slice())
                    .map(|(value, seed)| value * seed)
                    .sum()
            },
        )
        .unwrap();

        assert!(check.passed, "{check:#?}");
        assert_eq!(check.checks.len(), parameter_value.len());
    }

    mod row_gather_plan {
        use super::*;

        mod fn_checked {
            use super::*;

            mod when_table_rank_is_not_2 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let table = Tensor::from_vec(vec![1], vec![1.]).unwrap();
                    let indices = [0];
                    let index_shape = [1];
                    let result = RowGatherPlan::checked(&table, &indices, &index_shape);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::GatherTableRank {
                            rank: table.rank()
                        })
                    );
                }
            }

            mod when_indexes_length_does_not_match_index_shape {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let table = Tensor::from_vec(vec![2, 3], vec![1.; 6]).unwrap();
                    let indices = [0, 1, 0, 1];
                    let index_shape = [3];
                    let result = RowGatherPlan::checked(&table, &indices, &index_shape);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::GatherIndexCountMismatch {
                            expected: 3,
                            actual: 4,
                        })
                    );
                }
            }

            mod when_any_index_gte_table_rows_number {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let table = Tensor::from_vec(vec![3, 2], vec![1.; 6]).unwrap();
                    let indices = [0, 4, 1, 0];
                    let index_shape = [4];
                    let result = RowGatherPlan::checked(&table, &indices, &index_shape);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::GatherIndexOutOfBounds {
                            position: 1,
                            index: 4,
                            rows: 3
                        })
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_row_gather_plan() {
                    let table = Tensor::from_vec(vec![3, 2], vec![1.; 6]).unwrap();
                    let indices = [0, 2, 1, 0];
                    let index_shape = [4];
                    let result = RowGatherPlan::checked(&table, &indices, &index_shape);

                    assert!(result.is_ok());
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.indices, indices.to_vec());
                    assert_eq!(result.index_shape, index_shape.to_vec());
                    assert_eq!(result.input_shape, [table.shape()[0], table.shape()[1]]);
                    assert_eq!(
                        result.output_shape,
                        vec![result.index_shape[0], table.shape()[1]]
                    );
                    assert_eq!(result.output_len, 8);
                }
            }
        }

        mod fn_from_validated_indices {
            use super::*;

            #[test]
            fn it_computes_row_gather_plan() {
                let table = Tensor::from_vec(vec![3, 2], vec![1.; 6]).unwrap();
                let indices = [0, 2, 1, 0];
                let index_shape = [2, 2];
                let result = RowGatherPlan::from_validated_indices(
                    &table,
                    indices.to_vec(),
                    index_shape.to_vec(),
                );

                assert!(result.is_ok());
                let result = result.as_ref().unwrap();
                assert_eq!(result.indices, indices.to_vec());
                assert_eq!(result.index_shape, index_shape.to_vec());
                assert_eq!(result.input_shape, [table.shape()[0], table.shape()[1]]);
                assert_eq!(
                    result.output_shape,
                    vec![
                        result.index_shape[0],
                        result.index_shape[1],
                        table.shape()[1]
                    ]
                );
                assert_eq!(result.output_len, 8);
            }
        }

        mod fn_into_saved_context {
            use super::*;

            #[test]
            fn it_computes_saved_context_from_the_plan() {
                let table = Tensor::from_vec(vec![3, 2], vec![1.; 6]).unwrap();
                let indices = [0, 2, 1, 0];
                let index_shape = [2, 2];
                let plan = RowGatherPlan::from_validated_indices(
                    &table,
                    indices.to_vec(),
                    index_shape.to_vec(),
                )
                .unwrap();
                let result = plan.clone().into_saved_context();

                match result {
                    ModelSavedContext::GatherRows {
                        indices: result_indices,
                        index_shape: result_index_shape,
                        input_shape: result_input_shape,
                        output_shape: result_output_shape,
                    } => {
                        assert_eq!(result_indices, indices.to_vec());
                        assert_eq!(result_index_shape, index_shape.to_vec());
                        assert_eq!(result_input_shape, table.shape().to_vec());
                        assert_eq!(result_output_shape, plan.output_shape);
                    }
                    _ => panic!("Unexpected result: {:?}!", result),
                }
            }
        }
    }

    mod tensor_value {
        use super::*;

        mod fn_matmul_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_multiplies_two_tensors() {
                let tensor1 = Tensor::from_vec(vec![2, 1], vec![0.1, 0.2]).unwrap();
                let tensor_value1 = TensorValue::parameter(tensor1.clone()).unwrap();
                let tensor2 = Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap();
                let tensor_value2 = TensorValue::parameter(tensor2.clone()).unwrap();

                let context = AutogradContext::default();
                let result = tensor_value1.matmul_with_context(context, &tensor_value2);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor =
                    Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.2, 0.4]).unwrap();
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::MatMul);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![
                        ParentEdgeTest {
                            parent: tensor_value1.clone(),
                            parent_value_revision: tensor_value1.value_revision(),
                            saved: TensorSavedContext::Model(ModelSavedContext::MatmulLeft {
                                right: tensor2.clone(),
                                input_shape: tensor1.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            })
                        },
                        ParentEdgeTest {
                            parent: tensor_value2.clone(),
                            parent_value_revision: tensor_value2.value_revision(),
                            saved: TensorSavedContext::Model(ModelSavedContext::MatmulRight {
                                left: tensor1.clone(),
                                input_shape: tensor2.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            })
                        },
                    ]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value1, &tensor1, &seed, |candidate| {
                    let candidate = TensorValue::parameter(candidate.clone()).unwrap();
                    let right = TensorValue::constant(tensor2.clone()).unwrap();
                    candidate.matmul_with_context(context, &right).unwrap()
                });
                assert_sampled_gradient(&tensor_value2, &tensor2, &seed, |candidate| {
                    let left = TensorValue::constant(tensor1.clone()).unwrap();
                    let candidate = TensorValue::parameter(candidate.clone()).unwrap();
                    left.matmul_with_context(context, &candidate).unwrap()
                });
            }
        }

        mod fn_gather_rows_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_grabs_selected_rows_from_the_tensor() {
                let tensor = Tensor::from_vec(vec![6, 1], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                let indices = [0, 0, 1, 4];
                let index_shape = [4];

                let context = AutogradContext::default();
                let result = tensor_value.gather_rows_with_context(context, &indices, &index_shape);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor = Tensor::from_vec(vec![4, 1], vec![1., 1., 2., 5.]).unwrap();
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::GatherRows);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::GatherRows {
                            indices: indices.to_vec(),
                            index_shape: index_shape.to_vec(),
                            input_shape: tensor_value.shape().to_vec(),
                            output_shape: expected_tensor.shape().to_vec(),
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .gather_rows_with_context(context, &indices, &index_shape)
                        .unwrap()
                });
            }
        }

        mod fn_exp_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_calculates_exponent() {
                let tensor = Tensor::from_vec(vec![2, 1], vec![0.1, 0.2]).unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let result = tensor_value.exp_with_context(context);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor =
                    Tensor::from_vec(vec![2, 1], vec![0.1_f64.exp(), 0.2_f64.exp()]).unwrap();
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::Exp);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::Exp {
                            output: expected_tensor.clone(),
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .exp_with_context(context)
                        .unwrap()
                });
            }
        }

        mod fn_log_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_calculates_ln() {
                let tensor = Tensor::from_vec(vec![2, 1], vec![0.1, 0.2]).unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let result = tensor_value.log_with_context(context);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor =
                    Tensor::from_vec(vec![2, 1], vec![0.1_f64.ln(), 0.2_f64.ln()]).unwrap();
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::Log);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::Log {
                            input: tensor.clone(),
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .log_with_context(context)
                        .unwrap()
                });
            }
        }

        mod fn_silu_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_calculates_silu() {
                // Define large and large-negative values to ensure the sigmoid function can handle
                // them properly
                let tensor =
                    Tensor::from_vec(vec![6, 1], vec![1000.0, 2000.0, 0.1, 0.2, -1000.0, -2000.])
                        .unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let result = tensor_value.silu_with_context(context);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor = Tensor::from_vec(
                    vec![6, 1],
                    vec![
                        1000.0 * stable_sigmoid(1000.0),
                        2000.0 * stable_sigmoid(2000.0),
                        0.1 * stable_sigmoid(0.1),
                        0.2 * stable_sigmoid(0.2),
                        -1000.0 * stable_sigmoid(-1000.0),
                        -2000.0 * stable_sigmoid(-2000.0),
                    ],
                )
                .unwrap();
                assert!(expected_tensor.as_slice().iter().all(|x| x.is_finite()));
                assert!(result.value().clone().as_slice().iter().all(|x| {
                    if x == &0.0 {
                        x.is_sign_positive()
                    } else {
                        true
                    }
                }));
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::Silu);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::Silu {
                            input: tensor.clone(),
                            sigmoid: Tensor::from_vec(
                                vec![6, 1],
                                vec![
                                    stable_sigmoid(1000.0),
                                    stable_sigmoid(2000.0),
                                    stable_sigmoid(0.1),
                                    stable_sigmoid(0.2),
                                    stable_sigmoid(-1000.0),
                                    stable_sigmoid(-2000.0),
                                ]
                            )
                            .unwrap(),
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .silu_with_context(context)
                        .unwrap()
                });
            }
        }

        mod fn_log_softmax_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_calculates_log_softmax_with() {
                let tensor = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let axis = 1;
                let result = tensor_value.log_softmax_with_context(context, axis);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let forward = log_softmax_forward(&tensor.view(), axis, true).unwrap();
                assert_eq!(result.value().clone(), forward.value.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::LogSoftmax);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::LogSoftmax {
                            probabilities: forward.probabilities.unwrap().clone(),
                            axis,
                            input_shape: tensor.shape().to_vec()
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .log_softmax_with_context(context, axis)
                        .unwrap()
                });
            }
        }

        mod fn_causal_softmax_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            mod when_all_is_ok {
                use super::*;

                #[rustfmt::skip::macros(vec)]
                #[test]
                fn it_calculates_log_softmax_with() {
                    let tensor = Tensor::from_vec(vec![1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value.causal_softmax_with_context(context);

                    assert!(result.is_ok(), "{result:?}");
                    let result = &result.as_ref().unwrap();
                    let probabilities = Tensor::from_vec(
                        vec![1, 2, 2],
                        vec![
                            1.0, 0.0,
                            0.2689414213699951, 0.7310585786300049
                        ],
                    )
                    .unwrap();
                    assert_eq!(result.value().clone(), probabilities.clone());
                    assert_eq!(result.value_revision(), 0);
                    assert_eq!(result.operation(), TensorOperation::CausalSoftmax);
                    assert_eq!(result.tracks_gradient(), true);
                    assert_eq!(
                        result.parents(),
                        vec![ParentEdgeTest {
                            parent: tensor_value.clone(),
                            parent_value_revision: tensor_value.value_revision(),
                            saved: TensorSavedContext::Model(ModelSavedContext::CausalSoftmax {
                                probabilities: probabilities.clone(),
                                input_shape: tensor.shape().to_vec(),
                                tokens: tensor.shape()[tensor.rank() - 1]
                            })
                        }, ]
                    );
                    assert_eq!(result.is_released(), false);
                    assert_eq!(result.gradient_snapshot(), None);

                    let seed = backward_with_test_seed(result);
                    assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                        TensorValue::parameter(candidate.clone())
                            .unwrap()
                            .causal_softmax_with_context(context)
                            .unwrap()
                    });
                }
            }

            mod when_tensor_rank_is_lt_2 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![4], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value.causal_softmax_with_context(context);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::CausalSoftmaxRank {
                            rank: tensor.rank()
                        })
                    );
                }
            }

            mod when_queries_shape_ne_keys_shape {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 1, 4], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value.causal_softmax_with_context(context);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::CausalSoftmaxNonSquare {
                            queries: 1,
                            keys: 4,
                        })
                    );
                }
            }

            mod when_queries_and_keys_shape_eq_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 0, 0], vec![]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value.causal_softmax_with_context(context);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        TensorAutodiffError::Model(ModelOpError::CausalSoftmaxEmptyTokens)
                    );
                }
            }
        }

        mod fn_rotary_pairs_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[rustfmt::skip::macros(vec)]
            #[test]
            fn it_rotates_tokens() {
                let sin_tensor = Tensor::from_vec(vec![2, 2], vec![10., 20., 30., 40.]).unwrap();
                let cos_tensor = Tensor::from_vec(vec![2, 2], vec![50., 60., 70., 80.]).unwrap();

                let tensor = Tensor::from_vec(
                    vec![2, 2, 4],
                    vec![
                        1., 2., 3., 4.,
                        5., 6., 7., 8.,

                        1.1, 2.1, 3.1, 4.1,
                        5.1, 6.1, 7.1, 8.1,
                    ],
                )
                .unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let result =
                    tensor_value.rotary_pairs_with_context(context, &cos_tensor, &sin_tensor);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let expected_tensor = Tensor::from_vec(
                    tensor.shape().to_vec(),
                    vec![
                        rotate(tensor.as_slice()[0], tensor.as_slice()[1], sin_tensor.as_slice()[0], cos_tensor.as_slice()[0], false).0,
                        rotate(tensor.as_slice()[0], tensor.as_slice()[1], sin_tensor.as_slice()[0], cos_tensor.as_slice()[0], false).1,

                        rotate(tensor.as_slice()[2], tensor.as_slice()[3], sin_tensor.as_slice()[1], cos_tensor.as_slice()[1], false).0,
                        rotate(tensor.as_slice()[2], tensor.as_slice()[3], sin_tensor.as_slice()[1], cos_tensor.as_slice()[1], false).1,

                        rotate(tensor.as_slice()[4], tensor.as_slice()[5], sin_tensor.as_slice()[2], cos_tensor.as_slice()[2], false).0,
                        rotate(tensor.as_slice()[4], tensor.as_slice()[5], sin_tensor.as_slice()[2], cos_tensor.as_slice()[2], false).1,

                        rotate(tensor.as_slice()[6], tensor.as_slice()[7], sin_tensor.as_slice()[3], cos_tensor.as_slice()[3], false).0,
                        rotate(tensor.as_slice()[6], tensor.as_slice()[7], sin_tensor.as_slice()[3], cos_tensor.as_slice()[3], false).1,

                        rotate(tensor.as_slice()[8], tensor.as_slice()[9], sin_tensor.as_slice()[0], cos_tensor.as_slice()[0], false).0,
                        rotate(tensor.as_slice()[8], tensor.as_slice()[9], sin_tensor.as_slice()[0], cos_tensor.as_slice()[0], false).1,

                        rotate(tensor.as_slice()[10], tensor.as_slice()[11], sin_tensor.as_slice()[1], cos_tensor.as_slice()[1], false).0,
                        rotate(tensor.as_slice()[10], tensor.as_slice()[11], sin_tensor.as_slice()[1], cos_tensor.as_slice()[1], false).1,

                        rotate(tensor.as_slice()[12], tensor.as_slice()[13], sin_tensor.as_slice()[2], cos_tensor.as_slice()[2], false).0,
                        rotate(tensor.as_slice()[12], tensor.as_slice()[13], sin_tensor.as_slice()[2], cos_tensor.as_slice()[2], false).1,

                        rotate(tensor.as_slice()[14], tensor.as_slice()[15], sin_tensor.as_slice()[3], cos_tensor.as_slice()[3], false).0,
                        rotate(tensor.as_slice()[14], tensor.as_slice()[15], sin_tensor.as_slice()[3], cos_tensor.as_slice()[3], false).1,
                    ]
                ).unwrap();
                assert_eq!(result.value().clone(), expected_tensor.clone());
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::RotaryPairs);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::RotaryPairs {
                            cosines: cos_tensor.clone(),
                            sines: sin_tensor.clone(),
                            input_shape: tensor.shape().to_vec()
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .rotary_pairs_with_context(context, &cos_tensor, &sin_tensor)
                        .unwrap()
                });
            }
        }

        mod fn_indexed_mean_nll_with_context {
            use super::*;
            use crate::autograd::tensor_core::{ParentEdgeTest, TensorSavedContext};

            #[test]
            fn it_computes_mean_nll_loss_by_the_given_axis() {
                let tensor = Tensor::from_vec(vec![3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                let context = AutogradContext::default();
                let axis = 1;
                let targets = [0, 1, 0];
                let result = tensor_value.indexed_mean_nll_with_context(context, axis, &targets);

                assert!(result.is_ok(), "{result:?}");
                let result = &result.as_ref().unwrap();
                let forward =
                    indexed_mean_nll_forward(&tensor.view(), axis, &targets, true).unwrap();
                assert_eq!(
                    result.value().clone(),
                    Tensor::from_vec(vec![], vec![forward.loss]).unwrap()
                );
                assert_eq!(result.value_revision(), 0);
                assert_eq!(result.operation(), TensorOperation::IndexedMeanNll);
                assert_eq!(result.tracks_gradient(), true);
                assert_eq!(
                    result.parents(),
                    vec![ParentEdgeTest {
                        parent: tensor_value.clone(),
                        parent_value_revision: tensor_value.value_revision(),
                        saved: TensorSavedContext::Model(ModelSavedContext::IndexedMeanNll {
                            probabilities: forward.probabilities.unwrap(),
                            targets: targets.to_vec(),
                            axis,
                            input_shape: tensor.shape().to_vec(),
                            groups: targets.len(),
                        })
                    },]
                );
                assert_eq!(result.is_released(), false);
                assert_eq!(result.gradient_snapshot(), None);

                let seed = backward_with_test_seed(result);
                assert_sampled_gradient(&tensor_value, &tensor, &seed, |candidate| {
                    TensorValue::parameter(candidate.clone())
                        .unwrap()
                        .indexed_mean_nll_with_context(context, axis, &targets)
                        .unwrap()
                });
            }
        }
    }

    mod fn_apply_model_vjp {
        use super::*;

        mod when_saved_is_matmul_left {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap();
                let right =
                    Tensor::from_vec(vec![3, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

                // left [1, 3]
                // right [3, 2]
                // output(upstream) => [1, 2]
                let input_shape = vec![1, 3]; // left
                let saved = ModelSavedContext::MatmulLeft {
                    right: right.clone(),
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    Tensor::from_vec(input_shape.clone(), vec![0.5, 1.1, 1.7]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_matmul_right {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![1, 2], vec![1., 2.]).unwrap();
                let left = Tensor::from_vec(vec![1, 3], vec![0.1, 0.2, 0.3]).unwrap();

                // left [1, 3]
                // right [3, 2]
                // output(upstream) => [1, 2]

                let input_shape = vec![3, 2]; // right
                let saved = ModelSavedContext::MatmulRight {
                    left: left.clone(),
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    Tensor::from_vec(input_shape.clone(), vec![0.1, 0.2, 0.2, 0.4, 0.3, 0.6])
                        .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_gather_rows {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![3, 1], vec![1.1, 2.5, 3.8]).unwrap();
                let indices = vec![0, 0, 2];

                let input_shape = vec![4, 1];
                let saved = ModelSavedContext::GatherRows {
                    indices: indices.clone(),
                    index_shape: vec![indices.len()],
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    Tensor::from_vec(input_shape.clone(), vec![1.1 + 2.5, 0.0, 3.8, 0.0]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_exp {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![3, 1], vec![1.1, 2.5, 3.8]).unwrap();
                let output = Tensor::from_vec(vec![3, 1], vec![2., 3., 4.]).unwrap();

                let saved = ModelSavedContext::Exp {
                    output: output.clone(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    output.shape().to_vec(),
                    vec![1.1 * 2.0, 2.5 * 3.0, 3.8 * 4.0],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_log {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![3, 1], vec![1.1, 2.5, 3.8]).unwrap();
                let input = Tensor::from_vec(vec![3, 1], vec![2., 3., 4.]).unwrap();

                let saved = ModelSavedContext::Log {
                    input: input.clone(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input.shape().to_vec(),
                    vec![1.1 / 2.0, 2.5 / 3.0, 3.8 / 4.0],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_silu {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![3, 1], vec![1.1, 2.5, 3.8]).unwrap();
                let input = Tensor::from_vec(vec![3, 1], vec![2., 3., 4.]).unwrap();
                let sigmoid = Tensor::from_vec(
                    vec![3, 1],
                    vec![stable_sigmoid(2.), stable_sigmoid(3.), stable_sigmoid(4.)],
                )
                .unwrap();

                let saved = ModelSavedContext::Silu {
                    input: input.clone(),
                    sigmoid: sigmoid.clone(),
                };
                let local_derivative = |val, p| p * (1.0 + val * (1.0 - p));
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input.shape().to_vec(),
                    vec![
                        1.1 * local_derivative(input.as_slice()[0], sigmoid.as_slice()[0]),
                        2.5 * local_derivative(input.as_slice()[1], sigmoid.as_slice()[1]),
                        3.8 * local_derivative(input.as_slice()[2], sigmoid.as_slice()[2]),
                    ],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_log_softmax {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
                let probabilities =
                    Tensor::from_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
                let input_shape = vec![2, 3];
                let axis = 1;

                let saved = ModelSavedContext::LogSoftmax {
                    probabilities: probabilities.clone(),
                    axis,
                    input_shape: input_shape.clone(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input_shape.clone(),
                    vec![
                        1.0 - 0.1 * 6.0,
                        2.0 - 0.2 * 6.0,
                        3.0 - 0.3 * 6.0,
                        4.0 - 0.4 * 15.0,
                        5.0 - 0.5 * 15.0,
                        6.0 - 0.6 * 15.0,
                    ],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_causal_softmax {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
                let probabilities = Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
                let input_shape = vec![2, 2];

                let saved = ModelSavedContext::CausalSoftmax {
                    probabilities: probabilities.clone(),
                    tokens: input_shape[input_shape.len() - 1],
                    input_shape: input_shape.clone(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input_shape.clone(),
                    vec![
                        0.1 * (1.0 - 1.0 * 0.1),
                        0.0,
                        0.3 * (3.0 - (3.0 * 0.3 + 4.0 * 0.4)),
                        0.4 * (4.0 - (3.0 * 0.3 + 4.0 * 0.4)),
                    ],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_rotary_pairs {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream =
                    Tensor::from_vec(vec![2, 4], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
                let sines = Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
                let cosines = Tensor::from_vec(vec![2, 2], vec![0.5, 0.6, 0.7, 0.8]).unwrap();
                let input_shape = vec![2, 4];

                let saved = ModelSavedContext::RotaryPairs {
                    sines: sines.clone(),
                    cosines: cosines.clone(),
                    input_shape: input_shape.clone(),
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    rotary_pairs_forward(&upstream, &cosines, &sines, true).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_indexed_mean_nll {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_model() {
                let upstream = Tensor::from_vec(vec![], vec![10.0]).unwrap();
                let probabilities = Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
                let targets = vec![0, 1];
                let axis = 1;
                let input_shape = vec![2, 2];
                let groups = targets.len();

                let saved = ModelSavedContext::IndexedMeanNll {
                    probabilities: probabilities.clone(),
                    targets: targets.clone(),
                    axis,
                    input_shape: input_shape.clone(),
                    groups,
                };
                let result = apply_model_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let scale = upstream.as_slice()[0] / groups as f64;
                let expected_tensor = Tensor::from_vec(
                    input_shape.clone(),
                    vec![
                        scale * (0.1 - 1.0),
                        scale * 0.2,
                        scale * 0.3,
                        scale * (0.4 - 1.0),
                    ],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }
    }
}
