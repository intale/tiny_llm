use crate::autograd::model_ops::{ModelOpError, ModelSavedContext, apply_model_vjp};
use crate::nn::probability::ProbabilityError;
use crate::tensor::matmul::MatmulError;
use crate::tensor::ops::{
    TensorOpError, broadcast_shape, map_binary, mean_axis as tensor_mean_axis,
    sum_axis as tensor_sum_axis,
};
use crate::tensor::storage::{Tensor, TensorError};
use crate::tensor::view::{TensorView, TensorViewError};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::{BorrowMutError, Cell, Ref, RefCell, RefMut};
use std::error::Error;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutogradContext {
    records_graph: bool,
}

impl AutogradContext {
    /// Creates the default training context that records eligible parent edges.
    pub fn recording() -> Self {
        Self {
            records_graph: true,
        }
    }

    /// Creates an evaluation context that keeps forward checks but records no graph.
    pub fn no_grad() -> Self {
        Self {
            records_graph: false,
        }
    }

    /// Returns whether operations created with this context may retain parent edges.
    pub fn records_graph(self) -> bool {
        self.records_graph
    }
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self::recording()
    }
}

/// A deterministic rejection from tensor tape construction or reversal.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorAutodiffError {
    Tensor(TensorError),
    View(TensorViewError),
    Operation(TensorOpError),
    Matmul(MatmulError),
    Probability(ProbabilityError),
    Model(ModelOpError),
    BroadcastTargetMismatch {
        input: Vec<usize>,
        requested: Vec<usize>,
        inferred: Vec<usize>,
    },
    NonFiniteLeaf {
        operation: TensorOperation,
        index: usize,
        value: f64,
    },
    NonFiniteForward {
        operation: TensorOperation,
        index: usize,
        value: f64,
    },
    UntrackedOutput {
        operation: TensorOperation,
    },
    GraphReleased {
        operation: TensorOperation,
    },
    ReleasedOperand {
        operation: TensorOperation,
        operand: usize,
    },
    SeedShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NonScalarBackwardOutput {
        actual: Vec<usize>,
    },
    NonFiniteSeed {
        index: usize,
        value: f64,
    },
    StaleOperandValue {
        child: usize,
        parent: usize,
        operand: usize,
        recorded_revision: u64,
        current_revision: u64,
    },
    NonFiniteVjp {
        child: usize,
        parent: usize,
        operand: usize,
        index: usize,
        value: f64,
    },
    NonFinitePassAdjoint {
        node: usize,
        index: usize,
        previous: f64,
        contribution: f64,
    },
    NonFiniteAccumulatedGradient {
        node: usize,
        index: usize,
        stored: f64,
        pass_adjoint: f64,
    },
    GradientBorrowed,
    NotAParameter {
        operation: TensorOperation,
    },
}

impl fmt::Display for TensorAutodiffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(formatter),
            Self::View(error) => error.fmt(formatter),
            Self::Operation(error) => error.fmt(formatter),
            Self::Matmul(error) => error.fmt(formatter),
            Self::Probability(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::BroadcastTargetMismatch {
                input,
                requested,
                inferred,
            } => write!(
                formatter,
                "cannot broadcast shape {input:?} exactly to {requested:?}; broadcasting infers {inferred:?}"
            ),
            Self::NonFiniteLeaf {
                operation,
                index,
                value,
            } => write!(
                formatter,
                "{operation} tensor value at flat index {index} must be finite, got {value:?}"
            ),
            Self::NonFiniteForward {
                operation,
                index,
                value,
            } => write!(
                formatter,
                "{operation} produced non-finite value {value:?} at flat index {index}"
            ),
            Self::UntrackedOutput { operation } => write!(
                formatter,
                "cannot backpropagate from untracked {operation} output"
            ),
            Self::GraphReleased { operation } => {
                write!(
                    formatter,
                    "the {operation} operation tape has been released"
                )
            }
            Self::ReleasedOperand { operation, operand } => write!(
                formatter,
                "cannot build {operation}: operand {operand} reaches a released operation tape"
            ),
            Self::SeedShapeMismatch { expected, actual } => write!(
                formatter,
                "backward seed shape {actual:?} does not match output shape {expected:?}"
            ),
            Self::NonScalarBackwardOutput { actual } => write!(
                formatter,
                "backward() requires a rank-zero output with shape [], got {:?}; use backward_with_seed() for non-scalar outputs",
                actual
            ),
            Self::NonFiniteSeed { index, value } => write!(
                formatter,
                "backward seed at flat index {index} must be finite, got {value:?}"
            ),
            Self::StaleOperandValue {
                child,
                parent,
                operand,
                recorded_revision,
                current_revision,
            } => write!(
                formatter,
                "cannot backpropagate through operand {operand} from topology node {child} to {parent}: the forward pass recorded parent value revision {recorded_revision}, but its current revision is {current_revision}; run a new forward pass"
            ),
            Self::NonFiniteVjp {
                child,
                parent,
                operand,
                index,
                value,
            } => write!(
                formatter,
                "edge {operand} from topology node {child} to {parent} produced non-finite VJP value {value:?} at flat index {index}"
            ),
            Self::NonFinitePassAdjoint {
                node,
                index,
                previous,
                contribution,
            } => write!(
                formatter,
                "topology node {node} cannot accumulate pass-adjoint value {previous:?} plus {contribution:?} at flat index {index}"
            ),
            Self::NonFiniteAccumulatedGradient {
                node,
                index,
                stored,
                pass_adjoint,
            } => write!(
                formatter,
                "topology node {node} cannot accumulate stored gradient {stored:?} plus pass adjoint {pass_adjoint:?} at flat index {index}"
            ),
            Self::GradientBorrowed => formatter.write_str(
                "cannot mutate a parameter gradient while a read-only gradient borrow is active",
            ),
            Self::NotAParameter { operation } => {
                write!(
                    formatter,
                    "cannot clear a gradient on {operation}; only parameters store gradients"
                )
            }
        }
    }
}

impl Error for TensorAutodiffError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::View(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Matmul(error) => Some(error),
            Self::Probability(error) => Some(error),
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for TensorAutodiffError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<TensorViewError> for TensorAutodiffError {
    fn from(error: TensorViewError) -> Self {
        Self::View(error)
    }
}

impl From<TensorOpError> for TensorAutodiffError {
    fn from(error: TensorOpError) -> Self {
        Self::Operation(error)
    }
}

impl From<MatmulError> for TensorAutodiffError {
    fn from(error: MatmulError) -> Self {
        Self::Matmul(error)
    }
}

impl From<ProbabilityError> for TensorAutodiffError {
    fn from(error: ProbabilityError) -> Self {
        Self::Probability(error)
    }
}

impl From<ModelOpError> for TensorAutodiffError {
    fn from(error: ModelOpError) -> Self {
        Self::Model(error)
    }
}

/// The tensor operation represented by one tape node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOperation {
    Parameter,
    Constant,
    Detached,
    Add,
    Multiply,
    Reshape,
    Transpose,
    Broadcast,
    Sum,
    Mean,
    MatMul,
    GatherRows,
    Exp,
    Log,
    Silu,
    LogSoftmax,
    CausalSoftmax,
    RotaryPairs,
    IndexedMeanNll,
}

impl TensorOperation {
    /// A stable, locale-neutral name suitable for trace evidence.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::Constant => "constant",
            Self::Detached => "detached",
            Self::Add => "add",
            Self::Multiply => "mul",
            Self::Reshape => "reshape",
            Self::Transpose => "transpose",
            Self::Broadcast => "broadcast",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::MatMul => "matmul",
            Self::GatherRows => "gather_rows",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Silu => "silu",
            Self::LogSoftmax => "log_softmax",
            Self::CausalSoftmax => "causal_softmax",
            Self::RotaryPairs => "rotary_pairs",
            Self::IndexedMeanNll => "indexed_mean_nll",
        }
    }

    fn is_leaf(self) -> bool {
        matches!(self, Self::Parameter | Self::Constant | Self::Detached)
    }
}

impl fmt::Display for TensorOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One node in the deterministic parent-first topology of a backward pass
#[derive(Clone, Debug, PartialEq)]
pub struct TensorBackwardNode {
    pub topology_index: usize,
    pub operation: TensorOperation,
    pub shape: Vec<usize>,
    pub tracked: bool,
    pub parameter: bool,
    pub pass_adjoint: Option<Tensor>,
    pub accumulated_gradient: Option<Tensor>,
}

/// One ordered operand edge visited during tensor reverse traversal
#[derive(Clone, Debug, PartialEq)]
pub struct TensorBackwardEdge {
    pub reverse_index: usize,
    pub child: usize,
    pub parent: usize,
    pub operand: usize,
    pub saved: TensorSavedContext,
    pub upstream: Tensor,
    pub contribution: Tensor,
    pub parent_tracked: bool,
    pub parent_adjoint_before: Option<Tensor>,
    pub parent_adjoint_after: Option<Tensor>,
}

/// Whether a successful backward pass keeps or frees its operation tape
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphRetention {
    Retain,
    Release,
}

/// The trace returned by one explicitly observed, successfully committed pass
#[derive(Clone, Debug, PartialEq)]
pub struct TensorBackwardPass {
    pub seed: Tensor,
    pub retention: GraphRetention,
    pub nodes: Vec<TensorBackwardNode>,
    pub edges: Vec<TensorBackwardEdge>,
}

trait TensorBackwardObserver {
    type Output;

    fn observe_edge(
        &mut self,
        child: usize,
        parent: usize,
        operand: usize,
        saved: &TensorSavedContext,
        upstream: &Tensor,
        contribution: &Tensor,
        parent_tracked: bool,
        parent_adjoint_before: Option<&Tensor>,
        parent_adjoin_after: Option<&Tensor>,
    );

    fn finish(
        self,
        retention: GraphRetention,
        topology: &[TensorValue],
        pass_adjoints: &[Option<Tensor>],
        prospective_gradients: &[Option<Tensor>],
    ) -> Self::Output;
}

struct NoTensorBackwardTrace;

impl TensorBackwardObserver for NoTensorBackwardTrace {
    type Output = ();

    fn observe_edge(
        &mut self,
        child: usize,
        parent: usize,
        operand: usize,
        saved: &TensorSavedContext,
        upstream: &Tensor,
        contribution: &Tensor,
        parent_tracked: bool,
        parent_adjoint_before: Option<&Tensor>,
        parent_adjoin_after: Option<&Tensor>,
    ) {
    }

    fn finish(
        self,
        retention: GraphRetention,
        topology: &[TensorValue],
        pass_adjoints: &[Option<Tensor>],
        prospective_gradients: &[Option<Tensor>],
    ) -> Self::Output {
    }
}

#[derive(Default)]
struct RecordTensorBackwardTrace {
    edges: Vec<TensorBackwardEdge>,
}

impl TensorBackwardObserver for RecordTensorBackwardTrace {
    type Output = TensorBackwardPass;

    fn observe_edge(
        &mut self,
        child: usize,
        parent: usize,
        operand: usize,
        saved: &TensorSavedContext,
        upstream: &Tensor,
        contribution: &Tensor,
        parent_tracked: bool,
        parent_adjoint_before: Option<&Tensor>,
        parent_adjoin_after: Option<&Tensor>,
    ) {
        self.edges.push(TensorBackwardEdge {
            reverse_index: self.edges.len(),
            child,
            parent,
            operand,
            saved: saved.clone(),
            upstream: upstream.clone(),
            contribution: contribution.clone(),
            parent_tracked,
            parent_adjoint_before: parent_adjoint_before.cloned(),
            parent_adjoint_after: parent_adjoin_after.cloned(),
        })
    }

    fn finish(
        self,
        retention: GraphRetention,
        topology: &[TensorValue],
        pass_adjoints: &[Option<Tensor>],
        prospective_gradients: &[Option<Tensor>],
    ) -> Self::Output {
        let nodes = topology
            .iter()
            .enumerate()
            .map(|(topology_index, value)| TensorBackwardNode {
                topology_index,
                operation: value.operation(),
                shape: value.shape(),
                tracked: value.tracks_gradient(),
                parameter: value.is_parameter(),
                pass_adjoint: pass_adjoints[topology_index].clone(),
                accumulated_gradient: prospective_gradients[topology_index].clone(),
            })
            .collect();

        TensorBackwardPass {
            seed: pass_adjoints
                .last()
                .and_then(Clone::clone)
                .expect("a backward topology ends with its seeded output"),
            retention,
            nodes,
            edges: self.edges,
        }
    }
}

/// Accumulates an upstream gradient back into the shape of a broadcast operand.
///
/// `upstream` has the shape of the broadcast result, while `result` has the original operand shape.
/// Every upstream element is mapped to the corresponding element of `result`, and values that
/// originated from the same broadcasted operand element are summed together.
///
/// Missing leading axes and input axes of size `1` are represented with a destination stride of
/// zero. As a result, traversal along a broadcasted axis repeatedly targets the same destination
/// element, naturally accumulating the VJP contribution required to undo broadcasting.
///
/// For example, broadcasting an input of shape `[3]` to `[2, 3]` and receiving an upstream gradient
///
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// ```
///
/// accumulates into a result of shape `[3]` as
///
/// ```text
/// [a + d, b + e, c + f]
/// ```
///
/// `result` is expected to have been initialized before the call, typically with zeros. Existing
/// values are preserved and incremented rather than overwritten.
pub fn accumulate_unbroadcast(upstream: &Tensor, result: &mut Tensor) {
    let output_shape = upstream.shape();
    let input_shape = result.shape();
    debug_assert!(output_shape.len() >= input_shape.len());
    let padding = output_shape.len() - input_shape.len();

    let destination_strides = output_shape
        .iter()
        .enumerate()
        .map(|(output_axis, _)| {
            if output_axis < padding || input_shape[output_axis - padding] == 1 {
                0
            } else {
                result.strides()[output_axis - padding]
            }
        })
        .collect::<Vec<_>>();
    let destination_offsets = result
        .view()
        .projected_offsets(output_shape, &destination_strides, upstream.len())
        .expect("a checked broadcast VJP retains a valid destination traversal plan");

    for (&value, destination_offset) in upstream.as_slice().iter().zip(destination_offsets) {
        result.as_mut_slice()[destination_offset] += value;
    }
}

fn unbroadcast(upstream: &Tensor, input_shape: &[usize]) -> Result<Tensor, TensorAutodiffError> {
    let mut result = zeros(input_shape)?;
    accumulate_unbroadcast(upstream, &mut result);
    Ok(result)
}

fn first_nonfinite(tensor: &Tensor) -> Option<(usize, f64)> {
    tensor
        .as_slice()
        .iter()
        .enumerate()
        .find(|(_, value)| value.is_infinite())
        .and_then(|(index, val)| Some((index, *val)))
}

fn check_finite_leaf(
    tensor: &Tensor,
    operation: TensorOperation,
) -> Result<(), TensorAutodiffError> {
    if let Some((index, value)) = first_nonfinite(tensor) {
        Err(TensorAutodiffError::NonFiniteLeaf {
            operation,
            index,
            value,
        })
    } else {
        Ok(())
    }
}

fn check_finite_forward(
    tensor: &Tensor,
    operation: TensorOperation,
) -> Result<(), TensorAutodiffError> {
    if let Some((index, value)) = first_nonfinite(tensor) {
        Err(TensorAutodiffError::NonFiniteForward {
            operation,
            index,
            value,
        })
    } else {
        Ok(())
    }
}

/// Creates a Tensor of the given shape, filled with 0.0 values
fn zeros(shape: &[usize]) -> Result<Tensor, TensorAutodiffError> {
    let elements = shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(TensorError::ShapeOverflow)
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| TensorOpError::OutputAllocationFailed { elements })?;
    values.resize(elements, 0.0);
    Ok(Tensor::from_vec(shape.to_vec(), values)?)
}

fn ensure_operands_available(
    operation: TensorOperation,
    operands: &[&TensorValue],
) -> Result<(), TensorAutodiffError> {
    for (operand, value) in operands.iter().enumerate() {
        let topology = value.topology();
        match topology {
            Ok(_) => continue,
            Err(err) => match err {
                TensorAutodiffError::GraphReleased { .. } => {
                    return Err(TensorAutodiffError::ReleasedOperand { operation, operand });
                }
                _ => panic!("unhandled error: {:?}", err),
            },
        }
    }
    Ok(())
}

/// Returns the output axes that were expanded by broadcasting the input shape.
///
/// The returned axis indices are expressed in the coordinate system of
/// `output_shape`. An axis is considered broadcasted, and therefore must be
/// reduced when propagating an output-shaped adjoint back to the input shape,
/// when either:
///
/// - the axis is a leading dimension added because `output_shape` has a
///   greater rank than `input_shape`, or
/// - the aligned input dimension is `1` while the corresponding output
///   dimension is greater than `1`.
///
/// Dimensions that remain `1` in both shapes are not included because no
/// replication occurred along those axes.
///
/// For example, broadcasting shape `[2, 1, 3]` to `[4, 2, 5, 3]` returns
/// `[0, 2]`: axis `0` was introduced as leading rank padding, and axis `2`
/// expands the input dimension of size `1` to size `5`.
///
/// This function assumes that `output_shape` is a valid broadcast target for
/// `input_shape` and that its rank is at least as large as the input rank.
fn broadcast_reduced_axes(input_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    let padding = output_shape.len() - input_shape.len();

    (0..output_shape.len())
        .filter(|&axis| {
            axis < padding || (input_shape[axis - padding] == 1 && output_shape[axis] != 1)
        })
        .collect()
}

fn broadcast_context(input_shape: &[usize], output_shape: &[usize]) -> TensorSavedContext {
    TensorSavedContext::Broadcast {
        input_shape: input_shape.to_vec(),
        output_shape: output_shape.to_vec(),
        reduced_axes: broadcast_reduced_axes(input_shape, output_shape),
    }
}

fn multiply_context(
    input_shape: &[usize],
    output_shape: &[usize],
    other: Tensor,
) -> TensorSavedContext {
    TensorSavedContext::Multiply {
        other,
        input_shape: input_shape.to_vec(),
        output_shape: output_shape.to_vec(),
        reduced_axes: broadcast_reduced_axes(input_shape, output_shape),
    }
}

fn expand_reduction(
    upstream: &Tensor,
    input_shape: &[usize],
    axis: usize,
    keep_dim: bool,
    divisor: usize,
) -> Result<Tensor, TensorAutodiffError> {
    debug_assert!(divisor > 0);

    let divisor = divisor as f64;
    let mut result = zeros(input_shape)?;
    let mut source_strides = upstream.strides().to_vec();

    if keep_dim {
        source_strides[axis] = 0;
    } else {
        source_strides.insert(axis, 0);
    }

    let source_offsets = upstream
        .view()
        .projected_offsets(input_shape, &source_strides, result.len())
        .expect("a checked reduction VJP retains a valid source traversal plan");

    for (destination, source_offset) in result.as_mut_slice().iter_mut().zip(source_offsets) {
        *destination = upstream.as_slice()[source_offset] / divisor;
    }
    Ok(result)
}

fn apply_vjp(upstream: &Tensor, saved: &TensorSavedContext) -> Result<Tensor, TensorAutodiffError> {
    match saved {
        TensorSavedContext::Broadcast {
            input_shape,
            output_shape,
            ..
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            unbroadcast(upstream, input_shape)
        }
        TensorSavedContext::Multiply {
            other,
            input_shape,
            output_shape,
            ..
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            let product = map_binary(&upstream.view(), &other.view(), |a, b| a * b)?;
            unbroadcast(&product, input_shape)
        }
        TensorSavedContext::Reshape {
            input_shape,
            output_shape,
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            Ok(upstream.view().reshape(input_shape)?.materialize()?)
        }
        TensorSavedContext::Transpose {
            first_axis,
            second_axis,
            output_shape,
            ..
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            Ok(upstream
                .view()
                .transpose(*first_axis, *second_axis)?
                .materialize()?)
        }
        TensorSavedContext::Reduction {
            axis,
            keep_dim,
            divisor,
            input_shape,
            output_shape,
        } => {
            debug_assert_eq!(upstream.shape(), output_shape);
            expand_reduction(upstream, input_shape, *axis, *keep_dim, *divisor)
        }
        TensorSavedContext::Model(saved) => apply_model_vjp(upstream, saved),
    }
}

fn add_checked(
    left: &Tensor,
    right: &Tensor,
    nonfinite: impl Fn(usize, f64, f64) -> TensorAutodiffError,
) -> Result<Tensor, TensorAutodiffError> {
    debug_assert_eq!(left.shape(), right.shape());

    let mut values = Vec::new();
    values
        .try_reserve_exact(left.len())
        .map_err(|_| TensorOpError::OutputAllocationFailed {
            elements: left.len(),
        })?;

    for (index, (&left, &right)) in left.as_slice().iter().zip(right.as_slice()).enumerate() {
        let sum = left + right;
        if !sum.is_finite() {
            return Err(nonfinite(index, left, right));
        }
        values.push(sum);
    }
    Ok(Tensor::from_vec(left.shape().to_vec(), values)?)
}

/// Immutable forward facts used by one exact-shape vector-Jacobian product
#[derive(Clone, Debug, PartialEq)]
pub enum TensorSavedContext {
    /// Identity derivative followed by reduction back to the operand shape.
    Broadcast {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        reduced_axes: Vec<usize>,
    },
    /// The other operand's primal is needed before unbroadcasting.
    Multiply {
        other: Tensor,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        reduced_axes: Vec<usize>,
    },
    /// A reshape preserves flat row-major order in both directions.
    Reshape {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Swapping the same axes reverses a transpose.
    Transpose {
        first_axis: usize,
        second_axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Sum uses divisor one; mean uses the selected axis length.
    Reduction {
        axis: usize,
        keep_dim: bool,
        divisor: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Forward evidence for one model-critical local pullback.
    Model(ModelSavedContext),
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct ParentEdgeTest {
    pub parent: TensorValue,
    pub parent_value_revision: u64,
    pub saved: TensorSavedContext,
}

#[derive(Clone, Debug, PartialEq)]
struct ParentEdge {
    parent: TensorValue,
    parent_value_revision: u64,
    saved: TensorSavedContext,
}

impl ParentEdge {
    fn capture(parent: &TensorValue, saved: TensorSavedContext) -> Self {
        Self {
            parent: parent.clone(),
            parent_value_revision: parent.value_revision(),
            saved,
        }
    }

    #[cfg(test)]
    fn to_test_parent_edge(&self) -> ParentEdgeTest {
        ParentEdgeTest {
            parent: self.parent.clone(),
            parent_value_revision: self.parent_value_revision,
            saved: self.saved.clone(),
        }
    }
}

#[derive(Debug, PartialEq)]
struct NodeState {
    parents: Vec<ParentEdge>,
    parameter_gradient: Option<Tensor>,
    released: bool,
}

#[derive(Debug, PartialEq)]
struct Node {
    value: RefCell<Tensor>,
    value_revision: Cell<u64>,
    operation: TensorOperation,
    tracked: bool,
    state: RefCell<NodeState>,
}

/// Exclusive access to one live node primal prepared for an infallible commit
#[derive(Debug)]
pub struct TensorValueWriteGuard<'a> {
    value: RefMut<'a, Tensor>,
    revision: &'a Cell<u64>,
}

impl TensorValueWriteGuard<'_> {
    /// Writes one already-validated same-shape primal and advances its revision.
    pub fn commit(mut self, value: Tensor, next_revision: u64) {
        debug_assert_eq!(self.value.shape(), value.shape());
        debug_assert_eq!(self.revision.get().checked_add(1), Some(next_revision));
        *self.value = value;
        self.revision.set(next_revision);
    }
}

type NodeKey = *const Node;

/// One owned tensor value and its operation-level reverse-mode tape.
#[derive(Clone, PartialEq)]
pub struct TensorValue {
    node: Rc<Node>,
}

impl TensorValue {
    fn new_node(
        value: Tensor,
        operation: TensorOperation,
        parents: Vec<ParentEdge>,
        tracked: bool,
        parameter_gradient: Option<Tensor>,
    ) -> Self {
        Self {
            node: Rc::new(Node {
                value: RefCell::new(value),
                value_revision: Cell::new(0),
                operation,
                tracked,
                state: RefCell::new(NodeState {
                    parents,
                    parameter_gradient,
                    released: false,
                }),
            }),
        }
    }

    /// Creates a finite leaf parameter initialized with an exact-shape zero gradient.
    pub fn parameter(value: Tensor) -> Result<Self, TensorAutodiffError> {
        check_finite_leaf(&value, TensorOperation::Parameter)?;

        let gradient = zeros(value.shape())?;
        Ok(Self::new_node(
            value,
            TensorOperation::Parameter,
            Vec::new(),
            true,
            Some(gradient),
        ))
    }

    /// Creates a finite untracked tensor leaf
    pub fn constant(value: Tensor) -> Result<Self, TensorAutodiffError> {
        check_finite_leaf(&value, TensorOperation::Constant)?;

        Ok(Self::new_node(
            value,
            TensorOperation::Constant,
            Vec::new(),
            false,
            None,
        ))
    }

    pub fn tracks_gradient(&self) -> bool {
        self.node.tracked
    }

    pub fn operation(&self) -> TensorOperation {
        self.node.operation
    }

    /// Borrows the node-owned primal tensor.
    pub fn value(&self) -> Ref<'_, Tensor> {
        self.node.value.borrow()
    }

    /// Copies the node-owned primal into an independent tensor snapshot.
    pub fn value_snapshot(&self) -> Tensor {
        self.node.value.borrow().clone()
    }

    /// Returns the internal version of this node's primal value.
    pub fn value_revision(&self) -> u64 {
        self.node.value_revision.get()
    }

    /// Returns the revision that one successful primal commit would install.
    pub fn next_value_revision(&self) -> Option<u64> {
        self.value_revision().checked_add(1)
    }

    /// Requests exclusive primal access without panicking on an active reader
    pub fn try_value_write(&self) -> Result<TensorValueWriteGuard<'_>, BorrowMutError> {
        Ok(TensorValueWriteGuard {
            value: self.node.value.try_borrow_mut()?,
            revision: &self.node.value_revision,
        })
    }

    /// Copies the primal shape.
    pub fn shape(&self) -> Vec<usize> {
        self.node.value.borrow().shape().to_vec()
    }

    pub fn is_parameter(&self) -> bool {
        self.operation() == TensorOperation::Parameter
    }

    pub fn is_released(&self) -> bool {
        self.node.state.borrow().released
    }

    pub fn gradient(&self) -> Option<Ref<'_, Tensor>> {
        Ref::filter_map(self.node.state.borrow(), |state| {
            state.parameter_gradient.as_ref()
        })
        .ok()
    }

    /// Copies the accumulated parameter gradient into an independent snapshot.
    pub fn gradient_snapshot(&self) -> Option<Tensor> {
        self.gradient().as_deref().cloned()
    }

    /// Returns whether two handles refer to the same tape node.
    pub fn is_same_node(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }

    /// Copies the primal into a new untracked leaf and breaks all parent edges
    pub fn detach(&self) -> Self {
        Self::new_node(
            self.value_snapshot(),
            TensorOperation::Detached,
            Vec::new(),
            false,
            None,
        )
    }

    fn operation_node(
        context: AutogradContext,
        value: Tensor,
        operation: TensorOperation,
        mut parents: Vec<ParentEdge>,
    ) -> Result<Self, TensorAutodiffError> {
        check_finite_forward(&value, operation)?;

        let tracked =
            context.records_graph() && parents.iter().any(|edge| edge.parent.tracks_gradient());
        if !context.records_graph() {
            parents.clear();
        }

        Ok(Self::new_node(value, operation, parents, tracked, None))
    }

    /// Builds one checked model-operation node under an explicit recording policy
    pub fn model_operation_with_context<const N: usize>(
        context: AutogradContext,
        operation: TensorOperation,
        operands: [&Self; N],
        forward: impl FnOnce(
            [&Tensor; N],
        ) -> Result<(Tensor, [ModelSavedContext; N]), TensorAutodiffError>,
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(operation, &operands)?;

        let primals: [Ref<'_, Tensor>; N] = std::array::from_fn(|index| operands[index].value());
        let primal_refs: [&Tensor; N] = std::array::from_fn(|index| &*primals[index]);
        let (value, contexts) = forward(primal_refs)?;
        let parents = operands
            .into_iter()
            .zip(contexts)
            .map(|(parent, context)| {
                ParentEdge::capture(parent, TensorSavedContext::Model(context))
            })
            .collect();
        Self::operation_node(context, value, operation, parents)
    }

    fn key(&self) -> NodeKey {
        Rc::as_ptr(&self.node)
    }

    fn topology(&self) -> Result<Vec<Self>, TensorAutodiffError> {
        fn visit(
            node: &TensorValue,
            visited: &mut FxHashSet<NodeKey>,
            order: &mut Vec<TensorValue>,
        ) -> Result<(), TensorAutodiffError> {
            if !visited.insert(node.key()) {
                return Ok(());
            }
            let state = node.node.state.borrow();
            if state.released {
                return Err(TensorAutodiffError::GraphReleased {
                    operation: node.operation(),
                });
            }
            for edge in &state.parents {
                visit(&edge.parent, visited, order)?;
            }
            order.push(node.clone());
            Ok(())
        }

        let mut visited = FxHashSet::default();
        let mut order = Vec::new();
        visit(self, &mut visited, &mut order)?;
        Ok(order)
    }

    /// Checks the finite-value invariant of a parameter without creating a node
    pub fn validate_parameter_value(value: &Tensor) -> Result<(), TensorAutodiffError> {
        check_finite_leaf(value, TensorOperation::Parameter)
    }

    /// Adds two tensors using trailing-axis broadcasting
    pub fn add(&self, other: &Self) -> Result<Self, TensorAutodiffError> {
        self.add_with_context(AutogradContext::recording(), other)
    }

    /// Adds two tensors under the caller's explicit graph-recording policy
    pub fn add_with_context(
        &self,
        context: AutogradContext,
        other: &Self,
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(TensorOperation::Add, &[self, other])?;

        let left = self.value();
        let right = other.value();
        let value = map_binary(&left.view(), &right.view(), |a, b| a + b)?;
        let output_shape = value.shape().to_vec();
        let parents = vec![
            ParentEdge::capture(self, broadcast_context(left.shape(), &output_shape)),
            ParentEdge::capture(other, broadcast_context(right.shape(), &output_shape)),
        ];
        Self::operation_node(context, value, TensorOperation::Add, parents)
    }

    /// Multiplies two tensors and records one ordered edge per operand use
    pub fn mul(&self, other: &Self) -> Result<Self, TensorAutodiffError> {
        self.mul_with_context(AutogradContext::recording(), other)
    }

    /// Multiplies two tensors under the caller's explicit graph-recording policy
    pub fn mul_with_context(
        &self,
        context: AutogradContext,
        other: &Self,
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(TensorOperation::Multiply, &[self, other])?;

        let left = self.value();
        let right = other.value();
        let value = map_binary(&left.view(), &right.view(), |a, b| a * b)?;
        let output_shape = value.shape().to_vec();
        let parents = vec![
            ParentEdge::capture(
                self,
                multiply_context(left.shape(), &output_shape, right.clone()),
            ),
            ParentEdge::capture(
                other,
                multiply_context(right.shape(), &output_shape, left.clone()),
            ),
        ];
        Self::operation_node(context, value, TensorOperation::Multiply, parents)
    }

    /// Changes shape without changing row-major element order
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorAutodiffError> {
        self.reshape_with_context(AutogradContext::recording(), shape)
    }

    /// Changes shape under the caller's explicit graph-recording policy.
    pub fn reshape_with_context(
        &self,
        context: AutogradContext,
        shape: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(TensorOperation::Reshape, &[self])?;

        let input = self.value();
        let value = input.view().reshape(shape)?.materialize()?;
        let saved = TensorSavedContext::Reshape {
            input_shape: input.shape().to_vec(),
            output_shape: value.shape().to_vec(),
        };
        Self::operation_node(
            context,
            value,
            TensorOperation::Reshape,
            vec![ParentEdge::capture(self, saved)],
        )
    }

    /// Swaps two axes and materializes the logical result as owned storage
    pub fn transpose(
        &self,
        first_axis: usize,
        second_axis: usize,
    ) -> Result<Self, TensorAutodiffError> {
        self.transpose_with_context(AutogradContext::recording(), first_axis, second_axis)
    }

    /// Swaps two axes under the caller's explicit graph-recording policy.
    pub fn transpose_with_context(
        &self,
        context: AutogradContext,
        first_axis: usize,
        second_axis: usize,
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(TensorOperation::Transpose, &[self])?;

        let input = self.value();
        let value = input
            .view()
            .transpose(first_axis, second_axis)?
            .materialize()?;
        let saved = TensorSavedContext::Transpose {
            first_axis,
            second_axis,
            input_shape: input.shape().to_vec(),
            output_shape: value.shape().to_vec(),
        };
        Self::operation_node(
            context,
            value,
            TensorOperation::Transpose,
            vec![ParentEdge::capture(self, saved)],
        )
    }

    /// Broadcasts exactly to `shape`; the requested shape may only expand axes
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, TensorAutodiffError> {
        self.broadcast_to_with_context(AutogradContext::recording(), shape)
    }

    /// Broadcasts under the caller's explicit graph-recording policy.
    pub fn broadcast_to_with_context(
        &self,
        context: AutogradContext,
        shape: &[usize],
    ) -> Result<Self, TensorAutodiffError> {
        ensure_operands_available(TensorOperation::Broadcast, &[self])?;

        let input = self.value();
        let inferred = broadcast_shape(input.shape(), shape)?;
        if inferred != shape {
            return Err(TensorAutodiffError::BroadcastTargetMismatch {
                input: input.shape().to_vec(),
                requested: shape.to_vec(),
                inferred,
            });
        }
        let value = Tensor::from_vec(shape.to_vec(), input.as_slice().to_vec())?;
        Self::operation_node(
            context,
            value,
            TensorOperation::Broadcast,
            vec![ParentEdge::capture(
                self,
                broadcast_context(input.shape(), shape),
            )],
        )
    }

    /// Sums one axis and records how to expand its exact-shape VJP
    pub fn sum_axis(&self, axis: usize, keep_dim: bool) -> Result<Self, TensorAutodiffError> {
        self.sum_axis_with_context(AutogradContext::recording(), axis, keep_dim)
    }

    /// Sums one axis under the caller's explicit graph-recording policy.
    pub fn sum_axis_with_context(
        &self,
        context: AutogradContext,
        axis: usize,
        keep_dim: bool,
    ) -> Result<Self, TensorAutodiffError> {
        self.reduce_axis(context, axis, keep_dim, false)
    }

    /// Averages one non-empty axis and records the divisor for its VJP
    pub fn mean_axis(&self, axis: usize, keep_dim: bool) -> Result<Self, TensorAutodiffError> {
        self.mean_axis_with_context(AutogradContext::recording(), axis, keep_dim)
    }

    /// Averages one nonempty axis under the caller's explicit graph-recording policy.
    pub fn mean_axis_with_context(
        &self,
        context: AutogradContext,
        axis: usize,
        keep_dim: bool,
    ) -> Result<Self, TensorAutodiffError> {
        self.reduce_axis(context, axis, keep_dim, true)
    }

    fn reduce_axis(
        &self,
        context: AutogradContext,
        axis: usize,
        keep_dim: bool,
        mean: bool,
    ) -> Result<Self, TensorAutodiffError> {
        let operation = if mean {
            TensorOperation::Mean
        } else {
            TensorOperation::Sum
        };
        ensure_operands_available(operation, &[self])?;

        let input = self.value();
        let value = if mean {
            tensor_mean_axis(&input.view(), axis, keep_dim)?
        } else {
            tensor_sum_axis(&input.view(), axis, keep_dim)?
        };
        let divisor = if mean { input.shape()[axis] } else { 1 };
        let saved = TensorSavedContext::Reduction {
            axis,
            keep_dim,
            divisor,
            input_shape: input.shape().to_vec(),
            output_shape: value.shape().to_vec(),
        };
        Self::operation_node(
            context,
            value,
            operation,
            vec![ParentEdge::capture(self, saved)],
        )
    }

    /// Reverses a rank-zero output with an implicit scalar seed of one
    pub fn backward(&self) -> Result<(), TensorAutodiffError> {
        self.backward_scalar(NoTensorBackwardTrace)
    }

    /// Reverses a rank-zero output and records its node and edge evidence
    pub fn backward_with_trace(&self) -> Result<TensorBackwardPass, TensorAutodiffError> {
        self.backward_scalar(RecordTensorBackwardTrace::default())
    }

    fn backward_scalar<Observer: TensorBackwardObserver>(
        &self,
        observer: Observer,
    ) -> Result<Observer::Output, TensorAutodiffError> {
        if self.shape() != Vec::<usize>::new() {
            return Err(TensorAutodiffError::NonScalarBackwardOutput {
                actual: self.shape(),
            });
        }
        // We start with dL/dL = 1
        let seed = Tensor::from_vec(Vec::new(), vec![1.0])?;
        self.backward_with_observe(&seed.view(), GraphRetention::Retain, observer)
    }

    /// Runs a fresh exact-shape reverse pass without creating a trace record
    pub fn backward_with_seed(
        &self,
        seed: &TensorView<'_>,
        retention: GraphRetention,
    ) -> Result<(), TensorAutodiffError> {
        self.backward_with_observe(seed, retention, NoTensorBackwardTrace)
    }

    /// Runs a fresh exact-shape reverse pass and records its trace.
    pub fn backward_with_seed_and_trace(
        &self,
        seed: &TensorView<'_>,
        retention: GraphRetention,
    ) -> Result<TensorBackwardPass, TensorAutodiffError> {
        self.backward_with_observe(seed, retention, RecordTensorBackwardTrace::default())
    }

    /// Executes the reverse pass transactionally.
    ///
    /// All VJP results and per-node adjoint accumulations are computed in temporary pass-local
    /// storage first. Existing parameter gradients are not modified until the entire reverse
    /// traversal has completed successfully.
    ///
    /// Before committing any persistent state, the method verifies that:
    ///
    /// - every VJP contribution is finite,
    /// - every accumulated pass adjoint is finite, and
    /// - every prospective stored parameter gradient remains finite after adding the current pass
    ///   contribution to the gradient already stored on that parameter.
    ///
    /// If any of these checks fails, the method returns an error without committing partial
    /// results. Stored parameter gradients and graph edges therefore remain bit-identical to their
    /// pre-backward state.
    ///
    /// Only after all numerical checks succeed are the prospective gradients committed and, when
    /// graph release is requested, the retained graph edges released. This gives the backward pass
    /// all-or-nothing semantics: either the complete pass is committed, or the persistent autograd
    /// state is left unchanged.
    fn backward_with_observe<Observe: TensorBackwardObserver>(
        &self,
        seed: &TensorView<'_>,
        retention: GraphRetention,
        mut observer: Observe,
    ) -> Result<Observe::Output, TensorAutodiffError> {
        if self.is_released() {
            return Err(TensorAutodiffError::GraphReleased {
                operation: self.operation(),
            });
        }
        if !self.tracks_gradient() {
            return Err(TensorAutodiffError::UntrackedOutput {
                operation: self.operation(),
            });
        }

        let expected = self.shape();
        if seed.shape() != expected {
            return Err(TensorAutodiffError::SeedShapeMismatch {
                expected,
                actual: seed.shape().to_vec(),
            });
        }

        let seed = seed.materialize()?;
        if let Some((index, value)) = first_nonfinite(&seed) {
            return Err(TensorAutodiffError::NonFiniteSeed { index, value });
        }

        let topology = self.topology()?;
        let indices = topology
            .iter()
            .enumerate()
            .map(|(index, value)| (value.key(), index))
            .collect::<FxHashMap<_, _>>();

        for (child, value) in topology.iter().enumerate().rev() {
            let state = value.node.state.borrow();

            for (operand, edge) in state.parents.iter().enumerate() {
                let current_revision = edge.parent.value_revision();

                if edge.parent_value_revision != current_revision {
                    return Err(TensorAutodiffError::StaleOperandValue {
                        child,
                        parent: indices[&edge.parent.key()],
                        operand,
                        recorded_revision: edge.parent_value_revision,
                        current_revision,
                    });
                }
            }
        }

        let mut pass_adjoints = vec![None; topology.len()];
        pass_adjoints[topology.len() - 1] = Some(seed);

        for child in (0..topology.len()).rev() {
            let Some(upstream) = pass_adjoints[child].clone() else {
                continue;
            };
            let state = topology[child].node.state.borrow();

            for (operand, edge) in state.parents.iter().enumerate() {
                let parent = indices[&edge.parent.key()];
                let contribution = apply_vjp(&upstream, &edge.saved)?;
                if let Some((index, value)) = first_nonfinite(&contribution) {
                    return Err(TensorAutodiffError::NonFiniteVjp {
                        child,
                        parent,
                        operand,
                        index,
                        value,
                    });
                }

                let parent_tracked = edge.parent.tracks_gradient();
                if parent_tracked {
                    let previous = pass_adjoints[parent]
                        .clone()
                        .unwrap_or(zeros(edge.parent.shape().as_slice())?);
                    let next =
                        add_checked(&previous, &contribution, |index, previous, contribution| {
                            TensorAutodiffError::NonFinitePassAdjoint {
                                node: parent,
                                index,
                                previous,
                                contribution,
                            }
                        })?;
                    observer.observe_edge(
                        child,
                        parent,
                        operand,
                        &edge.saved,
                        &upstream,
                        &contribution,
                        true,
                        Some(&previous),
                        Some(&next),
                    );
                    pass_adjoints[parent] = Some(next);
                } else {
                    observer.observe_edge(
                        child,
                        parent,
                        operand,
                        &edge.saved,
                        &upstream,
                        &contribution,
                        false,
                        None,
                        None,
                    );
                }
            }
        }

        let mut prospective = vec![None; topology.len()];
        for (index, value) in topology.iter().enumerate() {
            let Some(stored) = value.gradient() else {
                continue;
            };
            let pass = pass_adjoints[index]
                .clone()
                .unwrap_or(zeros(value.shape().as_slice())?);
            prospective[index] = Some(add_checked(
                &stored,
                &pass,
                |element, stored, pass_adjoint| TensorAutodiffError::NonFiniteAccumulatedGradient {
                    node: index,
                    index: element,
                    stored,
                    pass_adjoint,
                },
            )?);
        }

        let observation = observer.finish(retention, &topology, &pass_adjoints, &prospective);

        let mut commits: Vec<(RefMut<'_, NodeState>, Tensor)> = Vec::new();
        for (value, gradient) in topology.iter().zip(prospective) {
            if let Some(gradient) = gradient {
                let state = value
                    .node
                    .state
                    .try_borrow_mut()
                    .map_err(|_| TensorAutodiffError::GradientBorrowed)?;
                commits.push((state, gradient));
            }
        }
        for (mut state, gradient) in commits {
            state.parameter_gradient = Some(gradient);
        }

        if retention == GraphRetention::Release {
            for value in &topology {
                if !value.operation().is_leaf() {
                    let mut state = value.node.state.borrow_mut();
                    state.parents.clear();
                    state.released = true;
                }
            }
        }
        Ok(observation)
    }

    /// Clears this parameter's accumulated gradient without changing its tape.
    pub fn zero_grad(&self) -> Result<(), TensorAutodiffError> {
        if !self.is_parameter() {
            return Err(TensorAutodiffError::NotAParameter {
                operation: self.operation(),
            });
        }

        let shape = self.shape();
        self.node
            .state
            .try_borrow_mut()
            .map_err(|_| TensorAutodiffError::GradientBorrowed)?
            .parameter_gradient = Some(zeros(&shape)?);
        Ok(())
    }

    #[cfg(test)]
    pub fn set_value_revision_for_test(&self, revision: u64) {
        self.node.value_revision.set(revision);
    }

    #[cfg(test)]
    pub fn parents(&self) -> Vec<ParentEdgeTest> {
        self.node
            .state
            .borrow()
            .parents
            .iter()
            .map(ParentEdge::to_test_parent_edge)
            .collect()
    }
}

impl fmt::Debug for TensorValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorValue")
            .field("shape", &self.shape())
            .field("operation", &self.operation())
            .field("tracks_gradient", &self.tracks_gradient())
            .field("gradient", &self.gradient())
            .field("released", &self.is_released())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fn_accumulate_unbroadcast {
        use super::*;

        #[test]
        fn it_accumulates_gradient_into_the_shape_of_a_broadcast_operand() {
            let output = Tensor::from_vec(vec![2, 3], vec![1.1, 2.2, 3.3, 5.5, 6.6, 7.7]).unwrap();
            let mut input = Tensor::from_vec(vec![3], vec![10., 20., 30.]).unwrap();

            accumulate_unbroadcast(&output, &mut input);
            assert_eq!(input.shape(), vec![3]);
            assert_eq!(input.get(&[0]).unwrap(), &(10. + 1.1 + 5.5));
            assert_eq!(input.get(&[1]).unwrap(), &(20. + 2.2 + 6.6));
            assert_eq!(input.get(&[2]).unwrap(), &(30. + 3.3 + 7.7));
        }
    }

    mod autograd_context {
        use super::*;

        mod fn_recording {
            use super::*;

            #[test]
            fn it_returns_recording_context() {
                assert_eq!(
                    AutogradContext::recording(),
                    AutogradContext {
                        records_graph: true
                    }
                )
            }
        }

        mod fn_no_grad {
            use super::*;

            #[test]
            fn it_returns_disabled_recording_context() {
                assert_eq!(
                    AutogradContext::no_grad(),
                    AutogradContext {
                        records_graph: false
                    }
                )
            }
        }

        mod fn_default {
            use super::*;

            #[test]
            fn it_returns_recording_context() {
                assert_eq!(
                    AutogradContext::default(),
                    AutogradContext {
                        records_graph: true
                    }
                )
            }
        }
    }

    mod fn_first_nonfinite {
        use super::*;

        mod when_there_is_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor =
                    Tensor::from_vec(vec![3], vec![1., f64::NEG_INFINITY, f64::INFINITY]).unwrap();

                assert_eq!(first_nonfinite(&tensor), Some((1, f64::NEG_INFINITY)));
            }
        }

        mod when_there_is_no_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor = Tensor::from_vec(vec![3], vec![1., 2., 3.]).unwrap();

                assert_eq!(first_nonfinite(&tensor), None);
            }
        }
    }

    mod fn_check_finite_leaf {
        use super::*;

        mod when_there_is_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor =
                    Tensor::from_vec(vec![3], vec![1., f64::NEG_INFINITY, f64::INFINITY]).unwrap();
                let operation = TensorOperation::Parameter;

                assert_eq!(
                    check_finite_leaf(&tensor, operation),
                    Err(TensorAutodiffError::NonFiniteLeaf {
                        operation,
                        index: 1,
                        value: f64::NEG_INFINITY,
                    })
                );
            }
        }

        mod when_there_is_no_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor = Tensor::from_vec(vec![3], vec![1., 2., 3.]).unwrap();
                let operation = TensorOperation::Parameter;

                assert_eq!(check_finite_leaf(&tensor, operation), Ok(()));
            }
        }
    }

    mod fn_check_finite_forward {
        use super::*;

        mod when_there_is_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor =
                    Tensor::from_vec(vec![3], vec![1., f64::NEG_INFINITY, f64::INFINITY]).unwrap();
                let operation = TensorOperation::Parameter;

                assert_eq!(
                    check_finite_forward(&tensor, operation),
                    Err(TensorAutodiffError::NonFiniteForward {
                        operation,
                        index: 1,
                        value: f64::NEG_INFINITY,
                    })
                );
            }
        }

        mod when_there_is_no_non_finite_values {
            use super::*;

            #[test]
            fn it_returns_first_non_finite_value() {
                let tensor = Tensor::from_vec(vec![3], vec![1., 2., 3.]).unwrap();
                let operation = TensorOperation::Parameter;

                assert_eq!(check_finite_forward(&tensor, operation), Ok(()));
            }
        }
    }

    mod fn_zeros {
        use super::*;

        mod when_number_of_elements_in_the_shape_overflows_usize {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape = [usize::MAX / 2, 3];
                let result = zeros(&shape);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err(),
                    Some(TensorAutodiffError::Tensor(TensorError::ShapeOverflow))
                );
            }
        }

        mod when_needed_memory_can_not_be_allocated {
            use super::*;

            #[test]
            fn it_returns_error() {
                let shape = [usize::MAX / 2];
                let result = zeros(&shape);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err(),
                    Some(TensorAutodiffError::Operation(
                        TensorOpError::OutputAllocationFailed {
                            elements: usize::MAX / 2
                        }
                    ))
                );
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_creates_a_tensor_filled_with_zeros() {
                let shape = [2, 3];
                let result = zeros(&shape);

                assert!(result.is_ok(), "{result:?}");
                assert_eq!(
                    result,
                    Ok(Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap())
                )
            }
        }
    }

    mod fn_ensure_operands_available {
        use super::*;

        mod when_one_of_operands_is_released {
            use super::*;

            #[test]
            fn it_returns_error() {
                let tensor1 = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                let tensor2 = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
                let operand1 = TensorValue::parameter(tensor1).unwrap();
                let operand2 = TensorValue::parameter(tensor2).unwrap();
                let operation = TensorOperation::Add;
                operand2.node.state.borrow_mut().released = true;
                let result = ensure_operands_available(operation, &[&operand1, &operand2]);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(
                    result.err(),
                    Some(TensorAutodiffError::ReleasedOperand {
                        operation,
                        operand: 1
                    })
                );
            }
        }

        mod when_none_of_operands_are_released {
            use super::*;

            #[test]
            fn it_returns_ok() {
                let tensor1 = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                let tensor2 = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
                let operand1 = TensorValue::parameter(tensor1).unwrap();
                let operand2 = TensorValue::parameter(tensor2).unwrap();
                let operation = TensorOperation::Add;
                let result = ensure_operands_available(operation, &[&operand1, &operand2]);

                assert_eq!(result, Ok(()));
            }
        }
    }

    mod fn_broadcast_reduced_axes {
        use super::*;

        mod when_shapes_are_equal {
            use super::*;

            #[test]
            fn it_returns_empty_array() {
                let input_shape = vec![2, 1, 3];
                let output_shape = input_shape.clone();

                assert_eq!(
                    broadcast_reduced_axes(&input_shape, &output_shape),
                    Vec::<usize>::new()
                );
            }
        }

        mod when_shapes_are_different {
            use super::*;

            #[test]
            fn it_computes_indexes_of_reduced_axes() {
                let input_shape = vec![2, 1, 3];
                let output_shape = vec![4, 2, 5, 3];

                assert_eq!(
                    broadcast_reduced_axes(&input_shape, &output_shape),
                    vec![0, 2]
                );
            }
        }
    }

    mod fn_broadcast_context {
        use super::*;

        #[test]
        fn it_computes_broadcast_tensor_saved_context() {
            let input_shape = vec![2, 1, 3];
            let output_shape = vec![4, 2, 5, 3];

            assert_eq!(
                broadcast_context(&input_shape, &output_shape),
                TensorSavedContext::Broadcast {
                    input_shape: input_shape.to_vec(),
                    output_shape: output_shape.to_vec(),
                    reduced_axes: vec![0, 2]
                }
            );
        }
    }

    mod fn_add_checked {
        use super::*;

        mod when_sum_overflows_f64 {
            use super::*;

            #[test]
            fn it_returns_error() {
                let left = Tensor::from_vec(vec![1, 2], vec![0., f64::MAX]).unwrap();
                let right = Tensor::from_vec(vec![1, 2], vec![0., f64::MAX]).unwrap();
                let nonfinite = |_, _, _| TensorAutodiffError::GradientBorrowed;
                let result = add_checked(&left, &right, nonfinite);

                assert!(result.is_err(), "{result:?}");
                assert_eq!(result.err().unwrap(), TensorAutodiffError::GradientBorrowed);
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_adds_to_tensors() {
                let left = Tensor::from_vec(vec![1, 2], vec![0., 1.]).unwrap();
                let right = Tensor::from_vec(vec![1, 2], vec![2., 3.]).unwrap();
                let nonfinite = |_, _, _| TensorAutodiffError::GradientBorrowed;
                let result = add_checked(&left, &right, nonfinite);

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(result, &Tensor::from_vec(vec![1, 2], vec![2., 4.]).unwrap());
            }
        }
    }

    mod fn_expand_reduction {
        use super::*;

        mod when_axis_was_removed {
            use super::*;

            #[test]
            fn it_restores_removed_axis() {
                let upstream = Tensor::from_vec(vec![2, 2], vec![0., 1., 2., 3.]).unwrap();
                let input_shape = [2, 3, 2];
                let axis = 1;
                let keep_dim = false;
                let divisor = 2;
                let result = expand_reduction(&upstream, &input_shape, axis, keep_dim, divisor);

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(
                    result,
                    &Tensor::from_vec(
                        input_shape.to_vec(),
                        vec![0., 1., 0., 1., 0., 1., 2., 3., 2., 3., 2., 3.]
                            .iter()
                            .map(|x| x / divisor as f64)
                            .collect()
                    )
                    .unwrap()
                );
            }
        }

        mod when_axis_was_kept {
            use super::*;

            #[test]
            fn it_restores_reduced_axis() {
                let upstream = Tensor::from_vec(vec![2, 1, 2], vec![0., 1., 2., 3.]).unwrap();
                let input_shape = [2, 3, 2];
                let axis = 1;
                let keep_dim = true;
                let divisor = 2;
                let result = expand_reduction(&upstream, &input_shape, axis, keep_dim, divisor);

                assert!(result.is_ok(), "{result:?}");
                let result = result.as_ref().unwrap();
                assert_eq!(
                    result,
                    &Tensor::from_vec(
                        input_shape.to_vec(),
                        vec![0., 1., 0., 1., 0., 1., 2., 3., 2., 3., 2., 3.]
                            .iter()
                            .map(|x| x / divisor as f64)
                            .collect()
                    )
                    .unwrap()
                );
            }
        }
    }

    mod fn_apply_vjp {
        use super::*;

        mod when_saved_is_broadcast {
            use super::*;

            #[test]
            fn it_unbroadcasts_values_into_operand_shape() {
                let upstream =
                    Tensor::from_vec(vec![2, 3], vec![1.1, 2.2, 3.3, 4.0, 5.0, 6.0]).unwrap();
                let output_shape = upstream.shape().to_vec();
                let input_shape = vec![3];
                let saved = TensorSavedContext::Broadcast {
                    input_shape: input_shape.clone(),
                    output_shape: output_shape.clone(),
                    reduced_axes: vec![0],
                };
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    Tensor::from_vec(input_shape.clone(), vec![5.1, 7.2, 9.3]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_multiply {
            use super::*;

            #[test]
            fn it_calculates_gradient_contribution_of_parent_edge_by_its_saved_context() {
                let upstream = Tensor::from_vec(vec![2, 2], vec![0.5, 1.5, 1.5, 3.0]).unwrap();
                let right = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();

                // left [2]
                // right [2, 2]
                // output(upstream) => [2, 2]
                let input_shape = vec![2]; // left
                let saved = TensorSavedContext::Multiply {
                    other: right.clone(),
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                    reduced_axes: vec![],
                };
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input_shape.clone(),
                    vec![0.5 * 1.0 + 1.5 * 3.0, 1.5 * 2.0 + 3.0 * 4.0],
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_reshape {
            use super::*;

            #[test]
            fn it_reshapes_tensor_into_original_shape() {
                let upstream = Tensor::from_vec(vec![2, 3], vec![0.; 6]).unwrap();
                let input_shape = vec![6, 1];
                let saved = TensorSavedContext::Reshape {
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                };
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(input_shape.clone(), vec![0.; 6]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_transpose {
            use super::*;

            #[test]
            fn it_swaps_two_axes_back_into_original_shape() {
                let upstream = Tensor::from_vec(vec![2, 3, 1], vec![0.; 6]).unwrap();
                let input_shape = vec![2, 1, 3];
                let saved = TensorSavedContext::Transpose {
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                    first_axis: 1,
                    second_axis: 2,
                };
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(input_shape.clone(), vec![0.; 6]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_reduction {
            use super::*;

            #[test]
            fn it_expands_reduced_axis() {
                let upstream = Tensor::from_vec(vec![2, 1, 2], vec![0., 1., 2., 3.]).unwrap();
                let input_shape = vec![2, 3, 2];
                let axis = 1;
                let keep_dim = true;
                let divisor = 2;
                let saved = TensorSavedContext::Reduction {
                    axis,
                    keep_dim,
                    divisor,
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                };
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor = Tensor::from_vec(
                    input_shape.to_vec(),
                    vec![0., 1., 0., 1., 0., 1., 2., 3., 2., 3., 2., 3.]
                        .iter()
                        .map(|x| x / divisor as f64)
                        .collect(),
                )
                .unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }

        mod when_saved_is_model {
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
                let saved = TensorSavedContext::Model(ModelSavedContext::MatmulLeft {
                    right: right.clone(),
                    input_shape: input_shape.clone(),
                    output_shape: upstream.shape().to_vec(),
                });
                let result = apply_vjp(&upstream, &saved);

                assert!(result.is_ok(), "{result:?}");
                let expected_tensor =
                    Tensor::from_vec(input_shape.clone(), vec![0.5, 1.1, 1.7]).unwrap();
                assert_eq!(result.unwrap(), expected_tensor);
            }
        }
    }

    mod tensor_value_write_guard {
        use super::*;

        mod fn_commit {
            use super::*;

            #[test]
            fn it_persists_a_tensor_and_its_revision() {
                let new_value = Tensor::from_vec(vec![2, 3], vec![2.0; 6]).unwrap();
                let next_revision = 10;

                let current_value =
                    RefCell::new(Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap());
                let current_revision = Cell::new(9);

                let tensor_value_write_guard = TensorValueWriteGuard {
                    value: current_value.borrow_mut(),
                    revision: &current_revision,
                };

                tensor_value_write_guard.commit(new_value, next_revision);

                assert_eq!(current_revision.get(), next_revision);
                assert_eq!(current_value.borrow().as_slice(), vec![2.0; 6].as_slice());
            }
        }
    }

    mod tensor_value {
        use super::*;

        mod fn_new_node {
            use super::*;

            #[test]
            fn it_create_new_node() {
                let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                let operation = TensorOperation::Add;
                let parents = vec![];
                let parents_ptr = parents.as_ptr();
                let tracked = true;
                let parameter_gradient = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
                let result = TensorValue::new_node(
                    tensor.clone(),
                    operation,
                    parents,
                    tracked,
                    Some(parameter_gradient.clone()),
                );

                assert_eq!(result.node.value.clone().into_inner(), tensor);
                assert_eq!(result.node.value_revision.get(), 0);
                assert_eq!(result.node.operation, operation);
                assert_eq!(result.node.tracked, tracked);
                assert_eq!(result.node.state.borrow().parents.as_ptr(), parents_ptr);
                assert_eq!(result.node.state.borrow().released, false);
                assert_eq!(
                    result
                        .node
                        .state
                        .borrow()
                        .parameter_gradient
                        .as_ref()
                        .unwrap(),
                    &parameter_gradient
                );
            }
        }

        mod fn_parameter {
            use super::*;

            mod when_tensor_contains_non_finite_value {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0., f64::INFINITY]).unwrap();
                    let result = TensorValue::parameter(tensor);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::NonFiniteLeaf {
                            operation: TensorOperation::Parameter,
                            index: 1,
                            value: f64::INFINITY,
                        })
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_creates_parameter_node() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let result = TensorValue::parameter(tensor.clone()).unwrap();

                    assert_eq!(result.node.value.clone().into_inner(), tensor);
                    assert_eq!(result.node.value_revision.get(), 0);
                    assert_eq!(result.node.operation, TensorOperation::Parameter);
                    assert_eq!(result.node.tracked, true);
                    assert_eq!(result.node.state.borrow().parents, vec![]);
                    assert_eq!(result.node.state.borrow().released, false);
                    assert_eq!(
                        result
                            .node
                            .state
                            .borrow()
                            .parameter_gradient
                            .as_ref()
                            .unwrap(),
                        &zeros(tensor.shape()).unwrap()
                    );
                }
            }
        }

        mod fn_constant {
            use super::*;

            mod when_tensor_contains_non_finite_value {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0., f64::INFINITY]).unwrap();
                    let result = TensorValue::constant(tensor);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::NonFiniteLeaf {
                            operation: TensorOperation::Constant,
                            index: 1,
                            value: f64::INFINITY,
                        })
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_creates_constant_node() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let result = TensorValue::constant(tensor.clone()).unwrap();

                    assert_eq!(result.node.value.clone().into_inner(), tensor);
                    assert_eq!(result.node.value_revision.get(), 0);
                    assert_eq!(result.node.operation, TensorOperation::Constant);
                    assert_eq!(result.node.tracked, false);
                    assert_eq!(result.node.state.borrow().parents, vec![]);
                    assert_eq!(result.node.state.borrow().released, false);
                    assert_eq!(result.node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_next_value_revision {
            use super::*;

            mod when_revision_can_not_be_incremented {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    tensor_value.node.value_revision.set(u64::MAX);

                    assert_eq!(tensor_value.next_value_revision(), None);
                }
            }

            mod when_revision_can_be_increased {
                use super::*;

                #[test]
                fn it_returns_new_revision() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    assert_eq!(tensor_value.next_value_revision(), Some(1));
                    assert_eq!(
                        tensor_value.value_revision(),
                        0,
                        "next_value_revision() should not change current revision value, but it did change to {}",
                        tensor_value.value_revision()
                    );
                }
            }
        }

        mod fn_try_value_write {
            use super::*;

            mod when_value_is_already_borrowed {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();

                    let _val = tensor_value.try_value_write().unwrap();
                    let result = tensor_value.try_value_write();

                    assert!(result.is_err(), "{result:?}");
                }
            }

            mod when_value_is_not_borrowed_yet {
                use super::*;

                #[test]
                fn it_creates_write_guard() {
                    let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    let result = tensor_value.try_value_write();

                    assert!(result.is_ok(), "{result:?}");
                }
            }
        }

        mod fn_detach {
            use super::*;

            #[test]
            fn it_creates_detached_node() {
                let tensor = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).unwrap();
                let result = TensorValue::constant(tensor.clone()).unwrap().detach();

                assert_eq!(result.node.value.clone().into_inner(), tensor);
                assert_eq!(result.node.value_revision.get(), 0);
                assert_eq!(result.node.operation, TensorOperation::Detached);
                assert_eq!(result.node.tracked, false);
                assert_eq!(result.node.state.borrow().parents, vec![]);
                assert_eq!(result.node.state.borrow().released, false);
                assert_eq!(result.node.state.borrow().parameter_gradient, None);
            }
        }

        mod fn_operation_node {
            use super::*;

            mod when_tensor_contains_non_finite_value {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0., f64::INFINITY]).unwrap();
                    let context = AutogradContext::default();
                    let operation = TensorOperation::Add;
                    let parents = vec![];
                    let result = TensorValue::operation_node(context, tensor, operation, parents);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::NonFiniteForward {
                            operation,
                            index: 1,
                            value: f64::INFINITY,
                        })
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                mod when_graph_is_in_record_mode {
                    use super::*;

                    mod when_one_of_parents_tracks_gradient {
                        use super::*;

                        #[test]
                        fn it_creates_operation_node_with_tracking() {
                            let parent_tensor = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
                            let parent_tensor_value =
                                TensorValue::parameter(parent_tensor).unwrap();
                            let parent_saved_context = TensorSavedContext::Reshape {
                                input_shape: vec![1, 1],
                                output_shape: vec![1],
                            };
                            let parent_edge = ParentEdge {
                                parent: parent_tensor_value,
                                parent_value_revision: 1,
                                saved: parent_saved_context,
                            };

                            let tensor = Tensor::from_vec(vec![2], vec![0., 1.]).unwrap();
                            let context = AutogradContext::default();
                            let operation = TensorOperation::Add;
                            let parents = vec![parent_edge];
                            let result = TensorValue::operation_node(
                                context,
                                tensor.clone(),
                                operation,
                                parents.clone(),
                            )
                            .unwrap();

                            assert_eq!(result.node.value.clone().into_inner(), tensor);
                            assert_eq!(result.node.value_revision.get(), 0);
                            assert_eq!(result.node.operation, operation);
                            assert_eq!(result.node.tracked, true);
                            assert_eq!(result.node.state.borrow().parents, parents);
                            assert_eq!(result.node.state.borrow().released, false);
                            assert_eq!(result.node.state.borrow().parameter_gradient, None);
                        }
                    }

                    mod when_no_parents_tracks_gradient {
                        use super::*;

                        #[test]
                        fn it_creates_operation_node_without_tracking() {
                            let parent_tensor = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
                            let parent_tensor_value = TensorValue::constant(parent_tensor).unwrap();
                            let parent_saved_context = TensorSavedContext::Reshape {
                                input_shape: vec![1, 1],
                                output_shape: vec![1],
                            };
                            let parent_edge = ParentEdge {
                                parent: parent_tensor_value,
                                parent_value_revision: 1,
                                saved: parent_saved_context,
                            };

                            let tensor = Tensor::from_vec(vec![2], vec![0., 1.]).unwrap();
                            let context = AutogradContext::default();
                            let operation = TensorOperation::Add;
                            let parents = vec![parent_edge];
                            let result = TensorValue::operation_node(
                                context,
                                tensor.clone(),
                                operation,
                                parents.clone(),
                            )
                            .unwrap();

                            assert_eq!(result.node.value.clone().into_inner(), tensor);
                            assert_eq!(result.node.value_revision.get(), 0);
                            assert_eq!(result.node.operation, operation);
                            assert_eq!(result.node.tracked, false);
                            assert_eq!(result.node.state.borrow().parents, parents);
                            assert_eq!(result.node.state.borrow().released, false);
                            assert_eq!(result.node.state.borrow().parameter_gradient, None);
                        }
                    }
                }

                mod when_graph_is_not_in_record_mode {
                    use super::*;

                    #[test]
                    fn it_creates_operation_node_without_tracking_and_without_parents() {
                        let parent_tensor = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
                        let parent_tensor_value = TensorValue::parameter(parent_tensor).unwrap();
                        let parent_saved_context = TensorSavedContext::Reshape {
                            input_shape: vec![1, 1],
                            output_shape: vec![1],
                        };
                        let parent_edge = ParentEdge {
                            parent: parent_tensor_value,
                            parent_value_revision: 1,
                            saved: parent_saved_context,
                        };

                        let tensor = Tensor::from_vec(vec![2], vec![0., 1.]).unwrap();
                        let context = AutogradContext::no_grad();
                        let operation = TensorOperation::Add;
                        let parents = vec![parent_edge];
                        let result = TensorValue::operation_node(
                            context,
                            tensor.clone(),
                            operation,
                            parents,
                        )
                        .unwrap();

                        assert_eq!(result.node.value.clone().into_inner(), tensor);
                        assert_eq!(result.node.value_revision.get(), 0);
                        assert_eq!(result.node.operation, operation);
                        assert_eq!(result.node.tracked, false);
                        assert_eq!(result.node.state.borrow().parents, vec![]);
                        assert_eq!(result.node.state.borrow().released, false);
                        assert_eq!(result.node.state.borrow().parameter_gradient, None);
                    }
                }
            }
        }

        mod fn_topology {
            use super::*;
            use crate::autograd::scalar::ScalarOperation;

            #[test]
            fn it_builds_a_chain_of_tensor_values() {
                let tensor1 = Tensor::from_vec(vec![2], vec![0.5, 1.0]).unwrap();
                let tensor_value1 = TensorValue::parameter(tensor1.clone()).unwrap();
                let tensor2 = Tensor::from_vec(vec![2], vec![3.0, 5.0]).unwrap();
                let tensor_value2 = TensorValue::parameter(tensor2.clone()).unwrap();
                let tensor3 = Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap();
                let tensor_value3 = TensorValue::constant(tensor3.clone()).unwrap();

                let final_tensor_value = tensor_value1
                    .add(&tensor_value2)
                    .unwrap()
                    .mul(&tensor_value3)
                    .unwrap();
                let result = final_tensor_value.topology().unwrap();

                assert_eq!(result.len(), 5);

                assert_eq!(result[0].value().clone(), tensor1.clone());
                assert_eq!(result[0].operation(), TensorOperation::Parameter);

                assert_eq!(result[1].value().clone(), tensor2.clone());
                assert_eq!(result[1].operation(), TensorOperation::Parameter);

                assert_eq!(
                    result[2].value().clone(),
                    Tensor::from_vec(vec![2], vec![3.5, 6.0]).unwrap()
                );
                assert_eq!(result[2].operation(), TensorOperation::Add);

                assert_eq!(result[3].value().clone(), tensor3.clone());
                assert_eq!(result[3].operation(), TensorOperation::Constant);

                assert_eq!(
                    result[4].value().clone(),
                    Tensor::from_vec(vec![2], vec![7.0, 18.0]).unwrap()
                );
                assert_eq!(result[4].operation(), TensorOperation::Multiply);
            }
        }

        mod fn_model_operation_with_context {
            use super::*;

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_computes_operation_node_based_on_the_result_of_forward_function() {
                    let tensor = Tensor::from_vec(vec![2], vec![0., 1.]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();

                    let operation = TensorOperation::Exp;
                    let context = AutogradContext::default();
                    let operands = [&tensor_value];
                    let dummy_exp_tensor = Tensor::from_vec(vec![2], vec![123., 321.]).unwrap();
                    let dummy_exp_forward = |_operands: [&Tensor; 1]| {
                        Ok((
                            dummy_exp_tensor.clone(),
                            [ModelSavedContext::Exp {
                                output: dummy_exp_tensor.clone(),
                            }],
                        ))
                    };

                    let result = TensorValue::model_operation_with_context(
                        context,
                        operation,
                        operands,
                        dummy_exp_forward,
                    );

                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        dummy_exp_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, operation);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![ParentEdge::capture(
                            &tensor_value,
                            TensorSavedContext::Model(ModelSavedContext::Exp {
                                output: dummy_exp_tensor.clone()
                            })
                        )]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0., 1.]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    tensor_value.node.state.borrow_mut().released = true;

                    let operation = TensorOperation::Exp;
                    let context = AutogradContext::default();
                    let operands = [&tensor_value];
                    let dummy_exp_forward = |_operands: [&Tensor; 1]| {
                        Err(TensorAutodiffError::Tensor(TensorError::ShapeOverflow))
                    };

                    let result = TensorValue::model_operation_with_context(
                        context,
                        operation,
                        operands,
                        dummy_exp_forward,
                    );

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation,
                            operand: 0
                        })
                    );
                }
            }
        }

        mod fn_add_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor1 = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value1 = TensorValue::parameter(tensor1).unwrap();
                    let tensor2 = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value2 = TensorValue::parameter(tensor2).unwrap();
                    tensor_value2.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value1.add_with_context(context, &tensor_value2);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Add,
                            operand: 1
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_adds_two_tensors() {
                    let tensor1 = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value1 = TensorValue::parameter(tensor1.clone()).unwrap();
                    let tensor2 = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value2 = TensorValue::parameter(tensor2.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value1.add_with_context(context, &tensor_value2);

                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor =
                        Tensor::from_vec(vec![2, 2], vec![1.5, 2.75, 3.5, 4.75]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Add);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![
                            ParentEdge::capture(
                                &tensor_value1,
                                broadcast_context(tensor1.shape(), tensor2.shape())
                            ),
                            ParentEdge::capture(
                                &tensor_value2,
                                broadcast_context(tensor2.shape(), tensor2.shape())
                            )
                        ]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_mul_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor1 = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value1 = TensorValue::parameter(tensor1).unwrap();
                    let tensor2 = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value2 = TensorValue::parameter(tensor2).unwrap();
                    tensor_value2.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value1.mul_with_context(context, &tensor_value2);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Multiply,
                            operand: 1
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_multiplies_two_tensors() {
                    let tensor1 = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value1 = TensorValue::parameter(tensor1.clone()).unwrap();
                    let tensor2 = Tensor::from_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
                    let tensor_value2 = TensorValue::parameter(tensor2.clone()).unwrap();

                    let context = AutogradContext::default();
                    let result = tensor_value1.mul_with_context(context, &tensor_value2);

                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor =
                        Tensor::from_vec(vec![2, 2], vec![0.5, 1.5, 1.5, 3.0]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Multiply);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![
                            ParentEdge::capture(
                                &tensor_value1,
                                multiply_context(tensor1.shape(), tensor2.shape(), tensor2.clone())
                            ),
                            ParentEdge::capture(
                                &tensor_value2,
                                multiply_context(tensor2.shape(), tensor2.shape(), tensor1.clone())
                            )
                        ]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_reshape_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    let new_shape = [1, 2];
                    tensor_value.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.reshape_with_context(context, &new_shape);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Reshape,
                            operand: 0
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_changes_the_shape_of_the_tensor_to_new_one() {
                    let tensor = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    let new_shape = [1, 2];

                    let context = AutogradContext::default();
                    let result = tensor_value.reshape_with_context(context, &new_shape);

                    println!("{:?}", result);
                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Reshape);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![ParentEdge::capture(
                            &tensor_value,
                            TensorSavedContext::Reshape {
                                input_shape: tensor.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            }
                        ),]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_transpose_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    let first_axis = 0;
                    let second_axis = 1;
                    tensor_value.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result =
                        tensor_value.transpose_with_context(context, first_axis, second_axis);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Transpose,
                            operand: 0
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_swaps_given_axis_by_their_places() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    let first_axis = 0;
                    let second_axis = 1;

                    let context = AutogradContext::default();
                    let result =
                        tensor_value.transpose_with_context(context, first_axis, second_axis);

                    println!("{:?}", result);
                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor = Tensor::from_vec(vec![2, 1], vec![0.5, 0.75]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Transpose);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![ParentEdge::capture(
                            &tensor_value,
                            TensorSavedContext::Transpose {
                                first_axis,
                                second_axis,
                                input_shape: tensor.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            }
                        ),]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_broadcast_to_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    let new_shape = [1, 2];
                    tensor_value.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.broadcast_to_with_context(context, &new_shape);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Broadcast,
                            operand: 0
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                mod when_tensor_broadcasts_exactly_into_given_shape {
                    use super::*;

                    #[test]
                    fn it_broadcasts_the_tensor_into_the_new_shape() {
                        let tensor = Tensor::from_vec(vec![2], vec![0.5, 0.75]).unwrap();
                        let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                        let new_shape = [1, 2];

                        let context = AutogradContext::default();
                        let result = tensor_value.broadcast_to_with_context(context, &new_shape);

                        println!("{:?}", result);
                        assert!(result.is_ok(), "{result:?}");
                        let result_node = &result.as_ref().unwrap().node;
                        let expected_tensor =
                            Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                        assert_eq!(
                            result_node.value.clone().into_inner(),
                            expected_tensor.clone()
                        );
                        assert_eq!(result_node.value_revision.get(), 0);
                        assert_eq!(result_node.operation, TensorOperation::Broadcast);
                        assert_eq!(result_node.tracked, true);
                        assert_eq!(
                            result_node.state.borrow().parents,
                            vec![ParentEdge::capture(
                                &tensor_value,
                                broadcast_context(tensor.shape(), expected_tensor.shape())
                            ),]
                        );
                        assert_eq!(result_node.state.borrow().released, false);
                        assert_eq!(result_node.state.borrow().parameter_gradient, None);
                    }
                }

                mod when_broadcasted_shape_differs_from_the_given_shape {
                    use super::*;

                    #[test]
                    fn it_returns_error() {
                        let tensor = Tensor::from_vec(vec![1, 2, 3], vec![0.; 6]).unwrap();
                        let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                        let new_shape = [2, 3];

                        let context = AutogradContext::default();
                        let result = tensor_value.broadcast_to_with_context(context, &new_shape);

                        assert!(result.is_err(), "{result:?}");
                        assert_eq!(
                            result.err(),
                            Some(TensorAutodiffError::BroadcastTargetMismatch {
                                input: tensor.shape().to_vec(),
                                requested: new_shape.to_vec(),
                                inferred: vec![1, 2, 3],
                            })
                        );
                    }
                }
            }
        }

        mod fn_sum_axis_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    let axis = 1;
                    let keep_dim = true;
                    tensor_value.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.sum_axis_with_context(context, axis, keep_dim);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Sum,
                            operand: 0
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_calculates_a_sum_of_the_given_axis() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    let axis = 1;
                    let keep_dim = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.sum_axis_with_context(context, axis, keep_dim);

                    println!("{:?}", result);
                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor = Tensor::from_vec(vec![1, 1], vec![1.25]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Sum);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![ParentEdge::capture(
                            &tensor_value,
                            TensorSavedContext::Reduction {
                                axis,
                                keep_dim,
                                divisor: 1,
                                input_shape: tensor.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            }
                        )]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_mean_axis_with_context {
            use super::*;

            mod when_operation_is_not_available {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor).unwrap();
                    let axis = 1;
                    let keep_dim = true;
                    tensor_value.node.state.borrow_mut().released = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.mean_axis_with_context(context, axis, keep_dim);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err(),
                        Some(TensorAutodiffError::ReleasedOperand {
                            operation: TensorOperation::Mean,
                            operand: 0
                        })
                    );
                }
            }

            mod when_operation_is_available {
                use super::*;

                #[test]
                fn it_calculates_mean_value_of_the_given_axis() {
                    let tensor = Tensor::from_vec(vec![1, 2], vec![0.5, 0.75]).unwrap();
                    let tensor_value = TensorValue::parameter(tensor.clone()).unwrap();
                    let axis = 1;
                    let keep_dim = true;

                    let context = AutogradContext::default();
                    let result = tensor_value.mean_axis_with_context(context, axis, keep_dim);

                    assert!(result.is_ok(), "{result:?}");
                    let result_node = &result.as_ref().unwrap().node;
                    let expected_tensor = Tensor::from_vec(vec![1, 1], vec![0.625]).unwrap();
                    assert_eq!(
                        result_node.value.clone().into_inner(),
                        expected_tensor.clone()
                    );
                    assert_eq!(result_node.value_revision.get(), 0);
                    assert_eq!(result_node.operation, TensorOperation::Mean);
                    assert_eq!(result_node.tracked, true);
                    assert_eq!(
                        result_node.state.borrow().parents,
                        vec![ParentEdge::capture(
                            &tensor_value,
                            TensorSavedContext::Reduction {
                                axis,
                                keep_dim,
                                divisor: 2,
                                input_shape: tensor.shape().to_vec(),
                                output_shape: expected_tensor.shape().to_vec(),
                            }
                        )]
                    );
                    assert_eq!(result_node.state.borrow().released, false);
                    assert_eq!(result_node.state.borrow().parameter_gradient, None);
                }
            }
        }

        mod fn_backward_with_trace {
            use super::*;
            use crate::support::gradcheck::sampled_tensor_gradient_check;

            fn constant(tensor: Result<Tensor, TensorError>) -> TensorValue {
                let tensor = tensor.unwrap();
                TensorValue::constant(tensor).unwrap()
            }

            fn forward(parameter_value: Tensor) -> (TensorValue, TensorValue) {
                let parameter = TensorValue::parameter(parameter_value).unwrap();
                // [4, 4] -> [4, 4] -> [1, 4, 4] -> [4, 1, 4] -> [1, 4, 4]
                let gathered = parameter.gather_rows(&[2, 0, 2, 3], &[4]).unwrap();
                let broadcast = gathered.broadcast_to(&[1, 4, 4]).unwrap();
                let transpose = broadcast.transpose(0, 1).unwrap();
                let reshape = transpose.reshape(&[1, 4, 4]).unwrap();

                let bias =
                    constant(Tensor::from_vec(vec![4], vec![0.10, -0.20, 0.05, 0.15])).detach();
                let add = reshape.add(&bias).unwrap();

                let scale = constant(Tensor::from_vec(
                    vec![1, 1, 4],
                    vec![0.50, 0.75, 1.00, 1.25],
                ));
                let multiply = add.mul(&scale).unwrap();

                // Reuse the parameter as the right matmul operand. Its gradient must combine
                // this direct edge with the longer path through gather and elementwise ops.
                let matmul = multiply.matmul(&parameter).unwrap();
                let exp = matmul.exp().unwrap();
                let log = exp.log().unwrap();
                let silu = log.silu().unwrap();
                let log_softmax = silu.log_softmax(2).unwrap();
                let causal_softmax = log_softmax.causal_softmax().unwrap();

                let angles = [0.00_f64, 0.00, 0.10, 0.20, 0.20, 0.40, 0.30, 0.60];
                let cosines =
                    Tensor::from_vec(vec![4, 2], angles.iter().map(|angle| angle.cos()).collect())
                        .unwrap();
                let sines =
                    Tensor::from_vec(vec![4, 2], angles.iter().map(|angle| angle.sin()).collect())
                        .unwrap();
                let rotary_pairs = causal_softmax.rotary_pairs(&cosines, &sines).unwrap();
                // [1, 4, 4] -> [4, 4] -> [1, 4] -> scalar
                let sum = rotary_pairs.sum_axis(0, false).unwrap();
                let mean = sum.mean_axis(0, true).unwrap();
                let loss = mean.indexed_mean_nll(1, &[2]).unwrap();

                (parameter, loss)
            }

            #[test]
            fn it_calculates_gradients_of_itself_and_all_parents() {
                let parameter_value = Tensor::from_vec(
                    vec![4, 4],
                    vec![
                        0.10, 0.20, 0.30, 0.40, //
                        0.50, 0.60, 0.70, 0.80, //
                        0.90, 1.00, 1.10, 1.20, //
                        1.30, 1.40, 1.50, 1.60,
                    ],
                )
                .unwrap();
                let (parameter, loss) = forward(parameter_value.clone());

                let backward = loss.backward_with_trace().unwrap();
                let gradient = parameter.gradient_snapshot().unwrap();

                assert_eq!(
                    backward
                        .nodes
                        .iter()
                        .map(|node| node.operation)
                        .collect::<Vec<_>>(),
                    [
                        TensorOperation::Parameter,
                        TensorOperation::GatherRows,
                        TensorOperation::Broadcast,
                        TensorOperation::Transpose,
                        TensorOperation::Reshape,
                        TensorOperation::Detached,
                        TensorOperation::Add,
                        TensorOperation::Constant,
                        TensorOperation::Multiply,
                        TensorOperation::MatMul,
                        TensorOperation::Exp,
                        TensorOperation::Log,
                        TensorOperation::Silu,
                        TensorOperation::LogSoftmax,
                        TensorOperation::CausalSoftmax,
                        TensorOperation::RotaryPairs,
                        TensorOperation::Sum,
                        TensorOperation::Mean,
                        TensorOperation::IndexedMeanNll,
                    ]
                );
                assert!(
                    backward
                        .nodes
                        .iter()
                        .all(|node| node.pass_adjoint.is_some() == node.tracked)
                );
                assert_eq!(
                    backward.nodes.last().unwrap().pass_adjoint,
                    Some(Tensor::from_vec(vec![], vec![1.0]).unwrap())
                );
                assert_eq!(backward.nodes[0].pass_adjoint, Some(gradient.clone()));
                assert_eq!(
                    backward.nodes[0].accumulated_gradient,
                    Some(gradient.clone())
                );
                assert!(gradient.as_slice().iter().any(|value| *value != 0.0));

                let mut checked_parameter = parameter_value;
                let check = sampled_tensor_gradient_check(
                    &mut checked_parameter,
                    &gradient.view(),
                    1.0e-5,
                    1.0e-6,
                    16,
                    |candidate| {
                        let (_, candidate_loss) = forward(candidate.clone());
                        candidate_loss.value().as_slice()[0]
                    },
                )
                .unwrap();

                assert!(check.passed, "{check:#?}");
                assert_eq!(check.checks.len(), checked_parameter.len());
            }
        }
    }
}
