//! A dependency-free scalar reverse-mode computation graph

use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// The operation that produced a scalar graph node
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ScalarOperation {
    Variable,
    Constant,
    Detached,
    Add,
    Multiply,
    Negate,
    Subtract,
    Exp,
    Tanh,
}

impl ScalarOperation {
    /// A stable, locale-neutral name suitable for deterministic evidence.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Variable => "variable",
            Self::Constant => "constant",
            Self::Detached => "detached",
            Self::Add => "add",
            Self::Multiply => "mul",
            Self::Negate => "neg",
            Self::Subtract => "sub",
            Self::Exp => "exp",
            Self::Tanh => "tanh",
        }
    }
}

impl fmt::Display for ScalarOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A deterministic rejection from scalar graph construction or backpropagation.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarAutodiffError {
    NonFiniteLeaf {
        operation: ScalarOperation,
        value: f64,
    },
    NonFiniteResult {
        operation: ScalarOperation,
        value: f64,
    },
    UntrackedOutput {
        operation: ScalarOperation,
    },
    NonFiniteSeed {
        seed: f64,
    },
    NonFiniteContribution {
        child: usize,
        parent: usize,
        operand: usize,
        upstream: f64,
        local_derivative: f64,
    },
    NonFinitePassAdjoint {
        node: usize,
        previous: f64,
        contribution: f64,
    },
    NonFiniteAccumulatedGradient {
        node: usize,
        stored: f64,
        pass_adjoint: f64,
    },
}

impl fmt::Display for ScalarAutodiffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteLeaf { operation, value } => {
                write!(
                    formatter,
                    "{operation} scalar value {value:?} must be finite"
                )
            }
            Self::NonFiniteResult { operation, value } => {
                write!(
                    formatter,
                    "{operation} produced non-finite scalar value {value:?}"
                )
            }
            Self::UntrackedOutput { operation } => write!(
                formatter,
                "cannot backpropagate from untracked {operation} output"
            ),
            Self::NonFiniteSeed { seed } => {
                write!(formatter, "backward seed {seed:?} must be finite")
            }
            Self::NonFiniteContribution {
                child,
                parent,
                operand,
                upstream,
                local_derivative,
            } => write!(
                formatter,
                "edge {operand} from topology node {child} to {parent} produced a non-finite contribution from upstream {upstream:?} and local derivative {local_derivative:?}"
            ),
            Self::NonFinitePassAdjoint {
                node,
                previous,
                contribution,
            } => write!(
                formatter,
                "topology node {node} cannot accumulate pass adjoint {previous:?} plus contribution {contribution:?}"
            ),
            Self::NonFiniteAccumulatedGradient {
                node,
                stored,
                pass_adjoint,
            } => write!(
                formatter,
                "topology node {node} cannot accumulate stored gradient {stored:?} plus pass adjoint {pass_adjoint:?}"
            ),
        }
    }
}

impl Error for ScalarAutodiffError {}

#[derive(Clone)]
struct ParentEdge {
    parent: Scalar,
    local_derivative: f64,
}

struct Node {
    value: f64,
    operation: ScalarOperation,
    parents: Vec<ParentEdge>,
    gradient: Option<f64>,
}

type NodeKey = *const RefCell<Node>;

#[derive(Clone)]
pub struct Scalar {
    node: Rc<RefCell<Node>>,
}

/// One node in the deterministic parent-first order used by a backward pass
#[derive(Clone, Debug, PartialEq)]
pub struct BackwardNode {
    pub topology_index: usize,
    pub operation: ScalarOperation,
    pub value: f64,
    pub tracked: bool,
    pub pass_adjoin: Option<f64>,
    pub accumulated_gradient: Option<f64>,
}

/// One ordered operand edge visited during reverse traversal
#[derive(Clone, Debug, PartialEq)]
pub struct BackwardEdge {
    pub reverse_index: usize,
    pub child: usize,
    pub parent: usize,
    pub operand: usize,
    pub local_derivative: f64,
    pub upstream: f64,
    pub contribution: f64,
    pub parent_tracked: bool,
    pub parent_adjoint_before: Option<f64>,
    pub parent_adjoin_after: Option<f64>,
}

/// Rust-authored evidence from one fresh, successfully committed backward pass
#[derive(Clone, Debug, PartialEq)]
pub struct BackwardPass {
    pub seed: f64,
    pub nodes: Vec<BackwardNode>,
    pub edges: Vec<BackwardEdge>,
}

impl Scalar {
    fn new_node(
        value: f64,
        operation: ScalarOperation,
        parents: Vec<ParentEdge>,
        tracked: bool,
    ) -> Self {
        Self {
            node: Rc::new(RefCell::new(Node {
                value,
                operation,
                parents,
                gradient: tracked.then_some(0.0),
            })),
        }
    }

    fn leaf(
        value: f64,
        operation: ScalarOperation,
        tracked: bool,
    ) -> Result<Self, ScalarAutodiffError> {
        if !value.is_finite() {
            return Err(ScalarAutodiffError::NonFiniteLeaf { operation, value });
        }

        Ok(Self::new_node(value, operation, Vec::new(), tracked))
    }

    pub fn tracks_gradient(&self) -> bool {
        self.node.borrow().gradient.is_some()
    }

    fn operation_node(
        value: f64,
        operation: ScalarOperation,
        parents: Vec<ParentEdge>,
    ) -> Result<Self, ScalarAutodiffError> {
        if !value.is_finite() {
            return Err(ScalarAutodiffError::NonFiniteResult { operation, value });
        }

        let tracked = parents.iter().any(|edge| edge.parent.tracks_gradient());
        Ok(Self::new_node(value, operation, parents, tracked))
    }

    /// Creates a finite leaf whose gradient is tracked
    pub fn variable(value: f64) -> Result<Self, ScalarAutodiffError> {
        Self::leaf(value, ScalarOperation::Variable, true)
    }

    /// Creates a finite leaf treated as a constant by backpropagation
    pub fn constant(value: f64) -> Result<Self, ScalarAutodiffError> {
        Self::leaf(value, ScalarOperation::Constant, false)
    }

    pub fn value(&self) -> f64 {
        self.node.borrow().value
    }

    pub fn operation(&self) -> ScalarOperation {
        self.node.borrow().operation
    }

    pub fn gradient(&self) -> Option<f64> {
        self.node.borrow().gradient
    }

    /// Returns whether two handles refer to the same graph node
    pub fn is_same_node(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }

    /// Adds two finite scalars and records both ordered operand edges
    pub fn add(&self, other: &Self) -> Result<Self, ScalarAutodiffError> {
        Self::operation_node(
            self.value() + other.value(),
            ScalarOperation::Add,
            vec![
                ParentEdge {
                    parent: self.clone(),
                    local_derivative: 1.0,
                },
                ParentEdge {
                    parent: other.clone(),
                    local_derivative: 1.0,
                },
            ],
        )
    }

    /// Multiplies two finite scalars and records one edge per operand use
    pub fn mul(&self, other: &Self) -> Result<Self, ScalarAutodiffError> {
        let left = self.value();
        let right = other.value();

        Self::operation_node(
            left * right,
            ScalarOperation::Multiply,
            vec![
                ParentEdge {
                    parent: self.clone(),
                    local_derivative: right,
                },
                ParentEdge {
                    parent: other.clone(),
                    local_derivative: left,
                },
            ],
        )
    }

    pub fn neg(&self) -> Result<Self, ScalarAutodiffError> {
        Self::operation_node(
            -self.value(),
            ScalarOperation::Negate,
            vec![ParentEdge {
                parent: self.clone(),
                local_derivative: -1.0,
            }],
        )
    }

    pub fn sub(&self, other: &Self) -> Result<Self, ScalarAutodiffError> {
        Self::operation_node(
            self.value() - other.value(),
            ScalarOperation::Subtract,
            vec![
                ParentEdge {
                    parent: self.clone(),
                    local_derivative: 1.0,
                },
                ParentEdge {
                    parent: other.clone(),
                    local_derivative: -1.0,
                },
            ],
        )
    }

    pub fn exp(&self) -> Result<Self, ScalarAutodiffError> {
        let value = self.value().exp();
        Self::operation_node(
            value,
            ScalarOperation::Exp,
            vec![ParentEdge {
                parent: self.clone(),
                local_derivative: value,
            }],
        )
    }

    pub fn tanh(&self) -> Result<Self, ScalarAutodiffError> {
        let value = self.value().tanh();
        Self::operation_node(
            value,
            ScalarOperation::Tanh,
            vec![ParentEdge {
                parent: self.clone(),
                local_derivative: 1.0 - value * value,
            }],
        )
    }

    /// Copies the primal into a new untracked constant with no parent edge
    pub fn detach(&self) -> Self {
        Self::new_node(self.value(), ScalarOperation::Detached, Vec::new(), false)
    }

    fn key(&self) -> NodeKey {
        Rc::as_ptr(&self.node)
    }

    fn topology(&self) -> Vec<Self> {
        fn visit(node: &Scalar, visited: &mut FxHashSet<NodeKey>, order: &mut Vec<Scalar>) {
            if !visited.insert(node.key()) {
                return;
            }
            let parents = node.node.borrow().parents.clone();
            for edge in parents {
                visit(&edge.parent, visited, order);
            }
            order.push(node.clone());
        }

        let mut visited = FxHashSet::default();
        let mut order = Vec::new();
        visit(self, &mut visited, &mut order);
        order
    }

    /// Accumulates one fresh reverse pass without reading stale intermediate grads.
    ///
    /// No stored gradient changes unless every contribution, pass adjoint, and prospective
    /// accumulated gradient is finite.
    // If to throw out every safe checks, the function's logic can be expressed as follows:
    // fn backward_with_seed(output, seed) {
    //     let topology = topological_sort(output);
    //
    //     let mut adjoints = vec![0; topology.len()];
    //     adjoints[output] = seed;
    //
    //     for child in topology.reverse() {
    //         for edge in child.parents {
    //             if edge.parent.tracked {
    //                 adjoints[parent] += adjoints[child] * edge.local_derivative;
    //             }
    //         }
    //     }
    //
    //     for node in topology {
    //         if node.tracked {
    //             node.gradient += adjoints[node];
    //         }
    //     }
    // }
    pub fn backward_with_seed(&self, seed: f64) -> Result<BackwardPass, ScalarAutodiffError> {
        if !self.tracks_gradient() {
            return Err(ScalarAutodiffError::UntrackedOutput {
                operation: self.operation(),
            });
        }
        if !seed.is_finite() {
            return Err(ScalarAutodiffError::NonFiniteSeed { seed });
        }

        let topology = self.topology();
        let indices = topology
            .iter()
            .enumerate()
            .map(|(index, scalar)| (scalar.key(), index))
            .collect::<FxHashMap<_, _>>();
        let mut pass_adjoints = vec![0.0; topology.len()];
        pass_adjoints[topology.len() - 1] = seed;
        let mut edges = Vec::new();

        for child in (0..topology.len()).rev() {
            let upstream = pass_adjoints[child];
            let parents = topology[child].node.borrow().parents.clone();
            for (operand, edge) in parents.iter().enumerate() {
                let parent = indices[&edge.parent.key()];
                let contribution = upstream * edge.local_derivative;
                if !contribution.is_finite() {
                    return Err(ScalarAutodiffError::NonFiniteContribution {
                        child,
                        parent,
                        operand,
                        upstream,
                        local_derivative: edge.local_derivative,
                    });
                }

                let parent_tracked = edge.parent.tracks_gradient();
                let (before, after) = if parent_tracked {
                    let previous = pass_adjoints[parent];
                    let next = previous + contribution;
                    if !next.is_finite() {
                        return Err(ScalarAutodiffError::NonFinitePassAdjoint {
                            node: parent,
                            previous,
                            contribution,
                        });
                    }
                    pass_adjoints[parent] = next;
                    (Some(previous), Some(next))
                } else {
                    (None, None)
                };

                edges.push(BackwardEdge {
                    reverse_index: edges.len(),
                    child,
                    parent,
                    operand,
                    local_derivative: edge.local_derivative,
                    upstream,
                    contribution,
                    parent_tracked,
                    parent_adjoint_before: before,
                    parent_adjoin_after: after,
                });
            }
        }

        let prospective = topology
            .iter()
            .enumerate()
            .map(|(index, scalar)| {
                scalar.gradient().map(|stored| {
                    let pass_adjoint = pass_adjoints[index];
                    let accumulated = stored + pass_adjoint;
                    if !accumulated.is_finite() {
                        Err(ScalarAutodiffError::NonFiniteAccumulatedGradient {
                            node: index,
                            stored,
                            pass_adjoint,
                        })
                    } else {
                        Ok(accumulated)
                    }
                })
            })
            .map(|candidate| candidate.transpose())
            .collect::<Result<Vec<_>, _>>()?;

        for (scalar, &gradient) in topology.iter().zip(&prospective) {
            if let Some(gradient) = gradient {
                scalar.node.borrow_mut().gradient = Some(gradient);
            }
        }

        let nodes = topology
            .iter()
            .enumerate()
            .map(|(topology_index, scalar)| BackwardNode {
                topology_index,
                operation: scalar.operation(),
                value: scalar.value(),
                tracked: scalar.tracks_gradient(),
                pass_adjoin: scalar
                    .tracks_gradient()
                    .then_some(pass_adjoints[topology_index]),
                accumulated_gradient: prospective[topology_index],
            })
            .collect();

        Ok(BackwardPass { seed, nodes, edges })
    }

    /// Accumulates one fresh reverse pass seeded by 1.0
    pub fn backward(&self) -> Result<BackwardPass, ScalarAutodiffError> {
        self.backward_with_seed(1.0)
    }

    /// Clears every reachable tracked node without changing the graph or values
    pub fn zero_grad(&self) {
        for scalar in self.topology() {
            let mut node = scalar.node.borrow_mut();
            if node.gradient.is_some() {
                node.gradient = Some(0.0);
            }
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Scalar")
            .field("value", &self.value())
            .field("operation", &self.operation())
            .field("gradient", &self.gradient())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod scalar {
        use super::*;

        fn dummy_parents() -> Vec<ParentEdge> {
            vec![
                ParentEdge {
                    local_derivative: 1.0,
                    parent: Scalar {
                        node: Rc::new(RefCell::new(Node {
                            value: 2.0,
                            operation: ScalarOperation::Constant,
                            parents: vec![],
                            gradient: Some(0.5),
                        })),
                    },
                },
                ParentEdge {
                    local_derivative: 1.0,
                    parent: Scalar {
                        node: Rc::new(RefCell::new(Node {
                            value: 3.0,
                            operation: ScalarOperation::Constant,
                            parents: vec![],
                            gradient: None,
                        })),
                    },
                },
            ]
        }

        fn dummy_parents_wo_gradient() -> Vec<ParentEdge> {
            vec![
                ParentEdge {
                    local_derivative: 1.0,
                    parent: Scalar {
                        node: Rc::new(RefCell::new(Node {
                            value: 2.0,
                            operation: ScalarOperation::Constant,
                            parents: vec![],
                            gradient: None,
                        })),
                    },
                },
                ParentEdge {
                    local_derivative: 1.0,
                    parent: Scalar {
                        node: Rc::new(RefCell::new(Node {
                            value: 3.0,
                            operation: ScalarOperation::Constant,
                            parents: vec![],
                            gradient: None,
                        })),
                    },
                },
            ]
        }

        mod fn_new_node {
            use super::*;

            mod when_tracked_is_false {
                use super::*;

                #[test]
                fn it_computes_new_scalar_without_gradient() {
                    let parents = dummy_parents();
                    let parents_ptr = parents.as_ptr();
                    let value = 1.5;
                    let operation = ScalarOperation::Exp;
                    let tracked = false;
                    let result = Scalar::new_node(value, operation, parents, tracked);

                    assert_eq!(result.node.borrow().value, value);
                    assert_eq!(result.node.borrow().operation, operation);
                    assert_eq!(result.node.borrow().parents.as_ptr(), parents_ptr);
                    assert_eq!(result.node.borrow().gradient, None);
                }
            }

            mod when_tracked_is_true {
                use super::*;

                #[test]
                fn it_computes_new_scalar_with_default_gradient() {
                    let parents = dummy_parents();
                    let parents_ptr = parents.as_ptr();
                    let value = 1.5;
                    let operation = ScalarOperation::Exp;
                    let tracked = true;
                    let result = Scalar::new_node(value, operation, parents, tracked);

                    assert_eq!(result.node.borrow().value, value);
                    assert_eq!(result.node.borrow().operation, operation);
                    assert_eq!(result.node.borrow().parents.as_ptr(), parents_ptr);
                    assert_eq!(result.node.borrow().gradient, Some(0.0));
                }
            }
        }

        mod fn_leaf {
            use super::*;

            mod when_value_is_finite {
                use super::*;

                mod when_tracked_is_false {
                    use super::*;

                    #[test]
                    fn it_computes_new_scalar_without_gradient() {
                        let value = 1.5;
                        let operation = ScalarOperation::Exp;
                        let tracked = false;
                        let result = Scalar::leaf(value, operation, tracked);

                        assert!(result.is_ok());
                        assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                        assert_eq!(result.as_ref().unwrap().node.borrow().operation, operation);
                        assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 0);
                        assert_eq!(result.as_ref().unwrap().node.borrow().gradient, None);
                    }
                }

                mod when_tracked_is_true {
                    use super::*;

                    #[test]
                    fn it_computes_new_scalar_with_default_gradient() {
                        let value = 1.5;
                        let operation = ScalarOperation::Exp;
                        let tracked = true;
                        let result = Scalar::leaf(value, operation, tracked);

                        assert!(result.is_ok());
                        assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                        assert_eq!(result.as_ref().unwrap().node.borrow().operation, operation);
                        assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 0);
                        assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
                    }
                }
            }

            mod when_value_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let value = f64::INFINITY;
                    let operation = ScalarOperation::Exp;
                    let tracked = true;
                    let result = Scalar::leaf(value, operation, tracked);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err(),
                        Some(ScalarAutodiffError::NonFiniteLeaf { operation, value })
                    );
                }
            }
        }

        mod fn_tracks_gradient {
            use super::*;

            mod when_gradient_is_tracked {
                use super::*;

                #[test]
                fn it_returns_true() {
                    let value = 1.5;
                    let operation = ScalarOperation::Exp;
                    let tracked = true;
                    let scalar = Scalar::leaf(value, operation, tracked).unwrap();

                    assert_eq!(scalar.tracks_gradient(), true);
                }
            }

            mod when_gradient_is_not_tracked {
                use super::*;

                #[test]
                fn it_returns_false() {
                    let value = 1.5;
                    let operation = ScalarOperation::Exp;
                    let tracked = false;
                    let scalar = Scalar::leaf(value, operation, tracked).unwrap();

                    assert_eq!(scalar.tracks_gradient(), false);
                }
            }
        }

        mod fn_operation_node {
            use super::*;

            mod when_value_is_finite {
                use super::*;

                mod when_one_of_parent_nodes_are_trackable {
                    use super::*;

                    #[test]
                    fn it_computes_new_scalar_with_default_gradient() {
                        let value = 1.5;
                        let operation = ScalarOperation::Exp;
                        let parents = dummy_parents();
                        let parents_ptr = parents.as_ptr();
                        let result = Scalar::operation_node(value, operation, parents);

                        assert!(result.is_ok());
                        assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                        assert_eq!(result.as_ref().unwrap().node.borrow().operation, operation);
                        assert_eq!(
                            result.as_ref().unwrap().node.borrow().parents.as_ptr(),
                            parents_ptr
                        );
                        assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
                    }
                }

                mod when_parents_are_empty {
                    use super::*;

                    #[test]
                    fn it_computes_new_scalar_without_gradient() {
                        let value = 1.5;
                        let operation = ScalarOperation::Exp;
                        let result = Scalar::operation_node(value, operation, vec![]);

                        assert!(result.is_ok());
                        assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                        assert_eq!(result.as_ref().unwrap().node.borrow().operation, operation);
                        assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 0);
                        assert_eq!(result.as_ref().unwrap().node.borrow().gradient, None);
                    }
                }

                mod when_parents_do_not_have_gradient {
                    use super::*;

                    #[test]
                    fn it_computes_new_scalar_without_gradient() {
                        let value = 1.5;
                        let operation = ScalarOperation::Exp;
                        let parents = dummy_parents_wo_gradient();
                        let result = Scalar::operation_node(value, operation, parents);

                        assert!(result.is_ok());
                        assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                        assert_eq!(result.as_ref().unwrap().node.borrow().operation, operation);
                        assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 2);
                        assert_eq!(result.as_ref().unwrap().node.borrow().gradient, None);
                    }
                }
            }

            mod when_value_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let value = f64::INFINITY;
                    let operation = ScalarOperation::Exp;
                    let parents = vec![];
                    let result = Scalar::operation_node(value, operation, parents);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err(),
                        Some(ScalarAutodiffError::NonFiniteResult { operation, value })
                    );
                }
            }
        }

        mod fn_variable {
            use super::*;

            #[test]
            fn it_computes_scalar_variable() {
                let value = 1.5;
                let result = Scalar::variable(value);

                assert!(result.is_ok());
                assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Variable
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 0);
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_constant {
            use super::*;

            #[test]
            fn it_computes_scalar_constant() {
                let value = 1.5;
                let result = Scalar::constant(value);

                assert!(result.is_ok());
                assert_eq!(result.as_ref().unwrap().node.borrow().value, value);
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Constant
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().parents.len(), 0);
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, None);
            }
        }

        mod fn_is_same_ptr {
            use super::*;

            mod when_node_is_the_same {
                use super::*;

                #[test]
                fn it_returns_true() {
                    let scalar = Scalar::variable(0.5).unwrap();

                    assert_eq!(scalar.is_same_node(&scalar), true);
                }
            }

            mod when_node_differs {
                use super::*;

                #[test]
                fn it_returns_false() {
                    let scalar1 = Scalar::variable(0.5).unwrap();
                    let scalar2 = Scalar::variable(0.5).unwrap();

                    assert_eq!(scalar1.is_same_node(&scalar2), false);
                }
            }
        }

        mod fn_add {
            use super::*;

            #[test]
            fn it_computes_addition_operation_scalar() {
                let scalar1 = Scalar::variable(1.5).unwrap();
                let scalar2 = Scalar::constant(2.5).unwrap();
                let result = scalar1.add(&scalar2);

                assert!(result.is_ok());

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar1.value() + scalar2.value()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Add
                );

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    1.0
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1].local_derivative,
                    1.0
                );

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar1.node.as_ptr()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1]
                        .parent
                        .node
                        .as_ptr(),
                    scalar2.node.as_ptr()
                );

                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_mul {
            use super::*;

            #[test]
            fn it_computes_multiplication_operation_scalar() {
                let scalar1 = Scalar::variable(1.5).unwrap();
                let scalar2 = Scalar::constant(2.5).unwrap();
                let result = scalar1.mul(&scalar2);

                assert!(result.is_ok());

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar1.value() * scalar2.value()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Multiply
                );

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    scalar2.value()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1].local_derivative,
                    scalar1.value()
                );

                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar1.node.as_ptr()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1]
                        .parent
                        .node
                        .as_ptr(),
                    scalar2.node.as_ptr()
                );

                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_neg {
            use super::*;

            #[test]
            fn it_computes_multiplication_by_minus_one_operation_scalar() {
                let scalar = Scalar::variable(1.5).unwrap();
                let result = scalar.neg();

                assert!(result.is_ok());
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar.value() * -1.0
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Negate
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    -1.0
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar.node.as_ptr()
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_sub {
            use super::*;

            #[test]
            fn it_computes_subtract_operation_scalar() {
                let scalar1 = Scalar::variable(1.5).unwrap();
                let scalar2 = Scalar::constant(2.5).unwrap();
                let result = scalar1.sub(&scalar2);

                assert!(result.is_ok());
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar1.value() - scalar2.value()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Subtract
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    1.0
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1].local_derivative,
                    -1.0
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar1.node.as_ptr()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[1]
                        .parent
                        .node
                        .as_ptr(),
                    scalar2.node.as_ptr()
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_exp {
            use super::*;

            #[test]
            fn it_computes_exponential_operation_scalar() {
                let scalar = Scalar::variable(1.5).unwrap();
                let result = scalar.exp();

                assert!(result.is_ok());
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar.value().exp()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Exp
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    scalar.value().exp()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar.node.as_ptr()
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_tanh {
            use super::*;

            #[test]
            fn it_computes_tanh_operation_scalar() {
                let scalar = Scalar::variable(1.5).unwrap();
                let result = scalar.tanh();

                assert!(result.is_ok());
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().value,
                    scalar.value().tanh()
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().operation,
                    ScalarOperation::Tanh
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0].local_derivative,
                    1.0 - scalar.value().tanh().powf(2.0)
                );
                assert_eq!(
                    result.as_ref().unwrap().node.borrow().parents[0]
                        .parent
                        .node
                        .as_ptr(),
                    scalar.node.as_ptr()
                );
                assert_eq!(result.as_ref().unwrap().node.borrow().gradient, Some(0.0));
            }
        }

        mod fn_detach {
            use super::*;

            #[test]
            fn it_computes_detached_scalar() {
                let parents = dummy_parents();
                let value = 1.5;
                let operation = ScalarOperation::Exp;
                let tracked = true;
                let result = Scalar::new_node(value, operation, parents, tracked).detach();

                assert_eq!(result.node.borrow().value, value);
                assert_eq!(result.node.borrow().operation, ScalarOperation::Detached);
                assert_eq!(result.node.borrow().parents.len(), 0);
                assert_eq!(result.node.borrow().gradient, None);
            }
        }

        mod fn_topology {
            use super::*;

            #[test]
            fn it_builds_a_chain_of_scalar_operations() {
                let scalar1 = Scalar::variable(1.0).unwrap();
                let scalar2 = Scalar::variable(2.0).unwrap();
                let scalar3 = Scalar::constant(4.0).unwrap();
                let final_scalar = scalar1.add(&scalar2).unwrap().mul(&scalar3).unwrap();
                let result = final_scalar.topology();

                assert_eq!(result.len(), 5);

                assert_eq!(result[0].value(), 1.0);
                assert_eq!(result[0].operation(), ScalarOperation::Variable);

                assert_eq!(result[1].value(), 2.0);
                assert_eq!(result[1].operation(), ScalarOperation::Variable);

                assert_eq!(result[2].value(), 3.0);
                assert_eq!(result[2].operation(), ScalarOperation::Add);

                assert_eq!(result[3].value(), 4.0);
                assert_eq!(result[3].operation(), ScalarOperation::Constant);

                assert_eq!(result[4].value(), 12.0);
                assert_eq!(result[4].operation(), ScalarOperation::Multiply);
            }
        }

        mod fn_backward_with_seed {
            use super::*;

            mod when_scalar_is_not_trackable {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let parents = dummy_parents();
                    let value = 1.5;
                    let operation = ScalarOperation::Exp;
                    let tracked = false;
                    let scalar = Scalar::new_node(value, operation, parents, tracked);
                    let result = scalar.backward_with_seed(1.0);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err(),
                        Some(ScalarAutodiffError::UntrackedOutput { operation })
                    );
                }
            }

            mod when_seed_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let scalar = Scalar::variable(1.0).unwrap();
                    let result = scalar.backward_with_seed(f64::INFINITY);

                    assert!(result.is_err());
                    assert_eq!(
                        result.err(),
                        Some(ScalarAutodiffError::NonFiniteSeed {
                            seed: f64::INFINITY
                        })
                    );
                }
            }

            mod when_scalar_is_ok {
                use super::*;
                use crate::support::gradcheck::scalar_gradient_check;

                #[test]
                fn it_calculates_gradients_of_itself_and_all_parents() {
                    let scalar1 = Scalar::variable(1.0).unwrap();
                    let scalar2 = Scalar::variable(2.0).unwrap();
                    let scalar3 = Scalar::constant(4.0).unwrap();
                    let scalar4 = Scalar::variable(13.0).unwrap();

                    let addition_scalar = scalar1.add(&scalar2).unwrap();
                    let mul_scalar = addition_scalar.mul(&scalar3).unwrap();
                    let sub_scalar = mul_scalar.sub(&scalar4).unwrap();
                    // Important step: we reuse sub_scalar twice to create two edges from it to
                    // another_add_scalar. This way we double sub_scalar contribution into
                    // another_add_scalar
                    let another_add_scalar = sub_scalar.add(&sub_scalar).unwrap();
                    let exp_scalar = another_add_scalar.exp().unwrap();
                    let tanh_scalar = exp_scalar.tanh().unwrap();

                    let seed = 1.0;
                    let result = tanh_scalar.backward_with_seed(seed).unwrap();

                    let lhs_d = |scalar: &Scalar| scalar.node.borrow().parents[0].local_derivative;
                    let rhs_d = |scalar: &Scalar| scalar.node.borrow().parents[1].local_derivative;

                    assert_eq!(tanh_scalar.gradient(), Some(seed));
                    assert_eq!(
                        exp_scalar.gradient(),
                        Some(tanh_scalar.gradient().unwrap() * lhs_d(&tanh_scalar))
                    );
                    assert_eq!(
                        another_add_scalar.gradient(),
                        Some(exp_scalar.gradient().unwrap() * lhs_d(&exp_scalar))
                    );
                    assert_eq!(
                        sub_scalar.gradient(),
                        Some(
                            another_add_scalar.gradient().unwrap() * lhs_d(&another_add_scalar)
                                + another_add_scalar.gradient().unwrap()
                                    * rhs_d(&another_add_scalar)
                        )
                    );
                    assert_eq!(
                        mul_scalar.gradient(),
                        Some(sub_scalar.gradient().unwrap() * lhs_d(&sub_scalar))
                    );
                    assert_eq!(
                        addition_scalar.gradient(),
                        Some(mul_scalar.gradient().unwrap() * lhs_d(&mul_scalar))
                    );

                    assert_eq!(
                        scalar4.gradient(),
                        Some(sub_scalar.gradient().unwrap() * rhs_d(&sub_scalar))
                    );
                    // scalar3 is a constant, thus it does not have gradient calculated
                    assert_eq!(scalar3.gradient(), None);
                    assert_eq!(
                        scalar2.gradient(),
                        Some(addition_scalar.gradient().unwrap() * rhs_d(&addition_scalar))
                    );
                    assert_eq!(
                        scalar1.gradient(),
                        Some(addition_scalar.gradient().unwrap() * lhs_d(&addition_scalar))
                    );

                    let check = scalar_gradient_check(
                        1.0,
                        scalar1.gradient().unwrap(),
                        1.0e-5,
                        1.0e-9,
                        |value| {
                            let val = (value + 2.0) * 4.0 - 13.0;
                            (val + val).exp().tanh()
                        },
                    )
                    .unwrap();

                    assert!(check.comparison.passed);
                    assert!(check.comparison.scaled_error < 1.0e-9);

                    assert_eq!(
                        result,
                        BackwardPass {
                            seed: 1.0,
                            nodes: vec![
                                BackwardNode {
                                    topology_index: 0,
                                    operation: ScalarOperation::Variable,
                                    value: 1.0,
                                    tracked: true,
                                    pass_adjoin: Some(1.063091892139229),
                                    accumulated_gradient: Some(1.063091892139229)
                                },
                                BackwardNode {
                                    topology_index: 1,
                                    operation: ScalarOperation::Variable,
                                    value: 2.0,
                                    tracked: true,
                                    pass_adjoin: Some(1.063091892139229),
                                    accumulated_gradient: Some(1.063091892139229)
                                },
                                BackwardNode {
                                    topology_index: 2,
                                    operation: ScalarOperation::Add,
                                    value: 3.0,
                                    tracked: true,
                                    pass_adjoin: Some(1.063091892139229),
                                    accumulated_gradient: Some(1.063091892139229)
                                },
                                BackwardNode {
                                    topology_index: 3,
                                    operation: ScalarOperation::Constant,
                                    value: 4.0,
                                    tracked: false,
                                    pass_adjoin: None,
                                    accumulated_gradient: None
                                },
                                BackwardNode {
                                    topology_index: 4,
                                    operation: ScalarOperation::Multiply,
                                    value: 12.0,
                                    tracked: true,
                                    pass_adjoin: Some(0.26577297303480724),
                                    accumulated_gradient: Some(0.26577297303480724)
                                },
                                BackwardNode {
                                    topology_index: 5,
                                    operation: ScalarOperation::Variable,
                                    value: 13.0,
                                    tracked: true,
                                    pass_adjoin: Some(-0.26577297303480724),
                                    accumulated_gradient: Some(-0.26577297303480724)
                                },
                                BackwardNode {
                                    topology_index: 6,
                                    operation: ScalarOperation::Subtract,
                                    value: -1.0,
                                    tracked: true,
                                    pass_adjoin: Some(0.26577297303480724),
                                    accumulated_gradient: Some(0.26577297303480724)
                                },
                                BackwardNode {
                                    topology_index: 7,
                                    operation: ScalarOperation::Add,
                                    value: -2.0,
                                    tracked: true,
                                    pass_adjoin: Some(0.13288648651740362),
                                    accumulated_gradient: Some(0.13288648651740362)
                                },
                                BackwardNode {
                                    topology_index: 8,
                                    operation: ScalarOperation::Exp,
                                    value: 0.1353352832366127,
                                    tracked: true,
                                    pass_adjoin: Some(0.9819057036668868),
                                    accumulated_gradient: Some(0.9819057036668868)
                                },
                                BackwardNode {
                                    topology_index: 9,
                                    operation: ScalarOperation::Tanh,
                                    value: 0.1345150412894902,
                                    tracked: true,
                                    pass_adjoin: Some(1.0),
                                    accumulated_gradient: Some(1.0)
                                }
                            ],
                            edges: vec![
                                BackwardEdge {
                                    reverse_index: 0,
                                    child: 9,
                                    parent: 8,
                                    operand: 0,
                                    local_derivative: 0.9819057036668868,
                                    upstream: 1.0,
                                    contribution: 0.9819057036668868,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(0.9819057036668868)
                                },
                                BackwardEdge {
                                    reverse_index: 1,
                                    child: 8,
                                    parent: 7,
                                    operand: 0,
                                    local_derivative: 0.1353352832366127,
                                    upstream: 0.9819057036668868,
                                    contribution: 0.13288648651740362,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(0.13288648651740362)
                                },
                                BackwardEdge {
                                    reverse_index: 2,
                                    child: 7,
                                    parent: 6,
                                    operand: 0,
                                    local_derivative: 1.0,
                                    upstream: 0.13288648651740362,
                                    contribution: 0.13288648651740362,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(0.13288648651740362)
                                },
                                BackwardEdge {
                                    reverse_index: 3,
                                    child: 7,
                                    parent: 6,
                                    operand: 1,
                                    local_derivative: 1.0,
                                    upstream: 0.13288648651740362,
                                    contribution: 0.13288648651740362,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.13288648651740362),
                                    parent_adjoin_after: Some(0.26577297303480724)
                                },
                                BackwardEdge {
                                    reverse_index: 4,
                                    child: 6,
                                    parent: 4,
                                    operand: 0,
                                    local_derivative: 1.0,
                                    upstream: 0.26577297303480724,
                                    contribution: 0.26577297303480724,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(0.26577297303480724)
                                },
                                BackwardEdge {
                                    reverse_index: 5,
                                    child: 6,
                                    parent: 5,
                                    operand: 1,
                                    local_derivative: -1.0,
                                    upstream: 0.26577297303480724,
                                    contribution: -0.26577297303480724,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(-0.26577297303480724)
                                },
                                BackwardEdge {
                                    reverse_index: 6,
                                    child: 4,
                                    parent: 2,
                                    operand: 0,
                                    local_derivative: 4.0,
                                    upstream: 0.26577297303480724,
                                    contribution: 1.063091892139229,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(1.063091892139229)
                                },
                                BackwardEdge {
                                    reverse_index: 7,
                                    child: 4,
                                    parent: 3,
                                    operand: 1,
                                    local_derivative: 3.0,
                                    upstream: 0.26577297303480724,
                                    contribution: 0.7973189191044217,
                                    parent_tracked: false,
                                    parent_adjoint_before: None,
                                    parent_adjoin_after: None
                                },
                                BackwardEdge {
                                    reverse_index: 8,
                                    child: 2,
                                    parent: 0,
                                    operand: 0,
                                    local_derivative: 1.0,
                                    upstream: 1.063091892139229,
                                    contribution: 1.063091892139229,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(1.063091892139229)
                                },
                                BackwardEdge {
                                    reverse_index: 9,
                                    child: 2,
                                    parent: 1,
                                    operand: 1,
                                    local_derivative: 1.0,
                                    upstream: 1.063091892139229,
                                    contribution: 1.063091892139229,
                                    parent_tracked: true,
                                    parent_adjoint_before: Some(0.0),
                                    parent_adjoin_after: Some(1.063091892139229)
                                }
                            ]
                        }
                    );
                }
            }
        }
    }
}
