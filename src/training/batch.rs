//! Deterministic mini-batches of complete causal windows.
//!
//! Documents remain separate until [`CausalWindowConfig`] has selected every
//! complete shifted pair. An epoch shuffles lightweight window descriptors,
//! never raw token streams, so neither document nor partition boundaries can
//! be crossed by batching. Each selected input and target occurrence is copied
//! directly from its document into its final batch storage.

use std::error::Error;
use std::fmt;

use crate::corpus::Partition;
use crate::data::{CausalWindowConfig, EncodedDocument};
use crate::nn::init::SplitMix64;

/// A rejected batching or token-normalization request.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchError {
    ZeroBatchSize,
    EmptyDocumentId,
    PartitionMismatch {
        document_index: usize,
        expected: Partition,
        actual: Partition,
    },
    DuplicateDocumentId {
        id: String,
        first: usize,
        repeated: usize,
    },
    WindowCountOverflow,
    TokenCountOverflow,
    AllocationFailed {
        elements: usize,
    },
    ContributionCountMismatch {
        expected: usize,
        actual: usize,
    },
    ZeroGradientWidth,
    GradientWidthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteLoss {
        value: f64,
    },
    NonFiniteGradient {
        coordinate: usize,
        value: f64,
    },
    NonFiniteAccumulation {
        quantity: &'static str,
        coordinate: Option<usize>,
        value: f64,
    },
    EmptyAccumulator,
}

impl fmt::Display for BatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatchSize => formatter.write_str("batch size must be positive"),
            Self::EmptyDocumentId => formatter.write_str("batch document ID must not be empty"),
            Self::PartitionMismatch {
                document_index,
                expected,
                actual,
            } => write!(
                formatter,
                "document {document_index} belongs to {actual:?}, expected {expected:?}"
            ),
            Self::DuplicateDocumentId {
                id,
                first,
                repeated,
            } => write!(
                formatter,
                "document ID {id:?} first appears at index {first} and repeats at index {repeated}"
            ),
            Self::WindowCountOverflow => {
                formatter.write_str("total causal-window count does not fit usize")
            }
            Self::TokenCountOverflow => {
                formatter.write_str("admitted target-token count does not fit usize")
            }
            Self::AllocationFailed { elements } => {
                write!(
                    formatter,
                    "could not reserve storage for {elements} batch elements"
                )
            }
            Self::ContributionCountMismatch { expected, actual } => write!(
                formatter,
                "batch needs {expected} token contributions, received {actual}"
            ),
            Self::ZeroGradientWidth => {
                formatter.write_str("a token gradient must have at least one coordinate")
            }
            Self::GradientWidthMismatch { expected, actual } => write!(
                formatter,
                "token gradient width {actual} does not match expected width {expected}"
            ),
            Self::NonFiniteLoss { value } => {
                write!(formatter, "token loss must be finite, received {value:?}")
            }
            Self::NonFiniteGradient { coordinate, value } => write!(
                formatter,
                "token gradient coordinate {coordinate} must be finite, received {value:?}"
            ),
            Self::NonFiniteAccumulation {
                quantity,
                coordinate,
                value,
            } => match coordinate {
                Some(coordinate) => write!(
                    formatter,
                    "{quantity} accumulation at coordinate {coordinate} became non-finite: {value:?}"
                ),
                None => write!(
                    formatter,
                    "{quantity} accumulation became non-finite: {value:?}"
                ),
            },
            Self::EmptyAccumulator => {
                formatter.write_str("cannot average zero admitted target tokens")
            }
        }
    }
}

impl Error for BatchError {}

/// The stable order used for one materialized epoch
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BatchOrder {
    /// Retain document order, then increasing window start within each document
    Sequential,
    /// Apply a deterministic Fisher-Yates permutation using the supplied seed
    Shuffled { seed: u64 },
}

/// A positive requested batch width plus its epoch-order policy
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MiniBatchConfig {
    batch_size: usize,
    order: BatchOrder,
}

impl MiniBatchConfig {
    pub fn new(batch_size: usize, order: BatchOrder) -> Result<Self, BatchError> {
        if batch_size == 0 {
            return Err(BatchError::ZeroBatchSize);
        }
        Ok(Self { batch_size, order })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn order(&self) -> &BatchOrder {
        &self.order
    }
}

/// One separately owned document exposed to the batch builder by reference
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BatchDocument<'a> {
    id: &'a str,
    partition: Partition,
    token_ids: &'a [u32],
}

impl<'a> BatchDocument<'a> {
    pub fn new(
        id: &'a str,
        partition: Partition,
        token_ids: &'a [u32],
    ) -> Result<Self, BatchError> {
        if id.is_empty() {
            return Err(BatchError::EmptyDocumentId);
        }

        Ok(Self {
            id,
            partition,
            token_ids,
        })
    }

    /// Borrows the already-validated provenance and token IDs of one encoded document
    pub fn from_encoded(document: &'a EncodedDocument) -> Self {
        Self {
            id: document.id(),
            partition: document.partition(),
            token_ids: document.token_ids(),
        }
    }

    pub fn id(&self) -> &'a str {
        self.id
    }

    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
}

/// The immutable origin of one complete causal window
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowProvenance {
    partition: Partition,
    document_index: usize,
    document_id: String,
    start: usize,
}

impl WindowProvenance {
    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn document_index(&self) -> usize {
        self.document_index
    }

    pub fn document_id(&self) -> &str {
        &self.document_id
    }

    pub fn start(&self) -> usize {
        self.start
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct WindowDescriptor {
    document_index: usize,
    start: usize,
}

/// One target token's scalar loss and parameter-gradient coordinates
#[derive(Clone, Debug, PartialEq)]
pub struct TokenContribution {
    loss: f64,
    gradient: Vec<f64>,
}

impl TokenContribution {
    pub fn new(loss: f64, gradient: Vec<f64>) -> Result<Self, BatchError> {
        if !loss.is_finite() {
            return Err(BatchError::NonFiniteLoss { value: loss });
        }
        if gradient.is_empty() {
            return Err(BatchError::ZeroGradientWidth);
        }
        if let Some((coordinate, &value)) = gradient
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BatchError::NonFiniteGradient { coordinate, value });
        }

        Ok(Self { loss, gradient })
    }

    pub fn loss(&self) -> f64 {
        self.loss
    }

    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    pub fn gradient_width(&self) -> usize {
        self.gradient.len()
    }
}

fn validate_gradient_sum(left: &[f64], right: &[f64]) -> Result<(), BatchError> {
    debug_assert_eq!(left.len(), right.len());

    for (coordinate, (&left, &right)) in left.iter().zip(right).enumerate() {
        let value = left + right;
        if !value.is_finite() {
            return Err(BatchError::NonFiniteAccumulation {
                quantity: "gradient",
                coordinate: Some(coordinate),
                value,
            });
        }
    }
    Ok(())
}

/// One scalar mean loss and one equally normalized parameter-gradient vector
#[derive(Clone, Debug, PartialEq)]
pub struct TokenMean {
    token_count: usize,
    mean_loss: f64,
    mean_gradient: Vec<f64>,
}

impl TokenMean {
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn mean_loss(&self) -> f64 {
        self.mean_loss
    }

    pub fn mean_gradient(&self) -> &[f64] {
        &self.mean_gradient
    }
}

/// Raw sums that can be merged before one final division by token count
#[derive(Clone, Debug, PartialEq)]
pub struct TokenMeanAccumulator {
    loss_sum: f64,
    gradient_sums: Vec<f64>,
    token_count: usize,
}

impl TokenMeanAccumulator {
    pub fn new(gradient_width: usize) -> Result<Self, BatchError> {
        if gradient_width == 0 {
            return Err(BatchError::ZeroGradientWidth);
        }
        let mut gradient_sums = Vec::new();
        gradient_sums
            .try_reserve_exact(gradient_width)
            .map_err(|_| BatchError::AllocationFailed {
                elements: gradient_width,
            })?;
        gradient_sums.resize(gradient_width, 0.0);

        Ok(Self {
            loss_sum: 0.0,
            gradient_sums,
            token_count: 0,
        })
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn loss_sum(&self) -> f64 {
        self.loss_sum
    }

    pub fn gradient_sums(&self) -> &[f64] {
        &self.gradient_sums
    }

    pub fn add_token(&mut self, contribution: &TokenContribution) -> Result<(), BatchError> {
        if contribution.gradient_width() != self.gradient_sums.len() {
            return Err(BatchError::GradientWidthMismatch {
                expected: self.gradient_sums.len(),
                actual: contribution.gradient_width(),
            });
        }
        let next_count = self
            .token_count
            .checked_add(1)
            .ok_or(BatchError::TokenCountOverflow)?;
        let next_loss = self.loss_sum + contribution.loss();
        if !next_loss.is_finite() {
            return Err(BatchError::NonFiniteAccumulation {
                quantity: "loss",
                coordinate: None,
                value: next_loss,
            });
        }
        validate_gradient_sum(&self.gradient_sums, contribution.gradient())?;

        self.loss_sum = next_loss;
        for (sum, &value) in self.gradient_sums.iter_mut().zip(contribution.gradient()) {
            *sum += value;
        }
        self.token_count = next_count;
        Ok(())
    }

    pub fn finish(self) -> Result<TokenMean, BatchError> {
        if self.token_count == 0 {
            return Err(BatchError::EmptyAccumulator);
        }
        let denominator = self.token_count as f64;
        let mean_loss = self.loss_sum / denominator;
        let mut mean_gradient = self.gradient_sums;
        for value in &mut mean_gradient {
            *value /= denominator;
        }

        Ok(TokenMean {
            token_count: self.token_count,
            mean_loss,
            mean_gradient,
        })
    }
}

/// One row-major `[batch, sequence]` stack with no padding rows or token IDs
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniBatch {
    partition: Partition,
    context_length: usize,
    inputs: Vec<u32>,
    targets: Vec<u32>,
    provenance: Vec<WindowProvenance>,
}

impl MiniBatch {
    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn batch_width(&self) -> usize {
        self.provenance.len()
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.batch_width(), self.context_length]
    }

    pub fn token_count(&self) -> usize {
        self.targets.len()
    }

    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }

    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    pub fn provenance(&self) -> &[WindowProvenance] {
        &self.provenance
    }

    pub fn input_row(&self, row: usize) -> Option<&[u32]> {
        let start = row.checked_mul(self.context_length)?;
        let end = start.checked_add(self.context_length)?;
        self.inputs.get(start..end)
    }

    pub fn target_row(&self, row: usize) -> Option<&[u32]> {
        let start = row.checked_mul(self.context_length)?;
        let end = start.checked_add(self.context_length)?;
        self.targets.get(start..end)
    }

    /// Averages one checked loss and parameter-gradient vector per target token
    pub fn average_token_contributions(
        &self,
        contributions: &[TokenContribution],
    ) -> Result<TokenMean, BatchError> {
        let expected = self.token_count();
        if contributions.len() != expected {
            return Err(BatchError::ContributionCountMismatch {
                expected,
                actual: contributions.len(),
            });
        }

        let gradient_width = contributions
            .first()
            .map(TokenContribution::gradient_width)
            .ok_or(BatchError::EmptyAccumulator)?;
        let mut accumulator = TokenMeanAccumulator::new(gradient_width)?;
        for contribution in contributions {
            accumulator.add_token(contribution)?;
        }
        accumulator.finish()
    }
}

fn validate_documents(
    partition: Partition,
    documents: &[BatchDocument<'_>],
) -> Result<(), BatchError> {
    for (document_index, document) in documents.iter().enumerate() {
        if document.partition() != partition {
            return Err(BatchError::PartitionMismatch {
                document_index,
                expected: partition,
                actual: document.partition(),
            });
        }

        if let Some(first) = documents[..document_index]
            .iter()
            .position(|candidate| candidate.id() == document.id())
        {
            return Err(BatchError::DuplicateDocumentId {
                id: document.id().to_owned(),
                first,
                repeated: document_index,
            });
        }
    }

    Ok(())
}

fn sample_below(rng: &mut SplitMix64, exclusive_upper: usize) -> usize {
    debug_assert!(exclusive_upper > 0);

    let bound = exclusive_upper as u64;
    let rejection_threshold = bound.wrapping_neg() % bound;
    loop {
        let draw = rng.next_u64();
        if draw >= rejection_threshold {
            return (draw % bound) as usize;
        }
    }
}

fn fisher_yates<T>(values: &mut [T], rng: &mut SplitMix64) {
    for upper_index in (1..values.len()).rev() {
        let selected = sample_below(rng, upper_index + 1);
        values.swap(upper_index, selected);
    }
}

/// Every mini-batch is one reproducible traversal of complete windows
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniBatchEpoch {
    partition: Partition,
    context_length: usize,
    config: MiniBatchConfig,
    window_count: usize,
    shuffle_state_after: Option<u64>,
    batches: Vec<MiniBatch>,
}

impl MiniBatchEpoch {
    /// Materializes one reproducible epoch of complete causal training windows.
    ///
    /// The builder first enumerates every complete causal window admitted by
    /// `window_config` across the supplied documents. Each window is created entirely
    /// within one source document and never crosses a document boundary.
    ///
    /// For a source fragment
    ///
    /// ```text
    /// [t0, t1, t2, ..., tN]
    /// ```
    ///
    /// with `context_length = N`, one materialized training row is:
    ///
    /// ```text
    /// input:  [t0, t1, ..., tN-1]
    /// target: [t1, t2, ..., tN]
    /// ```
    ///
    /// The resulting window descriptors are then ordered according to `config`:
    ///
    /// - [`BatchOrder::Sequential`] preserves the enumeration order of complete
    ///   windows: document order first, then increasing window start within each
    ///   document.
    /// - [`BatchOrder::Shuffled`] applies a deterministic Fisher-Yates permutation
    ///   driven by [`SplitMix64`] and the supplied seed.
    ///
    /// `Sequential` refers only to the ordering of training windows. It does not
    /// require every mini-batch to contain rows from a single document. If the
    /// remaining rows from one document do not fill the current mini-batch, complete
    /// windows from the next document may occupy the remaining rows.
    ///
    /// For example, with `batch_size = 2`:
    ///
    /// ```text
    /// row 0: document "foo", start 0
    /// row 1: document "bar", start 0
    /// ```
    ///
    /// may form one mini-batch. These rows remain independent training examples:
    /// tokens are never concatenated across documents, and no causal window is ever
    /// constructed from the end of one document and the beginning of another.
    ///
    /// Ordered windows are grouped into mini-batches of at most
    /// `config.batch_size()` rows. The final mini-batch may contain fewer rows, but
    /// every row is always complete: each contributes exactly `context_length` input
    /// tokens and exactly `context_length` target tokens.
    ///
    /// Inputs and targets are stored row-major in flat vectors, so every constructed
    /// [`MiniBatch`] satisfies:
    ///
    /// ```text
    /// inputs.len()     == batch_width * context_length
    /// targets.len()    == batch_width * context_length
    /// provenance.len() == batch_width
    /// ```
    ///
    /// Thus a smaller final mini-batch means fewer complete rows, never a partially
    /// filled row or an ignored trailing portion of a materialized row.
    ///
    /// Provenance is retained for every row so each training example can be traced
    /// back to its partition, source document, and causal-window start offset.
    ///
    /// When shuffled ordering is requested, [`MiniBatchEpoch::shuffle_state_after`]
    /// records the PRNG state after the deterministic permutation has completed.
    pub fn build(
        partition: Partition,
        documents: &[BatchDocument<'_>],
        window_config: CausalWindowConfig,
        config: MiniBatchConfig,
    ) -> Result<Self, BatchError> {
        validate_documents(partition, documents)?;

        let mut window_count: usize = 0;
        for document in documents {
            window_count = window_count
                .checked_add(window_config.window_count(document.token_ids().len()))
                .ok_or(BatchError::WindowCountOverflow)?;
        }

        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(window_count)
            .map_err(|_| BatchError::AllocationFailed {
                elements: window_count,
            })?;
        for (document_index, document) in documents.iter().copied().enumerate() {
            for window in window_config.windows(document.token_ids()) {
                descriptors.push(WindowDescriptor {
                    document_index,
                    start: window.start(),
                })
            }
        }
        debug_assert_eq!(descriptors.len(), window_count);

        let shuffle_state_after = match config.order() {
            BatchOrder::Sequential => None,
            BatchOrder::Shuffled { seed } => {
                let mut rng = SplitMix64::from_seed(*seed);
                fisher_yates(&mut descriptors, &mut rng);
                Some(rng.state())
            }
        };

        let batch_count = if window_count == 0 {
            0
        } else {
            (window_count - 1) / config.batch_size() + 1
        };
        let mut batches = Vec::new();
        batches
            .try_reserve_exact(batch_count)
            .map_err(|_| BatchError::AllocationFailed {
                elements: batch_count,
            })?;

        let context_length = window_config.context_length();
        let required_source_tokens = window_config.required_source_tokens();
        let mut descriptors = descriptors.into_iter();
        let mut remaining = window_count;
        while remaining > 0 {
            let width = remaining.min(config.batch_size());
            let token_count = width
                .checked_mul(context_length)
                .ok_or(BatchError::TokenCountOverflow)?;
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            let mut provenance = Vec::new();
            inputs
                .try_reserve_exact(token_count)
                .map_err(|_| BatchError::AllocationFailed {
                    elements: token_count,
                })?;
            targets
                .try_reserve_exact(token_count)
                .map_err(|_| BatchError::AllocationFailed {
                    elements: token_count,
                })?;
            provenance
                .try_reserve_exact(width)
                .map_err(|_| BatchError::AllocationFailed { elements: width })?;

            for _ in 0..width {
                let descriptor = descriptors
                    .next()
                    .expect("pre-counted descriptor must exist while batching");
                let document = documents[descriptor.document_index];
                let source_end = descriptor.start + required_source_tokens;
                let source = document
                    .token_ids()
                    .get(descriptor.start..source_end)
                    .expect("descriptor must name one complete causal window");

                inputs.extend_from_slice(&source[..context_length]);
                targets.extend_from_slice(&source[1..]);
                provenance.push(WindowProvenance {
                    partition,
                    document_index: descriptor.document_index,
                    document_id: document.id().to_owned(),
                    start: descriptor.start,
                });
            }

            debug_assert_eq!(inputs.len(), token_count);
            debug_assert_eq!(targets.len(), token_count);
            debug_assert_eq!(provenance.len(), width);

            batches.push(MiniBatch {
                partition,
                context_length,
                inputs,
                targets,
                provenance,
            });
            remaining -= width;
        }

        Ok(Self {
            partition,
            context_length,
            config,
            window_count,
            shuffle_state_after,
            batches,
        })
    }

    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn config(&self) -> MiniBatchConfig {
        self.config
    }

    pub fn window_count(&self) -> usize {
        self.window_count
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    pub fn shuffle_state_after(&self) -> Option<u64> {
        self.shuffle_state_after
    }

    pub fn batches(&self) -> &[MiniBatch] {
        &self.batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod mini_batch_config {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_batch_size_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let batch_size = 0;
                    let order = BatchOrder::Sequential;
                    let result = MiniBatchConfig::new(batch_size, order);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::ZeroBatchSize);
                }
            }

            mod when_batch_size_gt_0 {
                use super::*;

                #[test]
                fn it_computes_mini_batch_config() {
                    let batch_size = 1;
                    let order = BatchOrder::Sequential;
                    let result = MiniBatchConfig::new(batch_size, order);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.batch_size, batch_size);
                    assert_eq!(result.order, order);
                }
            }
        }
    }

    mod batch_document {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_id_is_empty {
                use super::*;

                #[test]
                fn returns_error() {
                    let id = "";
                    let partition = Partition::Validation;
                    let token_ids = [1, 2, 3];
                    let result = BatchDocument::new(id, partition, &token_ids);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::EmptyDocumentId);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_batch_document() {
                    let id = "foo";
                    let partition = Partition::Validation;
                    let token_ids = [1, 2, 3];
                    let result = BatchDocument::new(id, partition, &token_ids);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.id, id);
                    assert_eq!(result.partition, partition);
                    assert_eq!(result.token_ids, &token_ids);
                }
            }
        }

        mod fn_from_encoded {
            use super::*;

            #[test]
            fn it_computes_batch_document_from_encoded_document() {
                let id = "foo";
                let partition = Partition::Train;
                let token_ids = vec![1, 2, 3];
                let document = EncodedDocument::from_raw_parts(id, partition, token_ids);
                let result = BatchDocument::from_encoded(&document);

                assert_eq!(result.id, document.id());
                assert_eq!(result.partition, document.partition());
                assert_eq!(result.token_ids, document.token_ids());
            }
        }
    }

    mod token_contribution {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_loss_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let loss = f64::NEG_INFINITY;
                    let gradient = vec![1., 2.];
                    let result = TokenContribution::new(loss, gradient);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::NonFiniteLoss { value: loss }
                    );
                }
            }

            mod when_gradient_is_empty {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let loss = 1.0;
                    let gradient = vec![];
                    let result = TokenContribution::new(loss, gradient);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::ZeroGradientWidth);
                }
            }

            mod when_some_gradient_value_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let loss = 2.;
                    let gradient = vec![1., f64::NEG_INFINITY];
                    let result = TokenContribution::new(loss, gradient);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::NonFiniteGradient {
                            coordinate: 1,
                            value: f64::NEG_INFINITY
                        }
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_token_contribution() {
                    let loss = 2.;
                    let gradient = vec![1., 3.];
                    let result = TokenContribution::new(loss, gradient.clone());

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.loss, loss);
                    assert_eq!(result.gradient, gradient);
                }
            }
        }
    }

    mod token_mean_accumulator {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_gradient_width_is_0 {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gradient_width = 0;
                    let result = TokenMeanAccumulator::new(gradient_width);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::ZeroGradientWidth);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_token_mean_accumulator() {
                    let gradient_width = 2;
                    let result = TokenMeanAccumulator::new(gradient_width);

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.loss_sum, 0.);
                    assert_eq!(result.gradient_sums, vec![0.; gradient_width]);
                    assert_eq!(result.token_count, 0);
                }
            }
        }

        mod fn_add_token {
            use super::*;

            mod when_gradient_width_does_not_match {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gradient_width = 2;
                    let mut accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();

                    let loss = 0.;
                    let gradient = vec![1., 2., 3.];
                    let token_contribution =
                        TokenContribution::new(loss, gradient.clone()).unwrap();

                    let result = accumulator.add_token(&token_contribution);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::GradientWidthMismatch {
                            expected: gradient_width,
                            actual: gradient.len(),
                        }
                    );
                }
            }

            mod when_new_loss_sum_becomes_non_infinite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gradient_width = 2;
                    let mut accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();
                    accumulator.loss_sum = f64::MAX - 1.;

                    let loss = f64::MAX - 1.;
                    let gradient = vec![1., 2.];
                    let token_contribution =
                        TokenContribution::new(loss, gradient.clone()).unwrap();

                    let result = accumulator.add_token(&token_contribution);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::NonFiniteAccumulation {
                            quantity: "loss",
                            coordinate: None,
                            value: f64::INFINITY,
                        }
                    );
                }
            }

            mod when_gradients_sum_becomes_non_infinite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gradient_width = 2;
                    let mut accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();
                    accumulator.gradient_sums = vec![2., f64::MAX - 1.];

                    let loss = 0.;
                    let gradient = vec![1., f64::MAX - 1.];
                    let token_contribution =
                        TokenContribution::new(loss, gradient.clone()).unwrap();

                    let result = accumulator.add_token(&token_contribution);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::NonFiniteAccumulation {
                            quantity: "gradient",
                            coordinate: Some(1),
                            value: f64::INFINITY,
                        }
                    );
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_sums_token_gradient_and_loss_contribution() {
                    let gradient_width = 2;
                    let mut accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();
                    accumulator.loss_sum = 5.;
                    accumulator.gradient_sums = vec![2., 3.];
                    accumulator.token_count = 1;

                    let loss = 6.;
                    let gradient = vec![1., 4.];
                    let token_contribution =
                        TokenContribution::new(loss, gradient.clone()).unwrap();

                    let result = accumulator.add_token(&token_contribution);
                    assert!(result.is_ok(), "{result:?}");
                    assert_eq!(accumulator.loss_sum, 5. + 6.);
                    assert_eq!(accumulator.gradient_sums, vec![2. + 1., 3. + 4.]);
                    assert_eq!(accumulator.token_count, 2);
                }
            }
        }

        mod fn_finish {
            use super::*;

            mod when_token_count_is_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let gradient_width = 2;
                    let accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();
                    let result = accumulator.finish();

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::EmptyAccumulator)
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_token_mean() {
                    let gradient_width = 2;
                    let mut accumulator = TokenMeanAccumulator::new(gradient_width).unwrap();

                    let token_contribution1 = TokenContribution::new(6., vec![1., 4.]).unwrap();
                    let token_contribution2 = TokenContribution::new(10., vec![2., 3.]).unwrap();

                    accumulator.add_token(&token_contribution1).unwrap();
                    accumulator.add_token(&token_contribution2).unwrap();

                    let result = accumulator.finish();

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.token_count, 2);
                    assert_eq!(result.mean_loss, (6. + 10.) / 2.);
                    assert_eq!(result.mean_gradient, vec![(1. + 2.) / 2., (4. + 3.) / 2.]);
                }
            }
        }
    }

    mod mini_batch {
        use super::*;

        mod fn_input_row {
            use super::*;

            #[test]
            fn it_fetches_input_row() {
                let partition = Partition::Train;
                let context_length = 2;
                let inputs = vec![1, 2, 3, 4];
                let targets = vec![5, 6, 7, 8];
                let provenance = vec![];
                let mini_batch = MiniBatch {
                    partition,
                    context_length,
                    inputs,
                    targets,
                    provenance,
                };

                assert_eq!(mini_batch.input_row(1), Some(vec![3, 4].as_slice()));
            }
        }

        mod fn_target_row {
            use super::*;

            #[test]
            fn it_fetches_input_row() {
                let partition = Partition::Train;
                let context_length = 2;
                let inputs = vec![1, 2, 3, 4];
                let targets = vec![5, 6, 7, 8];
                let provenance = vec![];
                let mini_batch = MiniBatch {
                    partition,
                    context_length,
                    inputs,
                    targets,
                    provenance,
                };

                assert_eq!(mini_batch.target_row(1), Some(vec![7, 8].as_slice()));
            }
        }

        mod fn_average_token_contributions {
            use super::*;

            mod when_contributions_length_does_not_match_token_count {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let partition = Partition::Train;
                    let context_length = 2;
                    let inputs = vec![];
                    let targets = vec![5, 6, 7, 8];
                    let provenance = vec![];
                    let mini_batch = MiniBatch {
                        partition,
                        context_length,
                        inputs,
                        targets: targets.clone(),
                        provenance,
                    };
                    let contributions = vec![TokenContribution::new(0., vec![1.]).unwrap()];
                    let result = mini_batch.average_token_contributions(contributions.as_slice());

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::ContributionCountMismatch {
                            expected: targets.len(),
                            actual: contributions.len(),
                        }
                    );
                }
            }

            mod when_contributions_is_empty {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let partition = Partition::Train;
                    let context_length = 2;
                    let inputs = vec![];
                    let targets = vec![];
                    let provenance = vec![];
                    let mini_batch = MiniBatch {
                        partition,
                        context_length,
                        inputs,
                        targets,
                        provenance,
                    };
                    let result = mini_batch.average_token_contributions(&[]);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(result.err().unwrap(), BatchError::EmptyAccumulator);
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_computes_token_mean() {
                    let partition = Partition::Train;
                    let context_length = 2;
                    let inputs = vec![];
                    let targets = vec![5, 6];
                    let provenance = vec![];
                    let mini_batch = MiniBatch {
                        partition,
                        context_length,
                        inputs,
                        targets: targets.clone(),
                        provenance,
                    };
                    let contribution1 = TokenContribution::new(1., vec![1., 2.]).unwrap();
                    let contribution2 = TokenContribution::new(2., vec![4., 5.]).unwrap();
                    let contributions = vec![contribution1, contribution2];
                    let result = mini_batch.average_token_contributions(contributions.as_slice());

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.token_count, 2);
                    assert_eq!(result.mean_gradient, vec![2.5, 3.5]);
                    assert_eq!(result.mean_loss, 1.5);
                }
            }
        }
    }

    mod mini_batch_epoch {
        use super::*;

        mod fn_build {
            use super::*;

            mod when_documents_partition_differs_from_the_given_partition {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let partition = Partition::Train;
                    let encoded_doc =
                        EncodedDocument::from_raw_parts("foo", Partition::Test, vec![1]);
                    let batch_doc = BatchDocument::from_encoded(&encoded_doc);
                    let window_config = CausalWindowConfig::new(1, 1).unwrap();
                    let config = MiniBatchConfig::new(1, BatchOrder::Sequential).unwrap();
                    let result =
                        MiniBatchEpoch::build(partition, &[batch_doc], window_config, config);

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::PartitionMismatch {
                            document_index: 0,
                            expected: partition,
                            actual: encoded_doc.partition(),
                        }
                    );
                }
            }

            mod when_document_ids_are_duplicated {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let partition = Partition::Train;
                    let encoded_doc = EncodedDocument::from_raw_parts("foo", partition, vec![1]);
                    let batch_doc = BatchDocument::from_encoded(&encoded_doc);
                    let window_config = CausalWindowConfig::new(1, 1).unwrap();
                    let config = MiniBatchConfig::new(1, BatchOrder::Sequential).unwrap();
                    let result = MiniBatchEpoch::build(
                        partition,
                        &[batch_doc, batch_doc],
                        window_config,
                        config,
                    );

                    assert!(result.is_err(), "{result:?}");
                    assert_eq!(
                        result.err().unwrap(),
                        BatchError::DuplicateDocumentId {
                            id: "foo".to_owned(),
                            first: 0,
                            repeated: 1,
                        }
                    );
                }
            }

            mod when_order_is_sequential {
                use super::*;

                #[test]
                fn it_computes_batches_in_sequential_order() {
                    let partition = Partition::Train;
                    let encoded_doc1 =
                        EncodedDocument::from_raw_parts("foo", partition, vec![1, 2, 3]);
                    let encoded_doc2 =
                        EncodedDocument::from_raw_parts("bar", partition, vec![2, 5, 7, 8]);
                    let window_config = CausalWindowConfig::new(2, 1).unwrap();
                    let config = MiniBatchConfig::new(2, BatchOrder::Sequential).unwrap();
                    let result = MiniBatchEpoch::build(
                        partition,
                        &[
                            BatchDocument::from_encoded(&encoded_doc1),
                            BatchDocument::from_encoded(&encoded_doc2),
                        ],
                        window_config,
                        config,
                    );

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.partition, partition);
                    assert_eq!(result.context_length, window_config.context_length());
                    assert_eq!(result.config, config);
                    assert_eq!(result.window_count, 3);
                    assert_eq!(result.shuffle_state_after, None);
                    assert_eq!(result.batches.len(), 2);
                    // batch[0]
                    assert_eq!(result.batches[0].partition, partition);
                    assert_eq!(
                        result.batches[0].context_length,
                        window_config.context_length()
                    );
                    assert_eq!(result.batches[0].inputs, vec![1, 2, 2, 5]);
                    assert_eq!(result.batches[0].targets, vec![2, 3, 5, 7]);
                    assert_eq!(result.batches[0].provenance.len(), 2);
                    assert_eq!(
                        result.batches[0].provenance[0],
                        WindowProvenance {
                            partition,
                            document_index: 0,
                            document_id: encoded_doc1.id().into(),
                            start: 0
                        }
                    );
                    assert_eq!(
                        result.batches[0].provenance[1],
                        WindowProvenance {
                            partition,
                            document_index: 1,
                            document_id: encoded_doc2.id().into(),
                            start: 0
                        }
                    );
                    // batch[1]
                    assert_eq!(result.batches[1].partition, partition);
                    assert_eq!(
                        result.batches[1].context_length,
                        window_config.context_length()
                    );
                    assert_eq!(result.batches[1].inputs, vec![5, 7]);
                    assert_eq!(result.batches[1].targets, vec![7, 8]);
                    assert_eq!(result.batches[1].provenance.len(), 1);
                    assert_eq!(
                        result.batches[1].provenance[0],
                        WindowProvenance {
                            partition,
                            document_index: 1,
                            document_id: encoded_doc2.id().into(),
                            start: 1
                        }
                    );
                }
            }

            mod when_order_is_shuffled {
                use super::*;

                #[test]
                fn it_computes_batches_in_sequential_order() {
                    let partition = Partition::Train;
                    let encoded_doc1 =
                        EncodedDocument::from_raw_parts("foo", partition, vec![1, 2, 3]);
                    let encoded_doc2 =
                        EncodedDocument::from_raw_parts("bar", partition, vec![2, 5, 7, 8]);
                    let window_config = CausalWindowConfig::new(2, 1).unwrap();
                    let config = MiniBatchConfig::new(2, BatchOrder::Shuffled { seed: 0 }).unwrap();
                    let result = MiniBatchEpoch::build(
                        partition,
                        &[
                            BatchDocument::from_encoded(&encoded_doc1),
                            BatchDocument::from_encoded(&encoded_doc2),
                        ],
                        window_config,
                        config,
                    );

                    assert!(result.is_ok(), "{result:?}");
                    let result = result.as_ref().unwrap();
                    assert_eq!(result.partition, partition);
                    assert_eq!(result.context_length, window_config.context_length());
                    assert_eq!(result.config, config);
                    assert_eq!(result.window_count, 3);
                    assert_eq!(result.shuffle_state_after, Some(4354685564936845354));
                    assert_eq!(result.batches.len(), 2);
                    // batch[0]
                    assert_eq!(result.batches[0].partition, partition);
                    assert_eq!(
                        result.batches[0].context_length,
                        window_config.context_length()
                    );
                    assert_eq!(result.batches[0].inputs, vec![5, 7, 1, 2]);
                    assert_eq!(result.batches[0].targets, vec![7, 8, 2, 3]);
                    assert_eq!(result.batches[0].provenance.len(), 2);
                    assert_eq!(
                        result.batches[0].provenance[0],
                        WindowProvenance {
                            partition,
                            document_index: 1,
                            document_id: encoded_doc2.id().into(),
                            start: 1
                        }
                    );
                    assert_eq!(
                        result.batches[0].provenance[1],
                        WindowProvenance {
                            partition,
                            document_index: 0,
                            document_id: encoded_doc1.id().into(),
                            start: 0
                        }
                    );
                    // batch[1]
                    assert_eq!(result.batches[1].partition, partition);
                    assert_eq!(
                        result.batches[1].context_length,
                        window_config.context_length()
                    );
                    assert_eq!(result.batches[1].inputs, vec![2, 5]);
                    assert_eq!(result.batches[1].targets, vec![5, 7]);
                    assert_eq!(result.batches[1].provenance.len(), 1);
                    assert_eq!(
                        result.batches[1].provenance[0],
                        WindowProvenance {
                            partition,
                            document_index: 1,
                            document_id: encoded_doc2.id().into(),
                            start: 0
                        }
                    );
                }
            }
        }
    }
}
