//! Final-evaluation boundary with caller-asserted roles and identifiers.
//!
//! Corpus, split, tokenizer, selection-role, and fit-role values enter this
//! module as caller assertions. The evaluator checks their shape and mutual
//! consistency; it does not derive data or model lineage from the underlying
//! corpus, tokenizer, selection run, or bigram fit.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::bigram::{BigramError, BigramModel};
use crate::corpus::Partition;
use crate::metrics::{MetricError, score_assigned_probabilities};
use crate::models::decoder::{DecoderModel, DecoderModelConfig};
use crate::training::batch::MiniBatchEpoch;
use crate::training::trainer::{DecoderModelState, TrainerError, evaluate_no_grad};

pub const FINAL_EVALUATION_REPORT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvenanceField {
    Corpus,
    Split,
    Tokenizer,
    Context,
}

impl fmt::Display for ProvenanceField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Corpus => "corpus",
            Self::Split => "split",
            Self::Tokenizer => "tokenizer",
            Self::Context => "context",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluatedModel {
    SelectedDecoder,
    FrozenBigram,
}

impl fmt::Display for EvaluatedModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SelectedDecoder => "selected decoder",
            Self::FrozenBigram => "frozen bigram",
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum EvaluationError {
    EmptyProvenance {
        field: ProvenanceField,
    },
    ZeroContextLength,
    WrongSelectionPartition {
        actual: Partition,
    },
    InvalidSelectionLoss {
        value: f64,
    },
    WrongBaselinePartition {
        actual: Partition,
    },
    UnfittedBigram,
    WrongTestPartition {
        actual: Partition,
    },
    EmptyTestEpoch,
    AlreadyEvaluated,
    ProvenanceMismatch {
        field: ProvenanceField,
    },
    EpochContextMismatch {
        expected: usize,
        actual: usize,
    },
    ModelContextMismatch {
        expected: usize,
        actual: usize,
    },
    VocabularyMismatch {
        decoder: usize,
        bigram: usize,
    },
    TargetAlignmentMismatch {
        batch: usize,
        inputs: usize,
        targets: usize,
    },
    InputTokenOutOfRange {
        batch: usize,
        position: usize,
        id: u32,
        vocabulary_size: usize,
    },
    TargetTokenOutOfRange {
        batch: usize,
        position: usize,
        id: u32,
        vocabulary_size: usize,
    },
    SelectedStateMismatch,
    MissingGradient {
        name: String,
    },
    DecoderParameterChanged,
    DecoderGradientChanged,
    GraphRecorded {
        count: usize,
    },
    TargetCountMismatch {
        expected: usize,
        decoder: usize,
        bigram: usize,
    },
    NonFiniteScore {
        model: EvaluatedModel,
        mean_nll: f64,
    },
    Trainer(TrainerError),
    Metric(MetricError),
    Bigram(BigramError),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProvenance { field } => {
                write!(formatter, "{field} provenance assertion must not be blank")
            }
            Self::ZeroContextLength => {
                formatter.write_str("asserted provenance context length must be positive")
            }
            Self::WrongSelectionPartition { actual } => write!(
                formatter,
                "the caller selection-partition assertion must be Validation, got {actual:?}"
            ),
            Self::InvalidSelectionLoss { value } => write!(
                formatter,
                "caller-supplied selected-validation-loss value must be finite and nonnegative, got {value}"
            ),
            Self::WrongBaselinePartition { actual } => write!(
                formatter,
                "the caller fit-partition assertion must be Train, got {actual:?}"
            ),
            Self::UnfittedBigram => {
                formatter.write_str("the bigram must report at least one fitted document")
            }
            Self::WrongTestPartition { actual } => {
                write!(
                    formatter,
                    "final evaluation requires an epoch labeled Test, got {actual:?}"
                )
            }
            Self::EmptyTestEpoch => formatter.write_str("test epoch must contain targets"),
            Self::AlreadyEvaluated => {
                formatter.write_str("this final evaluator has already opened test data")
            }
            Self::ProvenanceMismatch { field } => {
                write!(formatter, "{field} provenance assertion does not match")
            }
            Self::EpochContextMismatch { expected, actual } => write!(
                formatter,
                "test epoch context length must match asserted evaluation context {expected}, got {actual}"
            ),
            Self::ModelContextMismatch { expected, actual } => write!(
                formatter,
                "decoder context capacity must match asserted evaluation context {expected}, got {actual}"
            ),
            Self::VocabularyMismatch { decoder, bigram } => write!(
                formatter,
                "decoder vocabulary {decoder} does not match bigram vocabulary {bigram}"
            ),
            Self::TargetAlignmentMismatch {
                batch,
                inputs,
                targets,
            } => write!(
                formatter,
                "test batch {batch} has {inputs} inputs but {targets} targets"
            ),
            Self::InputTokenOutOfRange {
                batch,
                position,
                id,
                vocabulary_size,
            } => write!(
                formatter,
                "test batch {batch} input {position} has token {id} outside vocabulary {vocabulary_size}"
            ),
            Self::TargetTokenOutOfRange {
                batch,
                position,
                id,
                vocabulary_size,
            } => write!(
                formatter,
                "test batch {batch} target {position} has token {id} outside vocabulary {vocabulary_size}"
            ),
            Self::SelectedStateMismatch => formatter
                .write_str("the supplied decoder does not match the supplied retained state"),
            Self::MissingGradient { name } => {
                write!(
                    formatter,
                    "decoder parameter {name} has no gradient storage"
                )
            }
            Self::DecoderParameterChanged => {
                formatter.write_str("final evaluation changed decoder parameter bits")
            }
            Self::DecoderGradientChanged => {
                formatter.write_str("final evaluation changed decoder gradient bits")
            }
            Self::GraphRecorded { count } => {
                write!(
                    formatter,
                    "final evaluation recorded {count} gradient graphs"
                )
            }
            Self::TargetCountMismatch {
                expected,
                decoder,
                bigram,
            } => write!(
                formatter,
                "test evidence has {expected} targets, decoder scored {decoder}, and bigram scored {bigram}"
            ),
            Self::NonFiniteScore { model, mean_nll } => {
                write!(formatter, "{model} produced non-finite mean NLL {mean_nll}")
            }
            Self::Trainer(error) => write!(formatter, "decoder evaluation failed: {error}"),
            Self::Metric(error) => write!(formatter, "metric evaluation failed: {error}"),
            Self::Bigram(error) => write!(formatter, "bigram evaluation failed: {error}"),
        }
    }
}

impl Error for EvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Trainer(error) => Some(error),
            Self::Metric(error) => Some(error),
            Self::Bigram(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TrainerError> for EvaluationError {
    fn from(error: TrainerError) -> Self {
        Self::Trainer(error)
    }
}

impl From<MetricError> for EvaluationError {
    fn from(error: MetricError) -> Self {
        Self::Metric(error)
    }
}

impl From<BigramError> for EvaluationError {
    fn from(error: BigramError) -> Self {
        Self::Bigram(error)
    }
}

/// Caller-supplied identifiers and context metadata shared by report participants.
///
/// Construction proves only that the three identifier strings are nonblank and
/// the context length is positive. Equality between values of this type proves
/// only that callers supplied matching assertions; it does not hash or inspect a
/// corpus, split construction, or tokenizer. The evaluator separately checks the
/// context value against the test epoch and decoder capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationProvenance {
    corpus_fingerprint: String,
    split_fingerprint: String,
    tokenizer_fingerprint: String,
    context_length: usize,
}

impl EvaluationProvenance {
    pub fn new(
        corpus_fingerprint: impl Into<String>,
        split_fingerprint: impl Into<String>,
        tokenizer_fingerprint: impl Into<String>,
        context_length: usize,
    ) -> Result<Self, EvaluationError> {
        let provenance = Self {
            corpus_fingerprint: corpus_fingerprint.into(),
            split_fingerprint: split_fingerprint.into(),
            tokenizer_fingerprint: tokenizer_fingerprint.into(),
            context_length,
        };
        for (field, value) in [
            (ProvenanceField::Corpus, provenance.corpus_fingerprint()),
            (ProvenanceField::Split, provenance.split_fingerprint()),
            (
                ProvenanceField::Tokenizer,
                provenance.tokenizer_fingerprint(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(EvaluationError::EmptyProvenance { field });
            }
        }
        if context_length == 0 {
            return Err(EvaluationError::ZeroContextLength);
        }
        Ok(provenance)
    }

    pub fn corpus_fingerprint(&self) -> &str {
        &self.corpus_fingerprint
    }

    pub fn split_fingerprint(&self) -> &str {
        &self.split_fingerprint
    }

    pub fn tokenizer_fingerprint(&self) -> &str {
        &self.tokenizer_fingerprint
    }

    pub const fn context_length(&self) -> usize {
        self.context_length
    }
}

/// A borrowed decoder accompanied by the caller's validation-role assertion.
///
/// Construction checks the asserted role and the numeric shape of the supplied
/// selection values. It cannot reconstruct how `selected_step`,
/// `selected_validation_loss`, or the model were selected. Before test access,
/// the evaluator does mechanically compare the retained state with the borrowed
/// model's exact configuration, names, shapes, and value bits.
#[derive(Clone, Copy, Debug)]
pub struct SelectedDecoder<'a> {
    state: &'a DecoderModelState,
    model: &'a DecoderModel,
    selected_step: usize,
    selected_validation_loss: f64,
    provenance: &'a EvaluationProvenance,
}

impl<'a> SelectedDecoder<'a> {
    pub fn new(
        state: &'a DecoderModelState,
        model: &'a DecoderModel,
        selected_step: usize,
        selected_validation_loss: f64,
        selection_partition_assertion: Partition,
        provenance: &'a EvaluationProvenance,
    ) -> Result<Self, EvaluationError> {
        if selection_partition_assertion != Partition::Validation {
            return Err(EvaluationError::WrongSelectionPartition {
                actual: selection_partition_assertion,
            });
        }
        if !selected_validation_loss.is_finite() || selected_validation_loss < 0.0 {
            return Err(EvaluationError::InvalidSelectionLoss {
                value: selected_validation_loss,
            });
        }
        Ok(Self {
            state,
            model,
            selected_step,
            selected_validation_loss,
            provenance,
        })
    }

    const fn model(self) -> &'a DecoderModel {
        self.model
    }

    const fn state(self) -> &'a DecoderModelState {
        self.state
    }

    pub const fn selected_step(self) -> usize {
        self.selected_step
    }

    pub const fn selected_validation_loss(self) -> f64 {
        self.selected_validation_loss
    }

    pub const fn provenance(self) -> &'a EvaluationProvenance {
        self.provenance
    }
}

/// A borrowed count baseline accompanied by the caller's training-role assertion.
///
/// Construction checks the asserted role and that the model reports at least one
/// fitted document. It cannot discover which documents produced the counts.
#[derive(Clone, Copy, Debug)]
pub struct FrozenBigram<'a> {
    model: &'a BigramModel,
    provenance: &'a EvaluationProvenance,
}

impl<'a> FrozenBigram<'a> {
    pub fn new(
        model: &'a BigramModel,
        fit_partition_assertion: Partition,
        provenance: &'a EvaluationProvenance,
    ) -> Result<Self, EvaluationError> {
        if fit_partition_assertion != Partition::Train {
            return Err(EvaluationError::WrongBaselinePartition {
                actual: fit_partition_assertion,
            });
        }
        if model.fitted_documents() == 0 {
            return Err(EvaluationError::UnfittedBigram);
        }
        Ok(Self { model, provenance })
    }

    pub const fn model(self) -> &'a BigramModel {
        self.model
    }

    pub const fn provenance(self) -> &'a EvaluationProvenance {
        self.provenance
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelScore {
    total_nll: f64,
    target_count: usize,
    mean_nll: f64,
    perplexity: f64,
}

impl ModelScore {
    fn new(
        model: EvaluatedModel,
        total_nll: f64,
        target_count: usize,
        mean_nll: f64,
        perplexity: f64,
    ) -> Result<Self, EvaluationError> {
        if !total_nll.is_finite() || !mean_nll.is_finite() || !perplexity.is_finite() {
            return Err(EvaluationError::NonFiniteScore { model, mean_nll });
        }
        Ok(Self {
            total_nll,
            target_count,
            mean_nll,
            perplexity,
        })
    }

    pub const fn total_nll(self) -> f64 {
        self.total_nll
    }

    pub const fn target_count(self) -> usize {
        self.target_count
    }

    pub const fn mean_nll(self) -> f64 {
        self.mean_nll
    }

    pub const fn perplexity(self) -> f64 {
        self.perplexity
    }
}

/// An owned versioned result with no setters and no model-selection operation.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalEvaluationReport {
    version: u32,
    selected_step: usize,
    selected_validation_loss: f64,
    provenance: EvaluationProvenance,
    test_document_ids: Vec<String>,
    window_slot_fingerprint: String,
    window_count: usize,
    batch_count: usize,
    window_target_slot_count: usize,
    document_transition_occurrence_count: usize,
    transition_multiplicity_counts: Vec<usize>,
    decoder: ModelScore,
    bigram: ModelScore,
    access_count: u8,
    recorded_graphs: usize,
    parameters_unchanged: bool,
    gradients_unchanged: bool,
}

impl FinalEvaluationReport {
    pub const fn version(&self) -> u32 {
        self.version
    }

    pub const fn selected_step(&self) -> usize {
        self.selected_step
    }

    pub const fn selected_validation_loss(&self) -> f64 {
        self.selected_validation_loss
    }

    pub const fn provenance(&self) -> &EvaluationProvenance {
        &self.provenance
    }

    pub fn test_document_ids(&self) -> &[String] {
        &self.test_document_ids
    }

    pub fn target_fingerprint(&self) -> &str {
        self.window_slot_fingerprint()
    }

    /// Fingerprints the ordered overlapping window-target slots that both models score.
    pub fn window_slot_fingerprint(&self) -> &str {
        &self.window_slot_fingerprint
    }

    pub const fn window_count(&self) -> usize {
        self.window_count
    }

    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub const fn target_count(&self) -> usize {
        self.window_target_slot_count()
    }

    /// Counts the overlapping window-target slots in the reported mean.
    pub const fn window_target_slot_count(&self) -> usize {
        self.window_target_slot_count
    }

    /// Counts document-position occurrences represented by the slots, without overlap repetition.
    pub const fn document_transition_occurrence_count(&self) -> usize {
        self.document_transition_occurrence_count
    }

    /// At index `m - 1`, stores how many transition occurrences appear in `m` slots.
    pub fn transition_multiplicity_counts(&self) -> &[usize] {
        &self.transition_multiplicity_counts
    }

    pub const fn decoder(&self) -> ModelScore {
        self.decoder
    }

    pub const fn bigram(&self) -> ModelScore {
        self.bigram
    }

    pub const fn access_count(&self) -> u8 {
        self.access_count
    }

    pub const fn recorded_graphs(&self) -> usize {
        self.recorded_graphs
    }

    pub const fn parameters_unchanged(&self) -> bool {
        self.parameters_unchanged
    }

    pub const fn gradients_unchanged(&self) -> bool {
        self.gradients_unchanged
    }

    pub fn loss_gap(&self) -> f64 {
        self.bigram.mean_nll - self.decoder.mean_nll
    }

    pub fn decoder_has_lower_loss(&self) -> bool {
        self.decoder.mean_nll < self.bigram.mean_nll
    }
}

#[derive(Debug)]
struct TestEvidence {
    document_ids: Vec<String>,
    window_slot_fingerprint: String,
    window_target_slot_count: usize,
    document_transition_occurrence_count: usize,
    transition_multiplicity_counts: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CheckedTokenPair {
    input: usize,
    target: usize,
}

/// One gate-opening inspection of a borrowed test epoch.
///
/// This type is private to the module. Current callers construct it only through
/// `inspect`; it exposes no mutation API and retains an immutable epoch borrow.
#[derive(Debug)]
struct InspectedTestEpoch<'a> {
    epoch: &'a MiniBatchEpoch,
    evidence: TestEvidence,
    checked_pairs: Vec<CheckedTokenPair>,
}

fn append_checked_aligned_tokens(
    batch: usize,
    inputs: &[u32],
    targets: &[u32],
    vocabulary_size: usize,
    checked_pairs: &mut Vec<CheckedTokenPair>,
) -> Result<(), EvaluationError> {
    if inputs.len() != targets.len() {
        return Err(EvaluationError::TargetAlignmentMismatch {
            batch,
            inputs: inputs.len(),
            targets: targets.len(),
        });
    }
    for (position, (&input, &target)) in inputs.iter().zip(targets).enumerate() {
        let Some(input_index) = usize::try_from(input)
            .ok()
            .filter(|id| *id < vocabulary_size)
        else {
            return Err(EvaluationError::InputTokenOutOfRange {
                batch,
                position,
                id: input,
                vocabulary_size,
            });
        };
        let Some(target_index) = usize::try_from(target)
            .ok()
            .filter(|id| *id < vocabulary_size)
        else {
            return Err(EvaluationError::TargetTokenOutOfRange {
                batch,
                position,
                id: target,
                vocabulary_size,
            });
        };
        checked_pairs.push(CheckedTokenPair {
            input: input_index,
            target: target_index,
        });
    }
    Ok(())
}

impl<'a> InspectedTestEpoch<'a> {
    fn inspect(epoch: &'a MiniBatchEpoch, vocabulary_size: usize) -> Result<Self, EvaluationError> {
        let mut document_ids = Vec::new();
        let mut window_target_slot_count = 0_usize;
        let mut transition_multiplicities = BTreeMap::<(String, usize), usize>::new();
        let mut fingerprint = 14_695_981_039_346_656_037_u64;
        let checked_pair_count = epoch
            .batches()
            .iter()
            .map(|batch| batch.targets().len())
            .sum();
        let mut checked_pairs = Vec::with_capacity(checked_pair_count);

        for (batch_index, batch) in epoch.batches().iter().enumerate() {
            append_checked_aligned_tokens(
                batch_index,
                batch.inputs(),
                batch.targets(),
                vocabulary_size,
                &mut checked_pairs,
            )?;
            for row in 0..batch.batch_width() {
                let origin = &batch.provenance()[row];
                if !document_ids
                    .iter()
                    .any(|existing| existing == origin.document_id())
                {
                    document_ids.push(origin.document_id().to_owned());
                }
                fnv1a_bytes(&mut fingerprint, origin.document_id().as_bytes());
                fnv1a_byte(&mut fingerprint, 0xff);
                fnv1a_bytes(&mut fingerprint, &(origin.start() as u64).to_le_bytes());
                let inputs = batch
                    .input_row(row)
                    .expect("batch row is constructed from complete inputs");
                let targets = batch
                    .target_row(row)
                    .expect("batch row is constructed from complete targets");
                for (slot, (&input, &target)) in inputs.iter().zip(targets).enumerate() {
                    fnv1a_bytes(&mut fingerprint, &input.to_le_bytes());
                    fnv1a_bytes(&mut fingerprint, &target.to_le_bytes());
                    window_target_slot_count += 1;
                    let absolute_target_position = origin.start() + slot + 1;
                    *transition_multiplicities
                        .entry((origin.document_id().to_owned(), absolute_target_position))
                        .or_default() += 1;
                }
            }
        }

        let document_transition_occurrence_count = transition_multiplicities.len();
        let maximum_multiplicity = transition_multiplicities
            .values()
            .copied()
            .max()
            .unwrap_or(0);
        let mut transition_multiplicity_counts = vec![0_usize; maximum_multiplicity];
        for multiplicity in transition_multiplicities.into_values() {
            transition_multiplicity_counts[multiplicity - 1] += 1;
        }
        debug_assert_eq!(window_target_slot_count, checked_pairs.len());
        debug_assert_eq!(
            window_target_slot_count,
            transition_multiplicity_counts
                .iter()
                .enumerate()
                .map(|(index, count)| (index + 1) * count)
                .sum::<usize>()
        );
        Ok(Self {
            epoch,
            evidence: TestEvidence {
                document_ids,
                window_slot_fingerprint: format!("fnv1a64:{fingerprint:016x}"),
                window_target_slot_count,
                document_transition_occurrence_count,
                transition_multiplicity_counts,
            },
            checked_pairs,
        })
    }

    const fn epoch(&self) -> &'a MiniBatchEpoch {
        self.epoch
    }

    const fn evidence(&self) -> &TestEvidence {
        &self.evidence
    }

    fn checked_pairs(&self) -> &[CheckedTokenPair] {
        &self.checked_pairs
    }

    fn into_evidence(self) -> TestEvidence {
        self.evidence
    }
}

fn require_matching_provenance_assertions(
    expected: &EvaluationProvenance,
    actual: &EvaluationProvenance,
) -> Result<(), EvaluationError> {
    for (field, matches) in [
        (
            ProvenanceField::Corpus,
            expected.corpus_fingerprint == actual.corpus_fingerprint,
        ),
        (
            ProvenanceField::Split,
            expected.split_fingerprint == actual.split_fingerprint,
        ),
        (
            ProvenanceField::Tokenizer,
            expected.tokenizer_fingerprint == actual.tokenizer_fingerprint,
        ),
        (
            ProvenanceField::Context,
            expected.context_length == actual.context_length,
        ),
    ] {
        if !matches {
            return Err(EvaluationError::ProvenanceMismatch { field });
        }
    }
    Ok(())
}

fn parameter_bits(model: &DecoderModel) -> Vec<u64> {
    let mut bits = Vec::new();
    for parameter in model.parameters() {
        bits.extend(
            parameter
                .tensor()
                .value()
                .as_slice()
                .iter()
                .map(|value| value.to_bits()),
        );
    }
    bits
}

fn decoder_configs_match_exactly(selected: DecoderModelConfig, model: DecoderModelConfig) -> bool {
    selected.vocabulary_size() == model.vocabulary_size()
        && selected.model_width() == model.model_width()
        && selected.heads() == model.heads()
        && selected.feed_forward_width() == model.feed_forward_width()
        && selected.layers() == model.layers()
        && selected.max_positions() == model.max_positions()
        && selected.rope_base().to_bits() == model.rope_base().to_bits()
        && selected.rms_epsilon().to_bits() == model.rms_epsilon().to_bits()
}

fn selected_state_matches_model(state: &DecoderModelState, model: &DecoderModel) -> bool {
    if !decoder_configs_match_exactly(state.config(), model.config())
        || state.named_tensors().len() != model.parameters().len()
    {
        return false;
    }
    state
        .named_tensors()
        .zip(model.parameters())
        .all(|((state_name, state_tensor), parameter)| {
            let model_tensor = parameter.tensor().value();
            state_name == parameter.name()
                && state_tensor.shape() == model_tensor.shape()
                && state_tensor
                    .as_slice()
                    .iter()
                    .zip(model_tensor.as_slice())
                    .all(|(state_value, model_value)| {
                        state_value.to_bits() == model_value.to_bits()
                    })
        })
}

fn gradient_bits(model: &DecoderModel) -> Result<Vec<u64>, EvaluationError> {
    let mut bits = Vec::new();
    for parameter in model.parameters() {
        let gradient =
            parameter
                .tensor()
                .gradient()
                .ok_or_else(|| EvaluationError::MissingGradient {
                    name: parameter.name().to_owned(),
                })?;
        bits.extend(gradient.as_slice().iter().map(|value| value.to_bits()));
    }
    Ok(bits)
}

fn fnv1a_byte(hash: &mut u64, byte: u8) {
    *hash ^= u64::from(byte);
    *hash = hash.wrapping_mul(1_099_511_628_211);
}

fn fnv1a_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        fnv1a_byte(hash, *byte);
    }
}

fn score_bigram(
    model: &BigramModel,
    inspected: &InspectedTestEpoch<'_>,
) -> Result<ModelScore, EvaluationError> {
    let mut probabilities = Vec::with_capacity(inspected.checked_pairs().len());
    for pair in inspected.checked_pairs() {
        probabilities
            .push(model.smoothed_probability_for_checked_indices(pair.input, pair.target)?);
    }
    let metrics = score_assigned_probabilities(&probabilities)?;
    ModelScore::new(
        EvaluatedModel::FrozenBigram,
        metrics.total_surprise(),
        metrics.target_count(),
        metrics.mean_nll(),
        metrics.perplexity(),
    )
}

/// Owns one test epoch and consumes its local scoring permission on first use.
///
/// This type is deliberately not cloneable. It enforces one access through one
/// owner; external dataset governance is still required to prevent another
/// process from constructing a separate owner over copied data. Construction
/// checks the epoch's stored `Test` enum value, not its external lineage.
#[derive(Debug)]
pub struct FinalEvaluator {
    test_epoch: MiniBatchEpoch,
    provenance: EvaluationProvenance,
    access_count: u8,
}

impl FinalEvaluator {
    pub fn new(
        test_epoch: MiniBatchEpoch,
        provenance: EvaluationProvenance,
    ) -> Result<Self, EvaluationError> {
        if test_epoch.partition() != Partition::Test {
            return Err(EvaluationError::WrongTestPartition {
                actual: test_epoch.partition(),
            });
        }
        if test_epoch.window_count() == 0 {
            return Err(EvaluationError::EmptyTestEpoch);
        }
        if test_epoch.context_length() != provenance.context_length() {
            return Err(EvaluationError::EpochContextMismatch {
                expected: provenance.context_length(),
                actual: test_epoch.context_length(),
            });
        }
        Ok(Self {
            test_epoch,
            provenance,
            access_count: 0,
        })
    }

    pub const fn access_count(&self) -> u8 {
        self.access_count
    }

    pub fn evaluate_once(
        &mut self,
        decoder: SelectedDecoder<'_>,
        bigram: FrozenBigram<'_>,
    ) -> Result<FinalEvaluationReport, EvaluationError> {
        if self.access_count != 0 {
            return Err(EvaluationError::AlreadyEvaluated);
        }
        require_matching_provenance_assertions(&self.provenance, decoder.provenance())?;
        require_matching_provenance_assertions(&self.provenance, bigram.provenance())?;

        if !selected_state_matches_model(decoder.state(), decoder.model()) {
            return Err(EvaluationError::SelectedStateMismatch);
        }

        let model_config = decoder.model().config();
        if model_config.max_positions() != self.provenance.context_length() {
            return Err(EvaluationError::ModelContextMismatch {
                expected: self.provenance.context_length(),
                actual: model_config.max_positions(),
            });
        }
        let decoder_vocabulary = model_config.vocabulary_size();
        let bigram_vocabulary = bigram.model().vocabulary_size();
        if decoder_vocabulary != bigram_vocabulary {
            return Err(EvaluationError::VocabularyMismatch {
                decoder: decoder_vocabulary,
                bigram: bigram_vocabulary,
            });
        }

        // Every metadata error above leaves the test unopened. From this line on,
        // even an error burns the local gate because token evidence is inspected.
        self.access_count = 1;
        let inspected = InspectedTestEpoch::inspect(&self.test_epoch, decoder_vocabulary)?;
        let model = decoder.model();
        let parameters_before = parameter_bits(model);
        let gradients_before = gradient_bits(model)?;

        let measured = evaluate_no_grad(model, inspected.epoch())?;
        if measured.recorded_graphs() != 0 {
            return Err(EvaluationError::GraphRecorded {
                count: measured.recorded_graphs(),
            });
        }
        let parameters_after = parameter_bits(model);
        if parameters_after != parameters_before {
            return Err(EvaluationError::DecoderParameterChanged);
        }
        let gradients_after = gradient_bits(model)?;
        if gradients_after != gradients_before {
            return Err(EvaluationError::DecoderGradientChanged);
        }

        let decoder_score = ModelScore::new(
            EvaluatedModel::SelectedDecoder,
            measured.mean_loss() * measured.token_count() as f64,
            measured.token_count(),
            measured.mean_loss(),
            measured.mean_loss().exp(),
        )?;
        let bigram_score = score_bigram(bigram.model(), &inspected)?;
        if inspected.evidence().window_target_slot_count != measured.token_count()
            || inspected.evidence().window_target_slot_count != bigram_score.target_count()
        {
            return Err(EvaluationError::TargetCountMismatch {
                expected: inspected.evidence().window_target_slot_count,
                decoder: measured.token_count(),
                bigram: bigram_score.target_count(),
            });
        }
        let evidence = inspected.into_evidence();

        Ok(FinalEvaluationReport {
            version: FINAL_EVALUATION_REPORT_VERSION,
            selected_step: decoder.selected_step(),
            selected_validation_loss: decoder.selected_validation_loss(),
            provenance: self.provenance.clone(),
            test_document_ids: evidence.document_ids,
            window_slot_fingerprint: evidence.window_slot_fingerprint,
            window_count: self.test_epoch.window_count(),
            batch_count: self.test_epoch.batch_count(),
            window_target_slot_count: evidence.window_target_slot_count,
            document_transition_occurrence_count: evidence.document_transition_occurrence_count,
            transition_multiplicity_counts: evidence.transition_multiplicity_counts,
            decoder: decoder_score,
            bigram: bigram_score,
            access_count: self.access_count,
            recorded_graphs: measured.recorded_graphs(),
            parameters_unchanged: true,
            gradients_unchanged: true,
        })
    }
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::data::CausalWindowConfig;
    use crate::models::decoder::{DecoderModel, DecoderModelConfig};
    use crate::nn::init::SplitMix64;
    use crate::training::batch::{BatchDocument, BatchOrder, MiniBatchConfig, MiniBatchEpoch};

    const TRAIN: [u32; 8] = [0, 1, 2, 3, 4, 0, 1, 2];
    const TEST_A: [u32; 6] = [4, 3, 2, 1, 0, 4];
    const TEST_B: [u32; 5] = [3, 2, 1, 0, 4];

    fn provenance() -> EvaluationProvenance {
        EvaluationProvenance::new("corpus-v1", "split-v1", "tokens-v1", 2).unwrap()
    }

    fn model(vocabulary_size: usize, context: usize) -> DecoderModel {
        DecoderModel::new(
            DecoderModelConfig::new(vocabulary_size, 4, 2, 4, 0, context, 10_000.0, 1e-6),
            &mut SplitMix64::from_seed(34),
        )
        .unwrap()
    }

    fn epoch(partition: Partition, documents: &[(&str, &[u32])]) -> MiniBatchEpoch {
        let documents = documents
            .iter()
            .map(|(id, tokens)| BatchDocument::new(id, partition, tokens).unwrap())
            .collect::<Vec<_>>();
        MiniBatchEpoch::build(
            partition,
            &documents,
            CausalWindowConfig::new(2, 1).unwrap(),
            MiniBatchConfig::new(3, BatchOrder::Sequential).unwrap(),
        )
        .unwrap()
    }

    fn baseline(vocabulary_size: usize) -> BigramModel {
        BigramModel::fit_training_documents(vocabulary_size, 1.0, &[&TRAIN[..]]).unwrap()
    }

    #[test]
    fn provenance_assertions_reject_blank_fields_and_zero_context() {
        assert_eq!(
            EvaluationProvenance::new(" ", "split", "tokens", 2),
            Err(EvaluationError::EmptyProvenance {
                field: ProvenanceField::Corpus,
            })
        );
        assert_eq!(
            EvaluationProvenance::new("corpus", "", "tokens", 2),
            Err(EvaluationError::EmptyProvenance {
                field: ProvenanceField::Split,
            })
        );
        assert_eq!(
            EvaluationProvenance::new("corpus", "split", "", 2),
            Err(EvaluationError::EmptyProvenance {
                field: ProvenanceField::Tokenizer,
            })
        );
        assert_eq!(
            EvaluationProvenance::new("corpus", "split", "tokens", 0),
            Err(EvaluationError::ZeroContextLength)
        );
    }

    #[test]
    fn typed_views_validate_caller_role_assertions_and_numeric_shape() {
        let decoder_model = model(5, 2);
        let state = DecoderModelState::snapshot(&decoder_model);
        let provenance = provenance();
        assert!(matches!(
            SelectedDecoder::new(
                &state,
                &decoder_model,
                2,
                1.0,
                Partition::Train,
                &provenance
            ),
            Err(EvaluationError::WrongSelectionPartition {
                actual: Partition::Train
            })
        ));
        for invalid_loss in [f64::NAN, -0.25, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                SelectedDecoder::new(
                    &state,
                    &decoder_model,
                    2,
                    invalid_loss,
                    Partition::Validation,
                    &provenance
                ),
                Err(EvaluationError::InvalidSelectionLoss { .. })
            ));
        }
        assert!(matches!(
            FrozenBigram::new(&baseline(5), Partition::Validation, &provenance),
            Err(EvaluationError::WrongBaselinePartition {
                actual: Partition::Validation
            })
        ));
        let empty =
            BigramModel::fit_training_documents(5, 1.0, [].as_slice())
                .unwrap();
        assert!(matches!(
            FrozenBigram::new(&empty, Partition::Train, &provenance),
            Err(EvaluationError::UnfittedBigram)
        ));
        assert_eq!(
            EvaluationError::EmptyProvenance {
                field: ProvenanceField::Tokenizer,
            }
            .to_string(),
            "tokenizer provenance assertion must not be blank",
        );
        assert_eq!(
            EvaluationError::WrongSelectionPartition {
                actual: Partition::Train,
            }
            .to_string(),
            "the caller selection-partition assertion must be Validation, got Train",
        );
        assert_eq!(
            EvaluationError::WrongBaselinePartition {
                actual: Partition::Validation,
            }
            .to_string(),
            "the caller fit-partition assertion must be Train, got Validation",
        );
        assert_eq!(
            EvaluationError::WrongTestPartition {
                actual: Partition::Validation,
            }
            .to_string(),
            "final evaluation requires an epoch labeled Test, got Validation",
        );
        assert_eq!(
            EvaluationError::ProvenanceMismatch {
                field: ProvenanceField::Corpus,
            }
            .to_string(),
            "corpus provenance assertion does not match",
        );
    }

    #[test]
    fn matching_caller_assertions_are_not_lineage_proof() {
        let evaluator_assertions = EvaluationProvenance::new(
            "arbitrary-corpus",
            "arbitrary-split",
            "arbitrary-tokenizer",
            2,
        )
        .unwrap();
        let decoder_assertions = EvaluationProvenance::new(
            "arbitrary-corpus",
            "arbitrary-split",
            "arbitrary-tokenizer",
            2,
        )
        .unwrap();
        let bigram_assertions = EvaluationProvenance::new(
            "arbitrary-corpus",
            "arbitrary-split",
            "arbitrary-tokenizer",
            2,
        )
        .unwrap();

        // This random decoder never passed through a selection procedure. The
        // constructor accepts the caller's Validation role and finite loss;
        // evaluate_once later checks only that the supplied state/model match.
        let unselected_model = model(5, 2);
        let unselected_state = DecoderModelState::snapshot(&unselected_model);
        let decoder = SelectedDecoder::new(
            &unselected_state,
            &unselected_model,
            999,
            0.25,
            Partition::Validation,
            &decoder_assertions,
        )
        .unwrap();

        // These counts come from an arbitrary slice. The wrapper cannot infer
        // its partition; it accepts the caller's Train role assertion.
        let arbitrary_fit: Vec<u32> = vec![4, 4, 4, 4];
        let arbitrary_bigram =
            BigramModel::fit_training_documents(5, 1.0, [arbitrary_fit.as_slice()].as_slice()).unwrap();
        let bigram =
            FrozenBigram::new(&arbitrary_bigram, Partition::Train, &bigram_assertions).unwrap();

        let mut evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("arbitrary-test", &TEST_A)]),
            evaluator_assertions.clone(),
        )
        .unwrap();
        let report = evaluator.evaluate_once(decoder, bigram).unwrap();

        assert_eq!(report.provenance(), &evaluator_assertions);
        assert_eq!(report.selected_step(), 999);
        assert_eq!(report.selected_validation_loss(), 0.25);
        assert_eq!(report.access_count(), 1);

        // Matching strings also cannot prove that two Test-labeled epochs refer
        // to the same underlying data: a second caller can reuse all three
        // identifiers for a different epoch, which produces a different checked
        // target fingerprint after the gate opens.
        let mut other_evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("different-test", &TEST_B)]),
            evaluator_assertions,
        )
        .unwrap();
        let other_report = other_evaluator.evaluate_once(decoder, bigram).unwrap();
        assert_eq!(report.provenance(), other_report.provenance());
        assert_ne!(
            report.target_fingerprint(),
            other_report.target_fingerprint()
        );
    }

    #[test]
    fn evaluator_requires_nonempty_test_data_with_matching_context() {
        let provenance = provenance();
        assert!(matches!(
            FinalEvaluator::new(
                epoch(Partition::Train, &[("train", &TEST_A)]),
                provenance.clone()
            ),
            Err(EvaluationError::WrongTestPartition {
                actual: Partition::Train
            })
        ));
        assert!(matches!(
            FinalEvaluator::new(
                epoch(Partition::Validation, &[("validation", &TEST_A)]),
                provenance.clone()
            ),
            Err(EvaluationError::WrongTestPartition {
                actual: Partition::Validation
            })
        ));
        assert_eq!(
            FinalEvaluator::new(
                epoch(Partition::Test, &[("empty", &[])]),
                provenance.clone()
            )
            .unwrap_err(),
            EvaluationError::EmptyTestEpoch
        );
        let wrong_context =
            EvaluationProvenance::new("corpus-v1", "split-v1", "tokens-v1", 3).unwrap();
        assert_eq!(
            FinalEvaluator::new(epoch(Partition::Test, &[("test", &TEST_A)]), wrong_context)
                .unwrap_err(),
            EvaluationError::EpochContextMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn one_report_scores_identical_targets_without_changing_decoder_bits() {
        let provenance = provenance();
        let decoder_model = model(5, 2);
        let state = DecoderModelState::snapshot(&decoder_model);
        let decoder = SelectedDecoder::new(
            &state,
            &decoder_model,
            7,
            1.25,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        let bigram_model = baseline(5);
        let bigram = FrozenBigram::new(&bigram_model, Partition::Train, &provenance).unwrap();
        let mut evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("test-a", &TEST_A), ("test-b", &TEST_B)]),
            provenance.clone(),
        )
        .unwrap();
        let report = evaluator.evaluate_once(decoder, bigram).unwrap();

        assert_eq!(report.version(), FINAL_EVALUATION_REPORT_VERSION);
        assert_eq!(report.selected_step(), 7);
        assert_eq!(report.selected_validation_loss(), 1.25);
        assert_eq!(report.provenance(), &provenance);
        assert_eq!(report.test_document_ids(), ["test-a", "test-b"]);
        assert_eq!(report.window_count(), 7);
        assert_eq!(report.batch_count(), 3);
        assert_eq!(report.target_count(), 14);
        assert_eq!(report.window_target_slot_count(), 14);
        assert_eq!(report.document_transition_occurrence_count(), 9);
        assert_eq!(report.transition_multiplicity_counts(), [4, 5]);
        assert_eq!(
            report.target_fingerprint(),
            report.window_slot_fingerprint()
        );
        assert_eq!(report.decoder().target_count(), report.target_count());
        assert_eq!(report.bigram().target_count(), report.target_count());
        assert!(report.decoder().mean_nll().is_finite());
        assert!(report.bigram().mean_nll().is_finite());
        assert_eq!(report.recorded_graphs(), 0);
        assert!(report.parameters_unchanged());
        assert!(report.gradients_unchanged());
        assert_eq!(report.access_count(), 1);
        assert_eq!(evaluator.access_count(), 1);
        assert_eq!(
            evaluator.evaluate_once(decoder, bigram).unwrap_err(),
            EvaluationError::AlreadyEvaluated
        );
    }

    #[test]
    fn selected_model_drift_is_rejected_before_the_test_gate_opens() {
        let provenance = provenance();
        let decoder_model = model(5, 2);
        let state = DecoderModelState::snapshot(&decoder_model);
        let decoder = SelectedDecoder::new(
            &state,
            &decoder_model,
            7,
            1.25,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        let parameter = decoder_model.parameters()[0].tensor();
        let mut changed = parameter.value_snapshot();
        changed.as_mut_slice()[0] += 1.0;
        let next_revision = parameter.next_value_revision().unwrap();
        parameter
            .try_value_write()
            .unwrap()
            .commit(changed, next_revision);

        let bigram_model = baseline(5);
        let bigram = FrozenBigram::new(&bigram_model, Partition::Train, &provenance).unwrap();
        let mut evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("test", &TEST_A)]),
            provenance.clone(),
        )
        .unwrap();

        assert_eq!(
            evaluator.evaluate_once(decoder, bigram).unwrap_err(),
            EvaluationError::SelectedStateMismatch
        );
        assert_eq!(evaluator.access_count(), 0);
    }

    #[test]
    fn signed_zero_config_drift_is_rejected_before_the_test_gate_opens() {
        let provenance = provenance();
        let selected_model = DecoderModel::new(
            DecoderModelConfig::new(5, 4, 2, 4, 0, 2, 10_000.0, 0.0),
            &mut SplitMix64::from_seed(34),
        )
        .unwrap();
        let changed_model = DecoderModel::new(
            DecoderModelConfig::new(5, 4, 2, 4, 0, 2, 10_000.0, -0.0),
            &mut SplitMix64::from_seed(34),
        )
        .unwrap();
        let state = DecoderModelState::snapshot(&selected_model);
        assert_eq!(
            state.bit_pattern(),
            DecoderModelState::snapshot(&changed_model).bit_pattern()
        );
        let decoder = SelectedDecoder::new(
            &state,
            &changed_model,
            7,
            1.25,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        let bigram_model = baseline(5);
        let bigram = FrozenBigram::new(&bigram_model, Partition::Train, &provenance).unwrap();
        let mut evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("test", &TEST_A)]),
            provenance.clone(),
        )
        .unwrap();

        assert_eq!(
            evaluator.evaluate_once(decoder, bigram).unwrap_err(),
            EvaluationError::SelectedStateMismatch
        );
        assert_eq!(evaluator.access_count(), 0);
    }

    #[test]
    fn provenance_assertion_and_shape_errors_before_open_leave_access_unused() {
        let provenance = provenance();
        let decoder_model = model(5, 2);
        let state = DecoderModelState::snapshot(&decoder_model);
        let bigram_model = baseline(5);
        let bigram = FrozenBigram::new(&bigram_model, Partition::Train, &provenance).unwrap();
        for (wrong, field) in [
            (
                EvaluationProvenance::new("other", "split-v1", "tokens-v1", 2).unwrap(),
                ProvenanceField::Corpus,
            ),
            (
                EvaluationProvenance::new("corpus-v1", "other", "tokens-v1", 2).unwrap(),
                ProvenanceField::Split,
            ),
            (
                EvaluationProvenance::new("corpus-v1", "split-v1", "other", 2).unwrap(),
                ProvenanceField::Tokenizer,
            ),
            (
                EvaluationProvenance::new("corpus-v1", "split-v1", "tokens-v1", 3).unwrap(),
                ProvenanceField::Context,
            ),
        ] {
            let decoder = SelectedDecoder::new(
                &state,
                &decoder_model,
                1,
                1.0,
                Partition::Validation,
                &wrong,
            )
            .unwrap();
            let mut evaluator = FinalEvaluator::new(
                epoch(Partition::Test, &[("test", &TEST_A)]),
                provenance.clone(),
            )
            .unwrap();
            assert_eq!(
                evaluator.evaluate_once(decoder, bigram).unwrap_err(),
                EvaluationError::ProvenanceMismatch { field }
            );
            assert_eq!(evaluator.access_count(), 0);

            let decoder = SelectedDecoder::new(
                &state,
                &decoder_model,
                1,
                1.0,
                Partition::Validation,
                &provenance,
            )
            .unwrap();
            let wrong_bigram = FrozenBigram::new(&bigram_model, Partition::Train, &wrong).unwrap();
            let mut evaluator = FinalEvaluator::new(
                epoch(Partition::Test, &[("test", &TEST_A)]),
                provenance.clone(),
            )
            .unwrap();
            assert_eq!(
                evaluator.evaluate_once(decoder, wrong_bigram).unwrap_err(),
                EvaluationError::ProvenanceMismatch { field }
            );
            assert_eq!(evaluator.access_count(), 0);
        }

        let mut evaluator = FinalEvaluator::new(
            epoch(Partition::Test, &[("test", &TEST_A)]),
            provenance.clone(),
        )
        .unwrap();

        let wrong_context_model = model(5, 3);
        let wrong_context_state = DecoderModelState::snapshot(&wrong_context_model);
        let decoder = SelectedDecoder::new(
            &wrong_context_state,
            &wrong_context_model,
            1,
            1.0,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        assert_eq!(
            evaluator.evaluate_once(decoder, bigram).unwrap_err(),
            EvaluationError::ModelContextMismatch {
                expected: 2,
                actual: 3,
            }
        );
        assert_eq!(evaluator.access_count(), 0);

        let wrong_vocabulary_model = model(6, 2);
        let wrong_vocabulary_state = DecoderModelState::snapshot(&wrong_vocabulary_model);
        let decoder = SelectedDecoder::new(
            &wrong_vocabulary_state,
            &wrong_vocabulary_model,
            1,
            1.0,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        assert_eq!(
            evaluator.evaluate_once(decoder, bigram).unwrap_err(),
            EvaluationError::VocabularyMismatch {
                decoder: 6,
                bigram: 5,
            }
        );
        assert_eq!(evaluator.access_count(), 0);
    }

    #[test]
    fn post_open_token_error_consumes_the_gate() {
        let provenance = provenance();
        let decoder_model = model(5, 2);
        let state = DecoderModelState::snapshot(&decoder_model);
        let decoder = SelectedDecoder::new(
            &state,
            &decoder_model,
            1,
            1.0,
            Partition::Validation,
            &provenance,
        )
        .unwrap();
        let bigram_model = baseline(5);
        let bigram = FrozenBigram::new(&bigram_model, Partition::Train, &provenance).unwrap();
        for (invalid, expected_input_error) in [([5, 0, 0, 0], true), ([0, 0, 5, 0], false)] {
            let mut evaluator = FinalEvaluator::new(
                epoch(Partition::Test, &[("test", &invalid)]),
                provenance.clone(),
            )
            .unwrap();
            let error = evaluator.evaluate_once(decoder, bigram).unwrap_err();
            assert!(if expected_input_error {
                matches!(error, EvaluationError::InputTokenOutOfRange { id: 5, .. })
            } else {
                matches!(error, EvaluationError::TargetTokenOutOfRange { id: 5, .. })
            });
            assert_eq!(evaluator.access_count(), 1);
            assert_eq!(
                evaluator.evaluate_once(decoder, bigram).unwrap_err(),
                EvaluationError::AlreadyEvaluated
            );
        }
    }

    #[test]
    fn alignment_guard_rejects_length_drift_before_zip_can_truncate() {
        let mut checked_pairs = Vec::new();
        assert_eq!(
            append_checked_aligned_tokens(3, &[5, 1], &[1], 5, &mut checked_pairs),
            Err(EvaluationError::TargetAlignmentMismatch {
                batch: 3,
                inputs: 2,
                targets: 1,
            })
        );
        assert!(checked_pairs.is_empty());
    }

    #[test]
    fn checked_pair_validation_preserves_input_then_target_error_order() {
        let mut checked_pairs = Vec::new();
        assert_eq!(
            append_checked_aligned_tokens(2, &[5], &[5], 5, &mut checked_pairs),
            Err(EvaluationError::InputTokenOutOfRange {
                batch: 2,
                position: 0,
                id: 5,
                vocabulary_size: 5,
            })
        );
        assert!(checked_pairs.is_empty());
        assert_eq!(
            append_checked_aligned_tokens(2, &[0, 5], &[5, 0], 5, &mut checked_pairs),
            Err(EvaluationError::TargetTokenOutOfRange {
                batch: 2,
                position: 0,
                id: 5,
                vocabulary_size: 5,
            })
        );
        assert!(checked_pairs.is_empty());
        append_checked_aligned_tokens(2, &[4, 3], &[3, 2], 5, &mut checked_pairs).unwrap();
        assert_eq!(
            checked_pairs,
            [
                CheckedTokenPair {
                    input: 4,
                    target: 3,
                },
                CheckedTokenPair {
                    input: 3,
                    target: 2,
                },
            ]
        );
    }

    #[test]
    fn evidence_fingerprint_covers_ordered_origins_inputs_and_targets() {
        let first = epoch(Partition::Test, &[("test-a", &TEST_A), ("test-b", &TEST_B)]);
        let reversed = epoch(Partition::Test, &[("test-b", &TEST_B), ("test-a", &TEST_A)]);
        let first_inspection = InspectedTestEpoch::inspect(&first, 5).unwrap();
        let repeated = InspectedTestEpoch::inspect(&first, 5).unwrap();
        let reversed_inspection = InspectedTestEpoch::inspect(&reversed, 5).unwrap();
        let first_evidence = first_inspection.evidence();
        let reversed_evidence = reversed_inspection.evidence();
        assert_eq!(first_evidence.window_target_slot_count, 14);
        assert_eq!(first_evidence.document_transition_occurrence_count, 9);
        assert_eq!(first_evidence.transition_multiplicity_counts, [4, 5]);
        assert_eq!(first_inspection.checked_pairs().len(), 14);
        assert_eq!(
            first_inspection.checked_pairs().first(),
            Some(&CheckedTokenPair {
                input: 4,
                target: 3,
            })
        );
        assert_eq!(
            first_inspection.checked_pairs().last(),
            Some(&CheckedTokenPair {
                input: 0,
                target: 4,
            })
        );
        assert_eq!(
            first_evidence.window_slot_fingerprint,
            repeated.evidence().window_slot_fingerprint
        );
        assert_ne!(
            first_evidence.window_slot_fingerprint,
            reversed_evidence.window_slot_fingerprint
        );
    }
}
