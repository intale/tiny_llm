//! Greedy decoding, temperature/top-k sampling, and unchanged generation

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use crate::autograd::tensor_core::AutogradContext;
use crate::models::decoder::{DecoderModel, DecoderModelError};
use crate::nn::init::SplitMix64;

const PROBABILITY_TOLERANCE: f64 = 1e-12;

/// The two intentionally distinct next-token policies
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SamplingMode {
    /// Select the highest logit, resolving equal logits by lower token ID
    Greedy,
    /// Sample after positive-temperature scaling and stable top-k truncation
    TemperatureTopK { temperature: f64, top_k: usize },
}

/// One vocabulary entry after ranking and probability construction
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplingCandidate {
    token_id: u32,
    logit: f64,
    rank: usize,
    retained: bool,
    probability: f64,
}

impl SamplingCandidate {
    pub fn token_id(self) -> u32 {
        self.token_id
    }

    pub fn logit(self) -> f64 {
        self.logit
    }

    /// One-based rank under descending logit and ascending token-ID ties.
    pub fn rank(self) -> usize {
        self.rank
    }

    pub fn retained(self) -> bool {
        self.retained
    }

    /// Whether the final represented probability is strictly positive.
    ///
    /// A token can be retained by top-k rank while falling outside this
    /// positive support when its exponential weight underflows to exact zero.
    pub fn has_positive_probability(self) -> bool {
        self.probability > 0.0
    }

    pub fn probability(self) -> f64 {
        self.probability
    }
}

/// Computes the sum of `f64` values using Kahan compensated summation.
///
/// Tracks floating-point rounding errors from previous additions and
/// compensates for them in subsequent iterations.
///
/// This reduces precision loss compared to naive summation, especially
/// when adding small values to a much larger accumulated sum.
///
/// Returns `0.0` for an empty iterator.
/// Does not prevent floating-point overflow or guarantee an exact result.
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        let corrected = value - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }
    sum
}

/// The complete token-ID-ordered distribution plus rank-ordered retained IDs
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingDistribution {
    mode: SamplingMode,
    candidates: Vec<SamplingCandidate>,
    retained_token_ids: Vec<u32>,
}

impl SamplingDistribution {
    pub fn mode(&self) -> SamplingMode {
        self.mode
    }

    /// Candidates are always returned in ascending token-ID order
    pub fn candidates(&self) -> &[SamplingCandidate] {
        &self.candidates
    }

    /// Retained IDs are returned in stable descending-logit rank order
    ///
    /// Membership is determined before probability construction, so this list can include an ID
    /// whose represented probability later underflows to zero
    pub fn retained_token_ids(&self) -> &[u32] {
        &self.retained_token_ids
    }

    /// Positive represented-probability support in ascending token-ID order
    pub fn positive_support_token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.candidates
            .iter()
            .filter(|candidate| candidate.has_positive_probability())
            .map(|candidate| candidate.token_id())
    }

    pub fn probability_sum(&self) -> f64 {
        compensated_sum(
            self.candidates
                .iter()
                .map(|candidate| candidate.probability),
        )
    }

    pub fn candidate(&self, token_id: u32) -> Option<SamplingCandidate> {
        self.candidates
            .get(usize::try_from(token_id).ok()?)
            .copied()
    }
}

/// One selected token and the half-open categorical interval that selected it
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SampledToken {
    token_id: u32,
    unit_draw: Option<f64>,
    interval_start: f64,
    interval_end: f64,
}

impl SampledToken {
    pub fn token_id(self) -> u32 {
        self.token_id
    }

    pub fn unit_draw(self) -> Option<f64> {
        self.unit_draw
    }

    pub fn interval_start(self) -> f64 {
        self.interval_start
    }

    pub fn interval_end(self) -> f64 {
        self.interval_end
    }
}

/// A compact selection paired with the complete distribution used to produce it
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingDecision {
    sampled: SampledToken,
    distribution: SamplingDistribution,
}

impl SamplingDecision {
    pub fn token_id(&self) -> u32 {
        self.sampled.token_id()
    }

    pub fn unit_draw(&self) -> Option<f64> {
        self.sampled.unit_draw()
    }

    pub fn interval_start(&self) -> f64 {
        self.sampled.interval_start()
    }

    pub fn interval_end(&self) -> f64 {
        self.sampled.interval_end()
    }

    pub fn distribution(&self) -> &SamplingDistribution {
        &self.distribution
    }
}

/// A setting or numerical input that cannot define a sampling distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingError {
    EmptyLogits,
    VocabularyTooLarge {
        classes: usize,
    },
    NonFiniteLogit {
        token_id: usize,
        value: f64,
    },
    InvalidTemperature {
        value: f64,
    },
    InvalidTopK {
        top_k: usize,
        vocabulary_size: usize,
    },
    AllocationFailed {
        values: usize,
    },
    InvalidProbabilitySum {
        value: f64,
    },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => formatter.write_str("sampling needs at least one finite logit"),
            Self::VocabularyTooLarge { classes } => write!(
                formatter,
                "sampling vocabulary of {classes} classes does not fit u32 token IDs"
            ),
            Self::NonFiniteLogit { token_id, value } => {
                write!(
                    formatter,
                    "logit for token {token_id} is not finite: {value}"
                )
            }
            Self::InvalidTemperature { value } => write!(
                formatter,
                "sampling temperature must be finite and positive, received {value}"
            ),
            Self::InvalidTopK {
                top_k,
                vocabulary_size,
            } => write!(
                formatter,
                "top-k must be in 1..={vocabulary_size}, received {top_k}"
            ),
            Self::AllocationFailed { values } => {
                write!(
                    formatter,
                    "cannot allocate sampling evidence for {values} values"
                )
            }
            Self::InvalidProbabilitySum { value } => write!(
                formatter,
                "sampling probabilities did not normalize to one: {value}"
            ),
        }
    }
}

impl Error for SamplingError {}

fn retained_count(mode: SamplingMode, vocabulary_size: usize) -> Result<usize, SamplingError> {
    match mode {
        SamplingMode::Greedy => Ok(1),
        SamplingMode::TemperatureTopK { temperature, top_k } => {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(SamplingError::InvalidTemperature { value: temperature });
            }
            if top_k == 0 || top_k > vocabulary_size {
                return Err(SamplingError::InvalidTopK {
                    top_k,
                    vocabulary_size,
                });
            }
            Ok(top_k)
        }
    }
}

fn stable_ranks(logits: &[f64]) -> Result<Vec<usize>, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if u32::try_from(logits.len() - 1).is_err() {
        return Err(SamplingError::VocabularyTooLarge {
            classes: logits.len(),
        });
    }

    for (token_id, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(SamplingError::NonFiniteLogit { token_id, value });
        }
    }

    let mut ranked = Vec::new();
    ranked
        .try_reserve_exact(logits.len())
        .map_err(|_| SamplingError::AllocationFailed {
            values: logits.len(),
        })?;
    ranked.extend(0..logits.len());
    ranked.sort_unstable_by(|&left, &right| {
        logits[right]
            .partial_cmp(&logits[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    Ok(ranked)
}

/// Computes a temperature-scaled logit difference relative to the maximum.
///
/// Uses different operation orders depending on the temperature to reduce
/// floating-point overflow:
///
/// - For `temperature < 1`, subtracts before dividing to avoid overflow
///   when scaling large logits individually.
/// - For `temperature >= 1`, divides before subtracting to reduce the risk
///   of overflow when computing the difference between extreme logits.
///
/// The result is used as the exponent argument in a numerically stable
/// softmax: `exp(scaled_gap(...))`.
///
/// Assumes finite logits, `logit <= maximum`, and a finite positive temperature.
/// The result may be negative infinity if the scaled difference overflows.
fn scaled_gap(logit: f64, maximum: f64, temperature: f64) -> f64 {
    if temperature < 1.0 {
        (logit - maximum) / temperature
    } else {
        logit / temperature - maximum / temperature
    }
}

struct PreparedSampling {
    mode: SamplingMode,
    ranked: Vec<usize>,
    probabilities: Vec<f64>,
    keep: usize,
}

fn prepare_sampling(logits: &[f64], mode: SamplingMode) -> Result<PreparedSampling, SamplingError> {
    let ranked = stable_ranks(logits)?;
    let keep = retained_count(mode, logits.len())?;
    let maximum = logits[ranked[0]];

    let mut probabilities = Vec::new();
    probabilities
        .try_reserve_exact(logits.len())
        .map_err(|_| SamplingError::AllocationFailed {
            values: logits.len(),
        })?;
    probabilities.resize(logits.len(), 0.0);

    match mode {
        SamplingMode::Greedy => probabilities[ranked[0]] = 1.0,
        SamplingMode::TemperatureTopK { temperature, .. } => {
            for (position, &token_id) in ranked[..keep].iter().enumerate() {
                let weight = if position == 0 {
                    1.0
                } else {
                    scaled_gap(logits[token_id], maximum, temperature).exp()
                };
                probabilities[token_id] = weight;
            }
            let weight_sum = compensated_sum(
                ranked[..keep]
                    .iter()
                    .map(|&token_id| probabilities[token_id]),
            );
            if !weight_sum.is_finite() || weight_sum <= 0.0 {
                return Err(SamplingError::InvalidProbabilitySum { value: weight_sum });
            }
            for &token_id in &ranked[..keep] {
                probabilities[token_id] /= weight_sum;
            }
        }
    }

    let probability_sum = compensated_sum(probabilities.iter().copied());
    if !probability_sum.is_finite() || (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(SamplingError::InvalidProbabilitySum {
            value: probability_sum,
        });
    }

    Ok(PreparedSampling {
        mode,
        ranked,
        probabilities,
        keep,
    })
}

/// Materializes a complete, inspectable sampling distribution. DEBUG ONLY
///
/// Combines the original logits with the ranking and probabilities
/// previously computed by `prepare_sampling()`.
///
/// Each candidate contains:
/// - Its token ID and original logit.
/// - Its one-based rank (highest logit has rank 1).
/// - Whether it was retained by greedy or top-k selection.
/// - Its final normalized probability.
///
/// Candidates are stored in ascending token-ID order, while
/// `retained_token_ids` preserves descending-logit rank order.
/// Equal logits are ranked by ascending token ID.
///
/// A retained candidate may have zero probability due to
/// floating-point underflow.
///
/// Does not recompute probabilities or consume randomness.
///
/// # Preconditions
///
/// `prepared` must be produced from the same `logits` slice
/// by `prepare_sampling()`.
fn materialize_distribution(
    logits: &[f64],
    prepared: &PreparedSampling,
) -> Result<SamplingDistribution, SamplingError> {
    let PreparedSampling {
        mode,
        ranked,
        probabilities,
        keep,
    } = prepared;

    let mut rank_by_token = Vec::new();
    rank_by_token
        .try_reserve_exact(logits.len())
        .map_err(|_| SamplingError::AllocationFailed {
            values: logits.len(),
        })?;
    rank_by_token.resize(logits.len(), 0);
    for (position, &token_id) in ranked.iter().enumerate() {
        rank_by_token[token_id] = position + 1;
    }

    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(logits.len())
        .map_err(|_| SamplingError::AllocationFailed {
            values: logits.len(),
        })?;
    for (token_id, ((&logit, &probability), &rank)) in logits
        .iter()
        .zip(probabilities)
        .zip(&rank_by_token)
        .enumerate()
    {
        candidates.push(SamplingCandidate {
            token_id: u32::try_from(token_id).expect("validated token ID must fit u32"),
            logit,
            rank,
            retained: rank <= *keep,
            probability,
        })
    }

    let mut retained_token_ids = Vec::new();
    retained_token_ids
        .try_reserve_exact(*keep)
        .map_err(|_| SamplingError::AllocationFailed { values: *keep })?;
    for &token_id in &ranked[..*keep] {
        retained_token_ids.push(u32::try_from(token_id).expect("validated token ID must fit u32"))
    }

    Ok(SamplingDistribution {
        mode: *mode,
        candidates,
        retained_token_ids,
    })
}

/// Selects a token from a previously prepared sampling distribution.
///
/// In `Greedy` mode, returns the highest-ranked token without consuming
/// randomness. The selected interval is `[0.0, 1.0)`, and `unit_draw`
/// is `None`.
///
/// In `TemperatureTopK` mode:
/// - Consumes exactly one uniform random value from `rng` in `[0, 1)`.
/// - Iterates over tokens in ascending token-ID order.
/// - Skips candidates with zero probability.
/// - Builds cumulative, half-open probability intervals `[start, end)`.
/// - Selects the token whose interval contains the random value.
/// - Extends the final positive-probability interval to `1.0` to
///   compensate for floating-point rounding errors.
///
/// Returns the selected token ID, the optional random draw,
/// and the corresponding probability interval.
///
/// Does not recompute logits, rankings, or probabilities.
///
/// # Preconditions
///
/// `prepared` must contain a valid, normalized distribution produced
/// by `prepare_sampling()`, with at least one positive-probability token.
fn select_prepared(prepared: &PreparedSampling, rng: &mut SplitMix64) -> SampledToken {
    if prepared.mode == SamplingMode::Greedy {
        return SampledToken {
            token_id: u32::try_from(prepared.ranked[0]).expect("validated token ID must fit u32"),
            unit_draw: None,
            interval_start: 0.0,
            interval_end: 1.0,
        };
    }

    let draw = rng.next_unit_f64();
    let final_id = prepared
        .probabilities
        .iter()
        .enumerate()
        .rev()
        .find(|(_, probability)| **probability > 0.0)
        .map(|(token_id, _)| token_id)
        .expect("a normalized distribution must have positive support");
    let mut start = 0.0;
    for (token_id, &probability) in prepared
        .probabilities
        .iter()
        .enumerate()
        .filter(|(_, probability)| **probability > 0.0)
    {
        let end = if token_id == final_id {
            1.0
        } else {
            (start + probability).min(1.0)
        };
        if draw < end || token_id == final_id {
            return SampledToken {
                token_id: u32::try_from(token_id).expect("validated token ID must fit u32"),
                unit_draw: Some(draw),
                interval_start: start,
                interval_end: end,
            };
        }
        start = end;
    }
    unreachable!("the final positive-support token covers every unit draw")
}

fn sample_with_observer<T>(
    logits: &[f64],
    mode: SamplingMode,
    rng: &mut SplitMix64,
    observe: impl FnOnce(&PreparedSampling) -> Result<T, SamplingError>,
) -> Result<(SampledToken, T), SamplingError> {
    let prepared = prepare_sampling(logits, mode)?;
    let observation = observe(&prepared)?;
    let sampled = select_prepared(&prepared, rng);
    Ok((sampled, observation))
}

/// Builds a complete, inspectable distribution without consuming randomness.
pub fn sampling_distribution(
    logits: &[f64],
    mode: SamplingMode,
) -> Result<SamplingDistribution, SamplingError> {
    let prepared = prepare_sampling(logits, mode)?;
    materialize_distribution(logits, &prepared)
}

/// Selects one token without retaining the complete inspectable distribution.
pub fn sample_next_token(
    logits: &[f64],
    mode: SamplingMode,
    rng: &mut SplitMix64,
) -> Result<SampledToken, SamplingError> {
    sample_with_observer(logits, mode, rng, |_| Ok(())).map(|(sampled, ())| sampled)
}

/// Selects one token and records the complete distribution used for inspection.
pub fn sample_next_token_with_trace(
    logits: &[f64],
    mode: SamplingMode,
    rng: &mut SplitMix64,
) -> Result<SamplingDecision, SamplingError> {
    let (sampled, distribution) = sample_with_observer(logits, mode, rng, |prepared| {
        materialize_distribution(logits, prepared)
    })?;
    Ok(SamplingDecision {
        sampled,
        distribution,
    })
}

/// Settings for one bounded, uncached autoregressive call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationConfig {
    mode: SamplingMode,
    eos_token: Option<u32>,
    max_new_tokens: usize,
}

impl GenerationConfig {
    pub fn new(mode: SamplingMode, eos_token: Option<u32>, max_new_tokens: usize) -> Self {
        Self {
            mode,
            eos_token,
            max_new_tokens,
        }
    }

    pub fn mode(self) -> SamplingMode {
        self.mode
    }

    pub fn eos_token(self) -> Option<u32> {
        self.eos_token
    }

    pub fn max_new_tokens(self) -> usize {
        self.max_new_tokens
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStop {
    Eos,
    TokenLimit,
    ContextLimit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationStep {
    prefix_length: usize,
    token_id: u32,
    unit_draw: Option<f64>,
    interval_start: f64,
    interval_end: f64,
}

impl GenerationStep {
    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    pub fn unit_draw(&self) -> Option<f64> {
        self.unit_draw
    }

    pub fn interval_start(&self) -> f64 {
        self.interval_start
    }

    pub fn interval_end(&self) -> f64 {
        self.interval_end
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationResult {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    steps: Vec<GenerationStep>,
    stop: GenerationStop,
    full_prefix_calls: usize,
}

impl GenerationResult {
    pub fn prompt(&self) -> &[u32] {
        &self.prompt
    }

    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub fn steps(&self) -> &[GenerationStep] {
        &self.steps
    }

    pub fn stop(&self) -> GenerationStop {
        self.stop
    }

    pub fn full_prefix_calls(&self) -> usize {
        self.full_prefix_calls
    }
}

#[derive(Debug)]
pub enum GenerationError {
    Sampling(SamplingError),
    Model(DecoderModelError),
    EmptyPrompt,
    PromptTooLong {
        tokens: usize,
        max_positions: usize,
    },
    PromptTokenOutOfBounds {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    EosTokenOutOfBounds {
        token_id: u32,
        vocabulary_size: usize,
    },
    LogitCountMismatch {
        expected: usize,
        actual: usize,
    },
    AllocationFailed {
        values: usize,
    },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sampling(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::EmptyPrompt => formatter.write_str("generation needs a nonempty prompt"),
            Self::PromptTooLong {
                tokens,
                max_positions,
            } => write!(
                formatter,
                "prompt has {tokens} tokens, exceeding context capacity {max_positions}"
            ),
            Self::PromptTokenOutOfBounds {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "prompt token {token_id} at position {position} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::EosTokenOutOfBounds {
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "EOS token {token_id} is out of bounds for vocabulary {vocabulary_size}"
            ),
            Self::LogitCountMismatch { expected, actual } => write!(
                formatter,
                "last-position logits need {expected} values, received {actual}"
            ),
            Self::AllocationFailed { values } => {
                write!(
                    formatter,
                    "cannot allocate generation evidence for {values} values"
                )
            }
        }
    }
}

impl Error for GenerationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Sampling(error) => Some(error),
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SamplingError> for GenerationError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<DecoderModelError> for GenerationError {
    fn from(error: DecoderModelError) -> Self {
        Self::Model(error)
    }
}

/// Generates tokens autoregressively using a caller-provided logit function.
///
/// Implements the generation loop independently of the underlying model.
///
/// On each iteration:
/// 1. Calls `last_logits` with the complete current token prefix.
/// 2. Expects one logit per vocabulary token.
/// 3. Selects the next token using the configured sampling mode.
/// 4. Appends the selected token to the prefix and generated output.
/// 5. Stops on EOS, the token limit, or the context limit.
///
/// `last_logits` must return logits for the next token after the given
/// prefix, ordered by token ID. It may use any inference implementation.
///
/// # Stopping conditions
///
/// - `Eos`: the newly generated token matches `config.eos_token`.
/// - `TokenLimit`: `config.max_new_tokens` tokens have been generated.
/// - `ContextLimit`: the prefix becomes longer than `max_positions`.
///
/// Stop conditions are checked in the order listed above.
///
/// A prefix of length `max_positions` may generate one additional token.
/// The resulting prefix can therefore have length `max_positions + 1`,
/// but it is never passed to `last_logits` again.
///
/// If `max_new_tokens` is zero, returns immediately after validation,
/// without invoking `last_logits` or consuming randomness.
///
/// # Randomness
///
/// Greedy sampling does not advance `rng`.
/// Temperature/top-k sampling consumes one random draw per generated token,
/// including when `top_k == 1`.
///
/// # Returns
///
/// A `GenerationResult` containing:
/// - A copy of the original prompt.
/// - The generated tokens, excluding the original prompt.
/// - Per-step sampling information.
/// - The reason generation stopped.
/// - The number of successful calls to `last_logits`.
fn generate_with<F>(
    vocabulary_size: usize,
    max_positions: usize,
    prompt: &[u32],
    config: GenerationConfig,
    rng: &mut SplitMix64,
    mut last_logits: F,
) -> Result<GenerationResult, GenerationError>
where
    F: FnMut(&[u32]) -> Result<Vec<f64>, GenerationError>,
{
    if prompt.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }
    if prompt.len() > max_positions {
        return Err(GenerationError::PromptTooLong {
            tokens: prompt.len(),
            max_positions,
        });
    }
    for (position, &token_id) in prompt.iter().enumerate() {
        if usize::try_from(token_id)
            .ok()
            .is_none_or(|token| token >= vocabulary_size)
        {
            return Err(GenerationError::PromptTokenOutOfBounds {
                position,
                token_id,
                vocabulary_size,
            });
        }
    }
    if let Some(token_id) = config.eos_token
        && usize::try_from(token_id)
        .ok()
        .is_none_or(|token| token >= vocabulary_size)
    {
        return Err(GenerationError::EosTokenOutOfBounds {
            token_id,
            vocabulary_size,
        });
    }
    retained_count(config.mode, vocabulary_size)?;

    let planned_steps = config.max_new_tokens.min(
        max_positions
            .checked_sub(prompt.len())
            .and_then(|remaining| remaining.checked_add(1))
            .ok_or(GenerationError::AllocationFailed { values: usize::MAX })?,
    );
    let prefix_capacity = prompt
        .len()
        .checked_add(planned_steps)
        .ok_or(GenerationError::AllocationFailed { values: usize::MAX })?;

    let mut prompt_copy = Vec::new();
    prompt_copy
        .try_reserve_exact(prompt.len())
        .map_err(|_| GenerationError::AllocationFailed {
            values: prompt.len(),
        })?;
    prompt_copy.extend_from_slice(prompt);

    let mut prefix = Vec::new();
    prefix
        .try_reserve_exact(prefix_capacity)
        .map_err(|_| GenerationError::AllocationFailed {
            values: prefix_capacity,
        })?;
    prefix.extend_from_slice(prompt);
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(planned_steps)
        .map_err(|_| GenerationError::AllocationFailed {
            values: planned_steps,
        })?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(planned_steps)
        .map_err(|_| GenerationError::AllocationFailed {
            values: planned_steps,
        })?;
    let mut full_prefix_calls = 0usize;

    if config.max_new_tokens == 0 {
        return Ok(GenerationResult {
            prompt: prompt_copy,
            generated,
            steps,
            stop: GenerationStop::TokenLimit,
            full_prefix_calls,
        });
    }

    let stop = loop {
        let prefix_length = prefix.len();
        let logits = last_logits(&prefix)?;
        full_prefix_calls += 1;
        if logits.len() != vocabulary_size {
            return Err(GenerationError::LogitCountMismatch {
                expected: vocabulary_size,
                actual: logits.len(),
            });
        }
        let decision = sample_next_token(&logits, config.mode, rng)?;
        let token_id = decision.token_id();
        let unit_draw = decision.unit_draw();
        let interval_start = decision.interval_start();
        let interval_end = decision.interval_end();
        prefix.push(token_id);
        generated.push(token_id);
        steps.push(GenerationStep {
            prefix_length,
            token_id,
            unit_draw,
            interval_start,
            interval_end,
        });

        if config.eos_token == Some(token_id) {
            break GenerationStop::Eos;
        }
        if generated.len() == config.max_new_tokens {
            break GenerationStop::TokenLimit;
        }
        if prefix.len() > max_positions {
            break GenerationStop::ContextLimit;
        }
    };

    Ok(GenerationResult {
        prompt: prompt_copy,
        generated,
        steps,
        stop,
        full_prefix_calls,
    })
}

/// Generates tokens with a DecoderModel without using a KV-cache.
///
/// Connects the DecoderModel forward pass to the autoregressive
/// generation loop implemented by `generate_with()`.
///
/// For every generated token:
/// 1. Runs the decoder on the complete current prefix.
/// 2. Computes logits with shape `[1, T, V]`, where:
///    - `T` is the current prefix length.
///    - `V` is the vocabulary size.
/// 3. Extracts the final position's `V` logits.
/// 4. Passes those logits to `generate_with()` for token selection.
///
/// The forward pass uses `AutogradContext::no_grad()` to avoid recording
/// an autograd graph.
///
/// Previously processed tokens are recomputed on every iteration.
/// No attention keys, values, or intermediate activations are cached.
///
/// # Arguments
///
/// - `model`: the decoder used to predict the next-token logits.
/// - `prompt`: the initial sequence of token IDs.
/// - `config`: sampling mode, optional EOS token, and generation limit.
/// - `rng`: random generator used by stochastic sampling.
///
/// # Returns
///
/// A `GenerationResult` containing generated token IDs, compact
/// per-step sampling information, the stopping reason, and the
/// number of complete-prefix decoder calls.
///
/// The original prompt is preserved separately from generated tokens.
pub fn generate_uncached(
    model: &DecoderModel,
    prompt: &[u32],
    config: GenerationConfig,
    rng: &mut SplitMix64,
) -> Result<GenerationResult, GenerationError> {
    let model_config = model.config();
    let vocabulary_size = model_config.vocabulary_size();
    let context = AutogradContext::no_grad();
    generate_with(
        vocabulary_size,
        model_config.max_positions(),
        prompt,
        config,
        rng,
        |prefix| {
            let forward = model.forward_with_context(context, prefix, &[1, prefix.len()])?;
            let logits = forward.logits().value();
            let expected = prefix
                .len()
                .checked_mul(vocabulary_size)
                .ok_or(GenerationError::AllocationFailed { values: usize::MAX })?;
            if logits.len() != expected {
                return Err(GenerationError::LogitCountMismatch {
                    expected,
                    actual: logits.len(),
                });
            }
            let start = expected - vocabulary_size;
            let mut final_logits = Vec::new();
            final_logits
                .try_reserve_exact(vocabulary_size)
                .map_err(|_| GenerationError::AllocationFailed {
                    values: vocabulary_size,
                })?;
            final_logits.extend_from_slice(&logits.as_slice()[start..]);
            Ok(final_logits)
        },
    )
}

#[cfg(test)]
mod course_tests {
    use super::*;
    use crate::models::decoder::DecoderModelConfig;
    use crate::nn::probability::softmax;
    use crate::tensor::storage::Tensor;

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left:.17} to be within {tolerance} of {right:.17}"
        );
    }

    fn assert_sample_matches_trace(sampled: SampledToken, traced: &SamplingDecision) {
        assert_eq!(sampled.token_id(), traced.token_id());
        assert_eq!(
            sampled.unit_draw().map(f64::to_bits),
            traced.unit_draw().map(f64::to_bits)
        );
        assert_eq!(
            sampled.interval_start().to_bits(),
            traced.interval_start().to_bits()
        );
        assert_eq!(
            sampled.interval_end().to_bits(),
            traced.interval_end().to_bits()
        );
    }

    fn assert_sampling_error_matches(left: SamplingError, right: SamplingError) {
        match (left, right) {
            (
                SamplingError::NonFiniteLogit {
                    token_id: left_token,
                    value: left_value,
                },
                SamplingError::NonFiniteLogit {
                    token_id: right_token,
                    value: right_value,
                },
            ) => {
                assert_eq!(left_token, right_token);
                assert_eq!(left_value.to_bits(), right_value.to_bits());
            }
            (left, right) => assert_eq!(left, right),
        }
    }

    #[test]
    fn stable_top_k_keeps_exactly_k_and_resolves_boundary_ties_by_token_id() {
        let logits = [0.0, 1.0, 1.0, 2.0];
        let distribution = sampling_distribution(
            &logits,
            SamplingMode::TemperatureTopK {
                temperature: 1.0,
                top_k: 2,
            },
        )
            .unwrap();
        assert_eq!(distribution.retained_token_ids(), &[3, 1]);
        assert_eq!(
            distribution
                .candidates()
                .iter()
                .map(|candidate| (candidate.rank(), candidate.retained()))
                .collect::<Vec<_>>(),
            [(4, false), (2, true), (3, false), (1, true)]
        );
        assert_eq!(distribution.candidates()[0].probability().to_bits(), 0);
        assert_eq!(distribution.candidates()[2].probability().to_bits(), 0);
        assert_close(
            distribution.candidates()[1].probability(),
            0.268_941_421_369_995_1,
            1e-15,
        );
        assert_close(
            distribution.candidates()[3].probability(),
            0.731_058_578_630_004_9,
            1e-15,
        );
        assert_close(distribution.probability_sum(), 1.0, PROBABILITY_TOLERANCE);
    }

    #[test]
    fn signed_zero_ties_and_greedy_do_not_advance_rng() {
        let mut rng = SplitMix64::from_seed(36);
        let before = rng.state();
        let decision =
            sample_next_token(&[-0.0, 0.0, -1.0], SamplingMode::Greedy, &mut rng).unwrap();
        assert_eq!(decision.token_id(), 0);
        assert_eq!(decision.unit_draw(), None);
        assert_eq!(
            (decision.interval_start(), decision.interval_end()),
            (0.0, 1.0)
        );
        assert_eq!(rng.state(), before);
    }

    #[test]
    fn top_k_one_matches_greedy_but_still_consumes_one_stochastic_draw() {
        let logits = [0.0, 1.0, 1.0, 2.0];
        let mut greedy_rng = SplitMix64::from_seed(36);
        let mut sample_rng = greedy_rng.clone();
        let greedy = sample_next_token(&logits, SamplingMode::Greedy, &mut greedy_rng).unwrap();
        let sampled = sample_next_token_with_trace(
            &logits,
            SamplingMode::TemperatureTopK {
                temperature: 9.0,
                top_k: 1,
            },
            &mut sample_rng,
        )
            .unwrap();
        assert_eq!(greedy.token_id(), 3);
        assert_eq!(sampled.token_id(), greedy.token_id());
        assert_eq!(sampled.distribution().candidates()[3].probability(), 1.0);
        assert_eq!(greedy_rng.state(), 36);
        assert_ne!(sample_rng.state(), 36);
    }

    #[test]
    fn lean_and_traced_sampling_share_exact_results_and_rng_state() {
        let cases = [
            (vec![-0.0, 0.0, -1.0], SamplingMode::Greedy),
            (
                vec![0.0, 1.0, 1.0, 2.0],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 3,
                },
            ),
            (
                vec![0.0, 1.0, 1.0, 2.0],
                SamplingMode::TemperatureTopK {
                    temperature: 9.0,
                    top_k: 1,
                },
            ),
            (
                vec![2.0, 2.0, 1.0],
                SamplingMode::TemperatureTopK {
                    temperature: f64::MIN_POSITIVE,
                    top_k: 3,
                },
            ),
            (
                vec![-f64::MAX, 0.0, f64::MAX],
                SamplingMode::TemperatureTopK {
                    temperature: f64::MAX,
                    top_k: 3,
                },
            ),
        ];

        for (logits, mode) in cases {
            let mut lean_rng = SplitMix64::from_seed(36);
            let mut traced_rng = lean_rng.clone();
            let sampled = sample_next_token(&logits, mode, &mut lean_rng).unwrap();
            let traced = sample_next_token_with_trace(&logits, mode, &mut traced_rng).unwrap();
            assert_sample_matches_trace(sampled, &traced);
            assert_eq!(lean_rng.state(), traced_rng.state());
            assert_eq!(
                traced.distribution(),
                &sampling_distribution(&logits, mode).unwrap()
            );
        }
    }

    #[test]
    fn lean_and_traced_sampling_reject_identical_inputs_without_a_draw() {
        let cases = [
            (Vec::new(), SamplingMode::Greedy),
            (
                vec![0.0, f64::NAN],
                SamplingMode::TemperatureTopK {
                    temperature: 0.0,
                    top_k: 0,
                },
            ),
            (
                vec![0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 0.0,
                    top_k: 0,
                },
            ),
            (
                vec![0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 0,
                },
            ),
            (
                vec![0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 2,
                },
            ),
        ];

        for (logits, mode) in cases {
            let mut lean_rng = SplitMix64::from_seed(99);
            let mut traced_rng = lean_rng.clone();
            let initial_state = lean_rng.state();
            assert_sampling_error_matches(
                sample_next_token(&logits, mode, &mut lean_rng).unwrap_err(),
                sample_next_token_with_trace(&logits, mode, &mut traced_rng).unwrap_err(),
            );
            assert_eq!(lean_rng.state(), initial_state);
            assert_eq!(traced_rng.state(), initial_state);
        }
    }

    #[test]
    fn observation_failure_cannot_advance_the_random_stream() {
        let mut rng = SplitMix64::from_seed(36);
        let initial_state = rng.state();
        let result = sample_with_observer(
            &[0.0, 1.0],
            SamplingMode::TemperatureTopK {
                temperature: 1.0,
                top_k: 2,
            },
            &mut rng,
            |_| Err::<(), _>(SamplingError::AllocationFailed { values: 2 }),
        );
        assert_eq!(
            result.unwrap_err(),
            SamplingError::AllocationFailed { values: 2 }
        );
        assert_eq!(rng.state(), initial_state);
    }

    #[test]
    fn full_vocabulary_distribution_matches_the_existing_stable_softmax() {
        let logits = [1_000.0, 999.0, 998.0, -1_000.0];
        let distribution = sampling_distribution(
            &logits,
            SamplingMode::TemperatureTopK {
                temperature: 1.0,
                top_k: logits.len(),
            },
        )
            .unwrap();
        let expected = softmax(
            &Tensor::from_vec(vec![1, logits.len()], logits.to_vec())
                .unwrap()
                .view(),
            1,
        )
            .unwrap();
        for (candidate, expected) in distribution.candidates().iter().zip(expected.as_slice()) {
            assert_close(candidate.probability(), *expected, 1e-15);
        }
    }

    #[test]
    fn uniform_llm_sized_vocabulary_normalizes_with_compensated_summation() {
        let logits = vec![0.0; 100_000];
        let distribution = sampling_distribution(
            &logits,
            SamplingMode::TemperatureTopK {
                temperature: 1.0,
                top_k: logits.len(),
            },
        )
            .unwrap();
        assert_eq!(distribution.candidates().len(), logits.len());
        assert_eq!(distribution.retained_token_ids().len(), logits.len());
        assert_eq!(distribution.retained_token_ids()[..3], [0, 1, 2]);
        assert_close(distribution.candidates()[99_999].probability(), 1e-5, 1e-18);
        assert_close(distribution.probability_sum(), 1.0, PROBABILITY_TOLERANCE);
    }

    #[test]
    fn temperature_limits_are_finite_normalized_and_keep_maximum_ties() {
        let logits = [2.0, 2.0, 1.0];
        let mode = SamplingMode::TemperatureTopK {
            temperature: f64::MIN_POSITIVE,
            top_k: 3,
        };
        let low = sampling_distribution(&logits, mode).unwrap();
        assert_eq!(low.retained_token_ids(), &[0, 1, 2]);
        assert!(
            low.candidates()
                .iter()
                .all(|candidate| candidate.retained())
        );
        assert_eq!(low.positive_support_token_ids().collect::<Vec<_>>(), [0, 1]);
        assert!(low.candidates()[0].has_positive_probability());
        assert!(low.candidates()[1].has_positive_probability());
        assert!(!low.candidates()[2].has_positive_probability());
        assert_eq!(low.candidates()[0].probability(), 0.5);
        assert_eq!(low.candidates()[1].probability(), 0.5);
        assert_eq!(
            low.candidates()[2].probability().to_bits(),
            0.0_f64.to_bits()
        );

        let mut expected_rng = SplitMix64::from_seed(36);
        let expected_draw = expected_rng.next_unit_f64();
        let mut rng = SplitMix64::from_seed(36);
        let sampled = sample_next_token(&logits, mode, &mut rng).unwrap();
        assert_eq!(sampled.token_id(), 1);
        assert_eq!(
            sampled.unit_draw().unwrap().to_bits(),
            expected_draw.to_bits()
        );
        assert_eq!(
            (sampled.interval_start(), sampled.interval_end()),
            (0.5, 1.0)
        );
        assert_eq!(rng.state(), expected_rng.state());

        let high = sampling_distribution(
            &[-f64::MAX, 0.0, f64::MAX],
            SamplingMode::TemperatureTopK {
                temperature: f64::MAX,
                top_k: 3,
            },
        )
            .unwrap();
        let probabilities = high
            .candidates()
            .iter()
            .map(|candidate| candidate.probability())
            .collect::<Vec<_>>();
        assert!(
            probabilities
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert_close(probabilities.iter().sum(), 1.0, PROBABILITY_TOLERANCE);
        assert!(probabilities[0] < probabilities[1] && probabilities[1] < probabilities[2]);
    }

    #[test]
    fn frozen_seeded_sequence_and_half_open_intervals_are_reproducible() {
        let logits = [0.0, 1.0, 1.0, 2.0];
        let mode = SamplingMode::TemperatureTopK {
            temperature: 1.0,
            top_k: 3,
        };
        let distribution = sampling_distribution(&logits, mode).unwrap();
        let expected = [
            0.0,
            0.211_941_557_617_085_44,
            0.211_941_557_617_085_44,
            0.576_116_884_765_829_1,
        ];
        for (candidate, expected) in distribution.candidates().iter().zip(expected) {
            assert_close(candidate.probability(), expected, 1e-15);
        }
        assert_eq!(distribution.retained_token_ids(), &[3, 1, 2]);

        let mut left = SplitMix64::from_seed(36);
        let mut right = SplitMix64::from_seed(36);
        let mut sequence = Vec::new();
        for _ in 0..8 {
            let decision = sample_next_token(&logits, mode, &mut left).unwrap();
            let draw = decision.unit_draw().unwrap();
            assert!(draw >= decision.interval_start() && draw < decision.interval_end());
            sequence.push(decision.token_id());
            assert_eq!(
                decision,
                sample_next_token(&logits, mode, &mut right).unwrap()
            );
        }
        assert_eq!(sequence, [3, 2, 2, 2, 3, 3, 3, 3]);
        assert_eq!(left.state(), right.state());
    }

    #[test]
    fn invalid_inputs_are_specific_and_transactional() {
        let cases = [
            sampling_distribution(&[], SamplingMode::Greedy).unwrap_err(),
            sampling_distribution(&[0.0, f64::NAN], SamplingMode::Greedy).unwrap_err(),
            sampling_distribution(
                &[0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 0.0,
                    top_k: 1,
                },
            )
                .unwrap_err(),
            sampling_distribution(
                &[0.0],
                SamplingMode::TemperatureTopK {
                    temperature: f64::INFINITY,
                    top_k: 1,
                },
            )
                .unwrap_err(),
            sampling_distribution(
                &[0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 0,
                },
            )
                .unwrap_err(),
            sampling_distribution(
                &[0.0],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 2,
                },
            )
                .unwrap_err(),
        ];
        assert!(matches!(cases[0], SamplingError::EmptyLogits));
        assert!(matches!(
            cases[1],
            SamplingError::NonFiniteLogit { token_id: 1, .. }
        ));
        assert!(matches!(
            cases[2],
            SamplingError::InvalidTemperature { value: 0.0 }
        ));
        assert!(matches!(cases[3], SamplingError::InvalidTemperature { .. }));
        assert!(matches!(
            cases[4],
            SamplingError::InvalidTopK { top_k: 0, .. }
        ));
        assert!(matches!(
            cases[5],
            SamplingError::InvalidTopK { top_k: 2, .. }
        ));

        let mut rng = SplitMix64::from_seed(99);
        let state = rng.state();
        assert!(
            sample_next_token(
                &[0.0, f64::INFINITY],
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 1,
                },
                &mut rng,
            )
                .is_err()
        );
        assert_eq!(rng.state(), state);
    }

    #[test]
    fn scripted_generation_includes_eos_and_audits_complete_prefixes() {
        let mut calls = Vec::new();
        let mut rng = SplitMix64::from_seed(1);
        let state = rng.state();
        let result = generate_with(
            5,
            4,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, Some(4), 4),
            &mut rng,
            |prefix| {
                calls.push(prefix.to_vec());
                Ok(if prefix.len() == 1 {
                    vec![0.0, 0.0, 2.0, 0.0, 0.0]
                } else {
                    vec![0.0, 0.0, 0.0, 0.0, 3.0]
                })
            },
        )
            .unwrap();
        assert_eq!(calls, [vec![0], vec![0, 2]]);
        assert_eq!(result.generated(), &[2, 4]);
        assert_eq!(result.stop(), GenerationStop::Eos);
        assert_eq!(result.full_prefix_calls(), 2);
        assert_eq!(
            result
                .steps()
                .iter()
                .map(GenerationStep::prefix_length)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            result
                .steps()
                .iter()
                .map(GenerationStep::token_id)
                .collect::<Vec<_>>(),
            [2, 4]
        );
        assert!(result.steps().iter().all(|step| step.unit_draw().is_none()));
        assert!(std::mem::size_of::<GenerationStep>() <= 64);
        assert_eq!(rng.state(), state);
    }

    #[test]
    fn context_and_token_limits_stop_without_an_extra_forward() {
        let mut context_calls = Vec::new();
        let mut rng = SplitMix64::from_seed(2);
        let context = generate_with(
            2,
            2,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 4),
            &mut rng,
            |prefix| {
                context_calls.push(prefix.to_vec());
                Ok(vec![0.0, 1.0])
            },
        )
            .unwrap();
        assert_eq!(context_calls, [vec![0], vec![0, 1]]);
        assert_eq!(context.generated(), &[1, 1]);
        assert_eq!(context.stop(), GenerationStop::ContextLimit);
        assert_eq!(context.full_prefix_calls(), 2);

        let mut token_calls = 0;
        let token_limit = generate_with(
            2,
            2,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 1),
            &mut rng,
            |_| {
                token_calls += 1;
                Ok(vec![0.0, 1.0])
            },
        )
            .unwrap();
        assert_eq!(token_limit.generated(), &[1]);
        assert_eq!(token_limit.stop(), GenerationStop::TokenLimit);
        assert_eq!(token_limit.full_prefix_calls(), 1);
        assert_eq!(token_calls, 1);

        let mut stochastic_rng = SplitMix64::from_seed(11);
        let mut expected_rng = stochastic_rng.clone();
        expected_rng.next_unit_f64();
        expected_rng.next_unit_f64();
        let stochastic = generate_with(
            2,
            4,
            &[0],
            GenerationConfig::new(
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 1,
                },
                None,
                2,
            ),
            &mut stochastic_rng,
            |_| Ok(vec![0.0, 1.0]),
        )
            .unwrap();
        assert_eq!(stochastic.stop(), GenerationStop::TokenLimit);
        assert_eq!(stochastic.full_prefix_calls(), 2);
        assert!(stochastic.steps().iter().all(|step| {
            step.unit_draw().is_some() && step.interval_start() == 0.0 && step.interval_end() == 1.0
        }));
        assert_eq!(stochastic_rng.state(), expected_rng.state());

        let zero = generate_with(
            2,
            2,
            &[0],
            GenerationConfig::new(SamplingMode::Greedy, None, 0),
            &mut rng,
            |_| panic!("zero budget must not call the model"),
        )
            .unwrap();
        assert_eq!(zero.stop(), GenerationStop::TokenLimit);
        assert_eq!(zero.full_prefix_calls(), 0);
    }

    #[test]
    fn generation_rejects_prompt_eos_settings_and_logit_shape_before_extra_work() {
        let mut rng = SplitMix64::from_seed(3);
        assert!(matches!(
            generate_with(
                2,
                2,
                &[],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
                |_| panic!("invalid prompt must not call model"),
            ),
            Err(GenerationError::EmptyPrompt)
        ));
        assert!(matches!(
            generate_with(
                2,
                2,
                &[0, 1, 0],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
                |_| panic!("long prompt must not call model"),
            ),
            Err(GenerationError::PromptTooLong { .. })
        ));
        assert!(matches!(
            generate_with(
                2,
                2,
                &[2],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
                |_| panic!("invalid token must not call model"),
            ),
            Err(GenerationError::PromptTokenOutOfBounds { .. })
        ));
        assert!(matches!(
            generate_with(
                2,
                2,
                &[0],
                GenerationConfig::new(SamplingMode::Greedy, Some(2), 1),
                &mut rng,
                |_| panic!("invalid EOS must not call model"),
            ),
            Err(GenerationError::EosTokenOutOfBounds { .. })
        ));
        assert!(matches!(
            generate_with(
                2,
                2,
                &[0],
                GenerationConfig::new(SamplingMode::Greedy, None, 1),
                &mut rng,
                |_| Ok(vec![0.0]),
            ),
            Err(GenerationError::LogitCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn actual_decoder_generation_is_graph_free_and_advances_once_per_sample() {
        let mut init_rng = SplitMix64::from_seed(36);
        let model = DecoderModel::new(
            DecoderModelConfig::new(5, 4, 2, 4, 1, 2, 10_000.0, 1e-6),
            &mut init_rng,
        )
            .unwrap();
        let before = model
            .parameters()
            .iter()
            .map(|parameter| {
                parameter.tensor().gradient().map(|gradient| {
                    gradient
                        .as_slice()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let mut sample_rng = SplitMix64::from_seed(36);
        let initial_state = sample_rng.state();
        let result = generate_uncached(
            &model,
            &[0],
            GenerationConfig::new(
                SamplingMode::TemperatureTopK {
                    temperature: 1.0,
                    top_k: 3,
                },
                None,
                4,
            ),
            &mut sample_rng,
        )
            .unwrap();
        assert_eq!(result.full_prefix_calls(), 2);
        assert_eq!(result.steps().len(), 2);
        assert_eq!(result.stop(), GenerationStop::ContextLimit);
        assert_ne!(sample_rng.state(), initial_state);
        assert_eq!(
            model
                .parameters()
                .iter()
                .map(|parameter| {
                    parameter.tensor().gradient().map(|gradient| {
                        gradient
                            .as_slice()
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>(),
            before
        );
    }
}
