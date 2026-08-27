//! Negative log-likelihood(NLL) and perplexity for observed target tokens.

use std::error::Error;
use std::fmt;

use crate::bigram::{BigramError, BigramModel};
use crate::corpus::Partition;
use crate::data::EncodedCorpusPartitions;

/// Aggregate probability metrics weighted by the number of observed targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSummary {
    total_surprise: f64,
    target_count: usize,
    mean_nll: f64,
    perplexity: f64,
}

impl MetricSummary {
    /// Returns the sum of `-ln(p)` over all observed targets.
    pub const fn total_surprise(&self) -> f64 {
        self.total_surprise
    }

    /// Returns the number of target probabilities in the aggregate.
    pub const fn target_count(&self) -> usize {
        self.target_count
    }

    /// Returns mean negative log-likelihood in nats per target.
    pub const fn mean_nll(&self) -> f64 {
        self.mean_nll
    }

    /// Returns the exponential of mean negative log-likelihood.
    pub const fn perplexity(&self) -> f64 {
        self.perplexity
    }
}

/// A rejected metric input or an error propagated from the bigram model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricError {
    /// No target exists from which to compute a mean.
    EmptyTargets,
    /// One assigned probability is non-finite or outside the closed unit interval.
    InvalidProbability { index: usize, probability: f64 },
    /// The bigram model rejected a token or count-table operation.
    Bigram(BigramError),
}

impl fmt::Display for MetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTargets => formatter.write_str("assigned probabilities must not be empty"),
            Self::InvalidProbability { index, probability } => write!(
                formatter,
                "assigned probability at index {index} must be finite and within [0, 1], got {probability}"
            ),
            Self::Bigram(error) => write!(formatter, "bigram scoring failed: {error}"),
        }
    }
}

impl Error for MetricError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bigram(err) => Some(err),
            Self::EmptyTargets | Self::InvalidProbability { .. } => None,
        }
    }
}

impl From<BigramError> for MetricError {
    fn from(error: BigramError) -> Self {
        Self::Bigram(error)
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct MetricsAccumulator {
    total_surprise: f64,
    target_count: usize,
}

impl MetricsAccumulator {
    fn observe(&mut self, probability: f64) {
        self.target_count += 1;
        if probability == 0.0 {
            self.total_surprise = f64::INFINITY;
        } else if probability == 1.0 {
            // Do nothing, as in this case total surprise should be increased by 0.0
        } else {
            self.total_surprise += -probability.ln();
        }
    }

    fn finish(self) -> Result<MetricSummary, MetricError> {
        if self.target_count == 0 {
            return Err(MetricError::EmptyTargets);
        }

        let mean_nll = self.total_surprise / self.target_count as f64;
        Ok(MetricSummary {
            total_surprise: self.total_surprise,
            target_count: self.target_count,
            mean_nll,
            perplexity: mean_nll.exp(),
        })
    }
}

/// Scores probabilities assigned to observed targets using natural logarithms.
///
/// The complete slice is validated before accumulation. Both `0.0` and `-0.0` are valid
/// impossible-evidence values and produce positive infinity without a clamp.
pub fn score_assigned_probabilities(probabilities: &[f64]) -> Result<MetricSummary, MetricError> {
    if probabilities.is_empty() {
        return Err(MetricError::EmptyTargets);
    }

    for (index, &probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(MetricError::InvalidProbability { index, probability });
        }
    }

    let mut accumulator = MetricsAccumulator::default();
    for &probability in probabilities {
        accumulator.observe(probability);
    }
    accumulator.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    mod metrics_accumulator {
        use super::*;

        mod fn_observe {
            use super::*;

            mod when_probability_is_zero {
                use super::*;

                #[test]
                fn it_sets_total_surprise_to_infinity() {
                    let mut accumulator = MetricsAccumulator::default();

                    accumulator.observe(0.0);
                    assert_eq!(accumulator.target_count, 1);
                    assert_eq!(accumulator.total_surprise, f64::INFINITY);
                }
            }

            mod when_probability_is_1 {
                use super::*;

                #[test]
                fn it_does_not_increase_total_surprise() {
                    let mut accumulator = MetricsAccumulator::default();

                    accumulator.observe(1.0);
                    assert_eq!(accumulator.target_count, 1);
                    assert_eq!(accumulator.total_surprise, 0.0);
                }
            }

            mod when_probability_is_between_0_and_1 {
                use super::*;

                #[test]
                fn it_adds_its_negative_ln_to_total_surprise() {
                    let mut accumulator = MetricsAccumulator::default();
                    accumulator.observe(0.5);
                    accumulator.observe(0.3);

                    assert_eq!(accumulator.target_count, 2);
                    assert_eq!(accumulator.total_surprise, -0.5f64.ln() - 0.3f64.ln());
                }
            }
        }

        mod fn_finish {
            use super::*;

            mod when_there_are_no_probabilities {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let accumulator = MetricsAccumulator::default();
                    assert_eq!(accumulator.finish(), Err(MetricError::EmptyTargets));
                }
            }

            mod when_there_are_probabilities {
                use super::*;

                #[test]
                fn it_returns_metric_summary() {
                    let mut accumulator = MetricsAccumulator::default();
                    accumulator.observe(0.5);
                    accumulator.observe(0.3);
                    let result = accumulator.finish();

                    let expected_total_targets = 2;
                    let expected_total_surprise = -0.5f64.ln() - 0.3f64.ln();
                    let expected_mean_nll = expected_total_surprise / expected_total_targets as f64;
                    assert_eq!(
                        result.as_ref().unwrap().total_surprise,
                        expected_total_surprise
                    );
                    assert_eq!(
                        result.as_ref().unwrap().target_count,
                        expected_total_targets
                    );
                    assert_eq!(result.as_ref().unwrap().mean_nll, expected_mean_nll);
                    assert_eq!(result.as_ref().unwrap().perplexity, expected_mean_nll.exp());
                }
            }
        }
    }

    mod fn_score_assigned_probabilities {
        use super::*;

        mod when_probabilities_are_empty {
            use super::*;

            #[test]
            fn it_returns_error() {
                let result = score_assigned_probabilities(&[]);

                assert_eq!(result, Err(MetricError::EmptyTargets));
            }
        }

        mod when_some_probability_is_infinite {
            use super::*;

            #[test]
            fn it_returns_error() {
                let result = score_assigned_probabilities(&[0.1, f64::INFINITY]);

                assert_eq!(
                    result,
                    Err(MetricError::InvalidProbability {
                        index: 1,
                        probability: f64::INFINITY
                    })
                );
            }
        }

        mod when_some_probability_is_gt_one {
            use super::*;

            #[test]
            fn it_returns_error() {
                let result = score_assigned_probabilities(&[0.1, 1.001]);

                assert_eq!(
                    result,
                    Err(MetricError::InvalidProbability {
                        index: 1,
                        probability: 1.001
                    })
                );
            }
        }

        mod when_some_probability_is_lt_zero {
            use super::*;

            #[test]
            fn it_returns_error() {
                let result = score_assigned_probabilities(&[0.1, -1.001]);

                assert_eq!(
                    result,
                    Err(MetricError::InvalidProbability {
                        index: 1,
                        probability: -1.001
                    })
                );
            }
        }

        mod when_all_is_ok {
            use super::*;

            #[test]
            fn it_calculates_metric_summary() {
                let probabilities = [0.1, 0.2, 0.3];
                let result = score_assigned_probabilities(&probabilities);

                let expected_total_surprise =
                    -probabilities[0].ln() - probabilities[1].ln() - probabilities[2].ln();
                let expected_mean_nll = expected_total_surprise / probabilities.len() as f64;
                assert_eq!(
                    result,
                    Ok(MetricSummary {
                        total_surprise: expected_total_surprise,
                        target_count: 3,
                        mean_nll: expected_mean_nll,
                        perplexity: expected_mean_nll.exp()
                    })
                )
            }
        }
    }
}
