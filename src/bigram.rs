//! A checked count-based bigram language model.

use crate::corpus::Partition;
use crate::data::EncodedCorpusPartitions;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::{HashMap};
use std::fmt;

type TokenId = u32;
type Count = u64;

fn init_vec<T: Clone>(size: usize) -> Result<Vec<T>, BigramError> {
    let mut vec: Vec<T> = Vec::new();
    vec.try_reserve_exact(size)
        .map_err(|_| BigramError::TableTooLarge)?;
    Ok(vec)
}

fn init_vec_with_default<T: Clone>(size: usize, default_value: T) -> Result<Vec<T>, BigramError> {
    let mut vec = init_vec(size)?;
    vec.resize(size, default_value);
    Ok(vec)
}

/// A frozen transition table fitted from separately stored training documents.
#[derive(Clone, Debug, PartialEq)]
pub struct BigramModel {
    vocabulary_size: usize,
    offsets: Vec<usize>,
    next_tokens: Vec<TokenId>,
    counts: Vec<Count>,
    row_totals: Vec<Count>,
    alpha: f64,
    fitted_documents: usize,
    fitted_transitions: u64,
}

impl BigramModel {
    /// Fits one count per adjacent pair in each caller-supplied training document.
    ///
    /// Documents ramin separate: the final token of one document is never paired with the first
    /// token of the next document.
    /// Vocabulary size must be gte max(training_documents.flatten()) + 1
    pub fn fit_training_documents(
        vocabulary_size: usize,
        alpha: f64,
        training_documents: &[&[u32]],
    ) -> Result<Self, BigramError> {
        let mut vocabulary: Vec<HashMap<TokenId, Count, FxBuildHasher>> =
            init_vec_with_default(vocabulary_size, FxHashMap::default())?;

        if vocabulary_size == 0 {
            return Err(BigramError::EmptyVocabulary);
        }
        let smoothing_mass = alpha * vocabulary_size as f64;
        if !alpha.is_finite() || alpha <= 0.0 || !smoothing_mass.is_finite() {
            return Err(BigramError::InvalidAlpha);
        }

        let mut fitted_transitions: u64 = 0;
        let mut fitted_documents: usize = 0;

        for document in training_documents {
            fitted_documents = fitted_documents
                .checked_add(1)
                .ok_or(BigramError::TooManyDocuments)?;

            for pair in document.windows(2) {
                let from = pair[0] as TokenId;
                let to = pair[1] as TokenId;
                let count = vocabulary[from as usize].entry(to).or_insert(0);
                *count = count
                    .checked_add(1)
                    .ok_or(BigramError::TooManyTransitions)?;
                fitted_transitions = fitted_transitions
                    .checked_add(1)
                    .ok_or(BigramError::TooManyTransitions)?;
            }
        }

        let mut row_totals: Vec<Count> = init_vec_with_default(vocabulary_size, 0)?;
        let mut offsets: Vec<usize> = init_vec(vocabulary_size + 1)?;
        let mut next_tokens: Vec<TokenId> = init_vec(fitted_transitions as usize)?;
        let mut counts: Vec<Count> = init_vec(fitted_transitions as usize)?;

        offsets.push(0);

        for (token, per_token_transitions) in vocabulary.iter().enumerate() {
            let mut sorted_transitions = per_token_transitions
                .iter()
                .map(|(t_id, count)| (*t_id, *count))
                .collect::<Vec<_>>();
            sorted_transitions.sort_unstable_by_key(|t| t.0);
            let mut total_count = 0;

            for (next_token, next_token_count) in sorted_transitions.iter() {
                total_count += next_token_count;
                next_tokens.push(*next_token);
                counts.push(*next_token_count);
            }

            offsets.push(next_tokens.len());
            row_totals[token] = total_count;
        }

        Ok(Self {
            vocabulary_size,
            offsets,
            next_tokens,
            counts,
            row_totals,
            alpha,
            fitted_documents,
            fitted_transitions,
        })
    }

    /// Extract training documents from encoded partitions and fit them into the model
    pub fn fit_encoded_training_partition(
        vocabulary_size: usize,
        alpha: f64,
        partitions: &EncodedCorpusPartitions,
    ) -> Result<Self, BigramError> {
        let documents = partitions
            .documents(Partition::Train)
            .iter()
            .map(|document| document.token_ids())
            .collect::<Vec<_>>();

        Self::fit_training_documents(vocabulary_size, alpha, documents.as_slice())
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn fitted_documents(&self) -> usize {
        self.fitted_documents
    }

    pub fn fitted_transitions(&self) -> u64 {
        self.fitted_transitions
    }

    /// Check whether a token can fit the vocabulary.
    fn token_index(&self, token: u32) -> Result<usize, BigramError> {
        let index = token as usize;
        if index >= self.vocabulary_size {
            return Err(BigramError::TokenOutOfRange);
        }

        Ok(index)
    }

    /// Returns how many times `from` transitions into `to`
    pub fn count(&self, from: u32, to: u32) -> Result<u64, BigramError> {
        let from_index = self.token_index(from)?;
        self.token_index(to)?;

        let offset_start = self.offsets[from_index];
        let offset_end = self.offsets[from_index + 1];
        let tokens = &self.next_tokens[offset_start..offset_end];
        let to_offset = tokens
            .binary_search(&to)
            .map_err(|_| BigramError::TokenOutOfRange);
        match to_offset {
            Ok(offset) => Ok(self.counts[offset_start + offset]),
            Err(_) => Ok(0),
        }
    }

    /// Returns all transition counts of `from` into any other token
    pub fn counts_row(&self, from: u32) -> Result<&[u64], BigramError> {
        let from_index = self.token_index(from)?;

        let offset_start = self.offsets[from_index];
        let offset_end = self.offsets[from_index + 1];
        Ok(&self.counts[offset_start..offset_end])
    }

    /// Calculates sum in row counts of the given `from`
    pub fn row_total(&self, from: u32) -> Result<u64, BigramError> {
        self.counts_row(from)?
            .iter()
            .try_fold(0u64, |total, count| {
                total
                    .checked_add(*count)
                    .ok_or(BigramError::TooManyTransitions)
            })
    }

    /// Calculate smoothed probably of transition `from` => `to` based on trusted indexes
    pub fn smoothed_probability_for_checked_indices(
        &self,
        from: usize,
        to: usize,
    ) -> Result<f64, BigramError> {
        let denominator =
            self.row_total(from as TokenId)? as f64 + self.alpha * self.vocabulary_size as f64;
        let numerator = self.count(from as TokenId, to as TokenId)? as f64 + self.alpha;

        Ok(numerator / denominator)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigramError {
    EmptyVocabulary,
    InvalidAlpha,
    TableTooLarge,
    TooManyDocuments,
    TooManyTransitions,
    TokenOutOfRange,
}

impl fmt::Display for BigramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyVocabulary => "vocabulary must not be empty",
            Self::InvalidAlpha => "smoothing alpha must be finite and positive",
            Self::TableTooLarge => "the square count table does not fit in memory",
            Self::TooManyDocuments => "the fitted document count overflowed usize",
            Self::TooManyTransitions => "a transition count overflowed u64",
            Self::TokenOutOfRange => "token ID is outside the vocabulary",
        })
    }
}

impl std::error::Error for BigramError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::bpe::{BOS_TOKEN_ID, EOS_TOKEN_ID};

    fn model_sample() -> Result<BigramModel, BigramError> {
        let documents: Vec<Vec<u32>> = vec![
            vec![BOS_TOKEN_ID, 3, 2, 2, 4, EOS_TOKEN_ID],
            vec![BOS_TOKEN_ID, EOS_TOKEN_ID],
            vec![BOS_TOKEN_ID, 2, 4, 4, 3, 5, EOS_TOKEN_ID],
        ];
        let vocabulary_size = 6; // should be equal to max token id + 1
        let alpha = 1.0;
        BigramModel::fit_training_documents(
            vocabulary_size,
            alpha,
            documents
                .iter()
                .map(|doc| doc.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    mod bigram_model {
        use super::*;

        mod fn_fit_training_documents {
            use super::*;

            mod when_vocabulary_size_is_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(0, 1.0, &[]),
                        Err(BigramError::EmptyVocabulary)
                    )
                }
            }

            mod when_alpha_is_infinite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(2, f64::INFINITY, &[]),
                        Err(BigramError::InvalidAlpha)
                    )
                }
            }

            mod when_alpha_is_negative {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(2, -0.2, &[]),
                        Err(BigramError::InvalidAlpha)
                    )
                }
            }

            mod when_smoothing_coefficient_is_not_finite {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(23, f64::MAX, &[]),
                        Err(BigramError::InvalidAlpha)
                    )
                }
            }

            mod when_vocabulary_size_is_too_large {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(usize::MAX, 2.0, &[]),
                        Err(BigramError::TableTooLarge)
                    )
                }
            }

            mod when_there_is_not_enough_memory_to_reserve {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        BigramModel::fit_training_documents(usize::MAX.isqrt() - 1, 2.0, &[]),
                        Err(BigramError::TableTooLarge)
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_counts_transitions_of_token_pairs_in_the_given_documents() {
                    let result = model_sample();

                    assert!(result.is_ok());
                    assert_eq!(result.as_ref().unwrap().vocabulary_size, 6);
                    assert_eq!(result.as_ref().unwrap().alpha, 1.0);
                    assert_eq!(result.as_ref().unwrap().fitted_documents, 3);
                    assert_eq!(result.as_ref().unwrap().fitted_transitions, 12);
                    assert_eq!(
                        result.as_ref().unwrap().counts,
                        vec![
                            1, 1, 1, // BOS_TOKEN_ID
                            // EOS_TOKEN_ID does not have any transitions
                            1, 2, // 2
                            1, 1, // 3
                            1, 1, 1, // 4
                            1, // 5
                        ]
                    );
                    assert_eq!(
                        result.as_ref().unwrap().offsets,
                        vec![0, 3, 3, 5, 7, 10, 11]
                    );
                    assert_eq!(
                        result.as_ref().unwrap().next_tokens,
                        vec![
                            1, 2, 3, // BOS_TOKEN_ID transitions
                            // EOS_TOKEN_ID does not have any transitions
                            2, 4, // 2 transitions
                            2, 5, // 3 transitions
                            1, 3, 4, // 4 transitions
                            1, // 5 transitions
                        ]
                    );
                    assert_eq!(result.as_ref().unwrap().row_totals, vec![3, 0, 3, 2, 3, 1]);
                }
            }
        }

        mod fn_fit_encoded_training_partition {
            use super::*;
            use crate::corpus::{Corpus, SplitManifest};
            use crate::support::{SIMPLE_CORPUS_FILE, SIMPLE_CORPUS_MANIFEST};
            use crate::tokenizer::bpe::BpeTokenizer;
            use crate::tokenizer::bpe_trainer::BpeTrainer;

            #[test]
            fn it_counts_transitions_of_token_pairs_from_encoded_partitions() {
                let bpe_trainer = BpeTrainer::new(1);
                let corpus = Corpus::from_file(SIMPLE_CORPUS_FILE).unwrap();
                let manifest = SplitManifest::from_file(SIMPLE_CORPUS_MANIFEST).unwrap();
                let partitions = manifest.partition(&corpus).unwrap();
                let training = bpe_trainer.train(&partitions).unwrap();
                let tokenizer = BpeTokenizer::from_training(&training).unwrap();
                // We have here 1 document - [BOS, 258, 113, EOS]
                let encoded_partitions =
                    EncodedCorpusPartitions::from_partitions(&partitions, &tokenizer);

                let vocabulary_size = 259; // should be equal to max token id + 1
                let alpha = 1.0;

                let result = BigramModel::fit_encoded_training_partition(
                    vocabulary_size,
                    alpha,
                    &encoded_partitions,
                );

                assert!(result.is_ok());
                assert_eq!(result.as_ref().unwrap().vocabulary_size, vocabulary_size);
                assert_eq!(result.as_ref().unwrap().alpha, alpha);
                assert_eq!(result.as_ref().unwrap().fitted_documents, 1);
                assert_eq!(result.as_ref().unwrap().fitted_transitions, 3);
                assert_eq!(
                    result.as_ref().unwrap().counts,
                    vec![
                        1, // BOS_TOKEN_ID
                        // EOS_TOKEN_ID does not have any counts
                        1, // "fo", 258 ID
                        1, // "o", 113 ID
                    ]
                );
                let mut expected_offsets = vec![0, 1];
                expected_offsets.append(&mut vec![1; 112]);
                expected_offsets.append(&mut vec![2; 145]);
                expected_offsets.append(&mut vec![3]);
                assert_eq!(result.as_ref().unwrap().offsets, expected_offsets);
                assert_eq!(
                    result.as_ref().unwrap().next_tokens,
                    vec![
                        258, // BOS_TOKEN_ID transitions
                        // EOS_TOKEN_ID does not have any transitions
                        1,   // 113 ID transitions
                        113, // 258 ID transitions
                    ]
                );
                let mut expected_row_totals = vec![1];
                expected_row_totals.append(&mut vec![0; 112]);
                expected_row_totals.append(&mut vec![1]);
                expected_row_totals.append(&mut vec![0; 144]);
                expected_row_totals.append(&mut vec![1]);
                assert_eq!(result.as_ref().unwrap().row_totals, expected_row_totals);
            }
        }

        mod fn_count {
            use super::*;

            #[test]
            fn it_counts_times_of_transitions() {
                let model = model_sample().unwrap();

                assert_eq!(model.count(4, 4), Ok(1));
                assert_eq!(model.count(2, 4), Ok(2));
                assert_eq!(model.count(2, 5), Ok(0));
                assert_eq!(model.count(6, 5), Err(BigramError::TokenOutOfRange));
            }
        }

        mod fn_counts_row {
            use super::*;

            #[test]
            fn it_returns_all_transitions_of_the_given_token() {
                let model = model_sample().unwrap();

                assert_eq!(model.counts_row(BOS_TOKEN_ID), Ok([1, 1, 1].as_slice()));
                assert_eq!(model.counts_row(EOS_TOKEN_ID), Ok([].as_slice()));
                assert_eq!(model.counts_row(2), Ok([1, 2].as_slice()));
                assert_eq!(model.counts_row(6), Err(BigramError::TokenOutOfRange));
            }
        }

        mod fn_row_total {
            use super::*;

            #[test]
            fn it_counts_row_total_of_the_given_token() {
                let model = model_sample().unwrap();

                assert_eq!(model.row_total(BOS_TOKEN_ID), Ok(3));
                assert_eq!(model.row_total(EOS_TOKEN_ID), Ok(0));
                assert_eq!(model.row_total(2), Ok(3));
                assert_eq!(model.row_total(6), Err(BigramError::TokenOutOfRange));
            }
        }

        mod fn_smoothed_probability_for_checked_indices {
            use super::*;

            #[test]
            fn it_returns_smoothed_probability_of_the_transition_of_the_couple() {
                let model = model_sample().unwrap();

                assert_eq!(
                    model.smoothed_probability_for_checked_indices(BOS_TOKEN_ID as usize, 3),
                    Ok(2.0 / 9.0)
                );
                assert_eq!(
                    model.smoothed_probability_for_checked_indices(EOS_TOKEN_ID as usize, 3),
                    Ok(1.0 / 6.0)
                );
                assert_eq!(
                    model.smoothed_probability_for_checked_indices(3, 3),
                    Ok(1.0 / 8.0)
                );
                assert_eq!(
                    model.smoothed_probability_for_checked_indices(6, 0),
                    Err(BigramError::TokenOutOfRange)
                );
            }
        }
    }
}
