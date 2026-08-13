//! Construction of shifted autoregressive input/target pairs.
//!
//! `CausalWindowConfig::windows` assumes its token slice contains exactly one document.
//! `EncodedCorpusPartitions` keeps encoded documents separate and lets callers open an iterator on
//! one document within one named partition at a time.

use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;

use crate::corpus::{CorpusPartitions, Partition};
use crate::tokenizer::bpe::BpeTokenizer;

/// Defines the gap between input and target tokens, thus describing how many tokens need to be
/// predicted. Setting this value to more than 1 is not recommended as it would mean the model
/// would need to fill this gap somehow by predicting that amount of tokens.
const TOKENS_TO_PREDICT: usize = 1;

/// A rejected configuration for causal-example construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalWindowConfigError {
    ZeroContextLength,
    ContextLengthOverflow,
    ZeroStride,
}

impl fmt::Display for CausalWindowConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroContextLength => "context length must be positive",
            Self::ContextLengthOverflow => {
                "context length is too large to require one additional source token"
            }
            Self::ZeroStride => "stride must be positive",
        })
    }
}

impl Error for CausalWindowConfigError {}

/// One complete input/target pair borrowed from a single document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalWindow<'a> {
    start: usize,
    input: &'a [u32],
    target: &'a [u32],
}

impl<'a> CausalWindow<'a> {
    /// Returns the input's zero-based position inside its document.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns exactly `context_length` source token IDs.
    pub fn input(&self) -> &'a [u32] {
        self.input
    }

    /// Returns the `T`-token source slice beginning one position after the input.
    pub fn target(&self) -> &'a [u32] {
        self.target
    }
}

/// A repeatable, exact-size iterator over one document's complete pairs.
#[derive(Clone, Debug)]
pub struct CausalWindows<'a> {
    tokens: &'a [u32],
    config: CausalWindowConfig,
    next_start: Option<usize>,
    remaining: usize,
}

impl<'a> Iterator for CausalWindows<'a> {
    type Item = CausalWindow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let input_start = self.next_start?;
        let input_end = input_start.checked_add(self.config.context_length)?;
        let target_start = input_start.checked_add(TOKENS_TO_PREDICT)?;
        let target_end = input_end.checked_add(TOKENS_TO_PREDICT)?;

        let input = self.tokens.get(input_start..input_end)?;
        let target = self.tokens.get(target_start..target_end)?;

        self.remaining -= 1;
        self.next_start = if self.remaining == 0 {
            None
        } else {
            input_start.checked_add(self.config.stride)
        };

        Some(CausalWindow {
            start: input_start,
            input,
            target,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for CausalWindows<'_> {}
impl FusedIterator for CausalWindows<'_> {}

/// The first selected suffix that is too short to emit a complete pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncompleteTail<'a> {
    start: usize,
    tokens: &'a [u32],
}

impl<'a> IncompleteTail<'a> {
    /// Returns the candidate position at which the too-short suffix begins.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the remaining source IDs; these may overlap earlier pairs.
    pub fn tokens(&self) -> &'a [u32] {
        self.tokens
    }
}

/// Config to determine the way we would like to slice window pairs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalWindowConfig {
    // T. Determines how many tokens we have in a single slice inside each window
    context_length: usize,
    // S. Determines the distance between windows
    stride: usize,
    // T + TOKENS_TO_PREDICT
    required_source_tokens: usize,
}

impl CausalWindowConfig {
    /// Validates the context length and distance between candidate starts.
    pub fn new(context_length: usize, stride: usize) -> Result<Self, CausalWindowConfigError> {
        if context_length == 0 {
            return Err(CausalWindowConfigError::ZeroContextLength);
        }
        if stride == 0 {
            return Err(CausalWindowConfigError::ZeroStride);
        }
        let Some(required_source_tokens) = context_length.checked_add(TOKENS_TO_PREDICT) else {
            return Err(CausalWindowConfigError::ContextLengthOverflow);
        };

        Ok(Self {
            context_length,
            stride,
            required_source_tokens,
        })
    }

    /// Returns the number of input IDs and target IDs in every emitted pair.
    pub fn context_length(self) -> usize {
        self.context_length
    }

    /// Returns the distance between consecutive candidate starts.
    pub fn stride(self) -> usize {
        self.stride
    }

    /// Returns the number of source IDs required to emit one shifted pair.
    pub fn required_source_tokens(self) -> usize {
        self.required_source_tokens
    }

    /// Counts complete (input, target) pairs.
    pub fn window_count(&self, document_length: usize) -> usize {
        if document_length < self.required_source_tokens {
            0
        } else {
            (document_length - self.required_source_tokens) / self.stride + 1
        }
    }

    /// Borrows every complete shifted pair selected inside one token slice.
    pub fn windows<'a>(&self, tokens: &'a [u32]) -> CausalWindows<'a> {
        CausalWindows {
            tokens,
            config: *self,
            next_start: Some(0),
            remaining: self.window_count(tokens.iter().len()),
        }
    }

    /// Returns the suffix at the first candidate start that cannot fill a pair.
    ///
    /// `None` means that the next candidate start lies at or beyond the end of the document. A
    /// returned suffix may overlap earlier complete pairs.
    pub fn incomplete_tail<'a>(&self, tokens: &'a [u32]) -> Option<IncompleteTail<'a>> {
        let window_count = self.window_count(tokens.len());
        let start = if window_count == 0 {
            0
        } else {
            (window_count - 1)
                .checked_mul(self.stride)?
                .checked_add(self.stride)?
        };

        if start < tokens.len() {
            Some(IncompleteTail {
                start,
                tokens: &tokens[start..],
            })
        } else {
            None
        }
    }
}

/// One owned token sequence with its stable document and partition identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedDocument {
    id: String,
    partition: Partition,
    token_ids: Vec<u32>,
}

impl EncodedDocument {
    /// Returns the stable source-document identity.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the frozen partition role inherited from the split manifest.
    pub fn partition(&self) -> Partition {
        self.partition
    }

    /// Returns the separately wrapped `[BOS, content..., EOS]` sequence.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Opens a fresh borrowed pair iterator without consuming this document.
    pub fn windows(&self, config: &CausalWindowConfig) -> CausalWindows<'_> {
        config.windows(&self.token_ids)
    }

    /// Reports the first candidate suffix that cannot fill a complete pair.
    pub fn incomplete_tail(&self, config: &CausalWindowConfig) -> Option<IncompleteTail<'_>> {
        config.incomplete_tail(&self.token_ids)
    }
}

/// Encoded documents kept in three disjoin owned collections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedCorpusPartitions {
    train: Vec<EncodedDocument>,
    validation: Vec<EncodedDocument>,
    test: Vec<EncodedDocument>,
}

impl EncodedCorpusPartitions {
    /// Applies one frozen tokenizer independently to every frozen source document.
    pub fn from_partitions(partitions: &CorpusPartitions<'_>, tokenizer: &BpeTokenizer) -> Self {
        let encode = |partition| {
            partitions
                .documents(partition)
                .iter()
                .map(|document| EncodedDocument {
                    id: document.id().to_owned(),
                    partition,
                    token_ids: tokenizer.encode_utf8_document(document.text()),
                })
                .collect()
        };

        Self {
            train: encode(Partition::Train),
            validation: encode(Partition::Validation),
            test: encode(Partition::Test),
        }
    }

    /// Returns only the separately encoded documents for one partition.
    pub fn documents(&self, partition: Partition) -> &[EncodedDocument] {
        match partition {
            Partition::Train => &self.train,
            Partition::Validation => &self.validation,
            Partition::Test => &self.test,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct CausalWindowsComparison<'a>(CausalWindows<'a>);

    impl<'a> PartialEq for CausalWindowsComparison<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.0.tokens == other.0.tokens
                && self.0.config == other.0.config
                && self.0.next_start == other.0.next_start
                && self.0.remaining == other.0.remaining
        }
    }

    mod causal_windows {
        use super::*;

        mod fn_next {
            use super::*;

            mod when_there_are_not_remaining_windows {
                use super::*;

                #[test]
                fn it_does_not_produce_any_result() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    let causal_windows = CausalWindows {
                        tokens: &[],
                        config,
                        next_start: Some(0),
                        remaining: 0,
                    };

                    assert_eq!(causal_windows.collect::<Vec<_>>(), vec![])
                }
            }

            mod when_remaining_windows_count_is_greater_than_actual_windows_number {
                use super::*;

                #[test]
                fn it_does_not_produce_any_result() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    let causal_windows = CausalWindows {
                        tokens: &[],
                        config,
                        next_start: Some(0),
                        remaining: 2,
                    };

                    assert_eq!(causal_windows.collect::<Vec<_>>(), vec![])
                }
            }

            mod when_remaining_windows_count_is_consistent_with_actual_windows_number {
                use super::*;

                #[test]
                fn it_iterates_through_result() {
                    let tokens = [1, 2, 3, 4, 5, 6];
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    let causal_windows = CausalWindows {
                        tokens: &tokens,
                        config,
                        next_start: Some(0),
                        remaining: 2,
                    };

                    assert_eq!(
                        causal_windows.collect::<Vec<_>>(),
                        vec![
                            CausalWindow {
                                start: 0,
                                input: &[1, 2, 3],
                                target: &[2, 3, 4]
                            },
                            CausalWindow {
                                start: 2,
                                input: &[3, 4, 5],
                                target: &[4, 5, 6]
                            }
                        ]
                    )
                }
            }

            mod when_input_start_is_on_the_edge_of_its_type_capacity {
                use super::*;

                #[test]
                fn it_does_not_produce_any_result() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    let causal_windows = CausalWindows {
                        tokens: &[1, 2, 3, 4, 5, 6],
                        config,
                        next_start: Some(usize::MAX),
                        remaining: 2,
                    };

                    assert_eq!(causal_windows.collect::<Vec<_>>(), vec![])
                }
            }

            mod when_target_slice_is_out_of_range {
                use super::*;

                #[test]
                fn it_iterates_only_through_available_combinations() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    let causal_windows = CausalWindows {
                        tokens: &[1, 2, 3, 4, 5],
                        config,
                        next_start: Some(0),
                        remaining: 2,
                    };

                    assert_eq!(
                        causal_windows.collect::<Vec<_>>(),
                        vec![CausalWindow {
                            start: 0,
                            input: &[1, 2, 3],
                            target: &[2, 3, 4]
                        }]
                    )
                }
            }
        }
    }

    mod causal_window_config {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_context_length_is_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        CausalWindowConfig::new(0, 1),
                        Err(CausalWindowConfigError::ZeroContextLength)
                    )
                }
            }

            mod when_stride_is_zero {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        CausalWindowConfig::new(1, 0),
                        Err(CausalWindowConfigError::ZeroStride)
                    )
                }
            }

            mod when_required_source_tokens_is_out_of_its_type_capacity {
                use super::*;

                #[test]
                fn it_returns_error() {
                    assert_eq!(
                        CausalWindowConfig::new(usize::MAX, 1),
                        Err(CausalWindowConfigError::ContextLengthOverflow)
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                #[test]
                fn it_returns_config() {
                    let result = CausalWindowConfig::new(3, 2);
                    assert!(result.is_ok());
                    assert_eq!(
                        result.unwrap(),
                        CausalWindowConfig {
                            context_length: 3,
                            stride: 2,
                            required_source_tokens: 3 + TOKENS_TO_PREDICT
                        }
                    )
                }
            }
        }

        mod fn_window_count {
            use super::*;

            mod when_document_length_is_less_than_required_source_tokens_number {
                use super::*;

                #[test]
                fn it_returns_zero() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    assert_eq!(config.window_count(3), 0)
                }
            }

            mod when_document_length_is_gte_than_required_source_tokens_number {
                use super::*;

                #[test]
                fn it_counts_pairs_number() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    assert_eq!(config.window_count(5), 1)
                }
            }
        }

        mod fn_windows {
            use super::*;

            #[test]
            fn it_returns_causal_windows_with_correct_attributes() {
                let config = CausalWindowConfig::new(3, 2).unwrap();
                let tokens = [1, 2, 3, 4];
                let result = config.windows(&[1, 2, 3, 4]);

                assert_eq!(result.tokens, &tokens);
                assert_eq!(result.config, config);
                assert_eq!(result.next_start, Some(0));
                assert_eq!(result.remaining, 1);
            }
        }

        mod fn_incomplete_tail {
            use super::*;

            mod when_empty_tokens_sequence_is_passed {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    assert_eq!(config.incomplete_tail(&[]), None)
                }
            }

            mod when_incomplete_tail_is_absent {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let config = CausalWindowConfig::new(3, 6).unwrap();
                    assert_eq!(config.incomplete_tail(&[1, 2, 3, 4, 5, 6]), None)
                }
            }

            mod when_incomplete_tail_is_present {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let config = CausalWindowConfig::new(3, 2).unwrap();
                    assert_eq!(
                        config.incomplete_tail(&[1, 2, 3, 4, 5, 6]),
                        Some(IncompleteTail {
                            start: 4,
                            tokens: &[5, 6],
                        })
                    )
                }
            }
        }
    }

    mod encoded_document {
        use super::*;

        mod fn_windows {
            use super::*;

            #[test]
            fn it_returns_causal_windows() {
                let token_ids = vec![1, 2, 3];
                let encoded_document = EncodedDocument {
                    id: "foo".to_owned(),
                    partition: Partition::Train,
                    token_ids: token_ids.clone(),
                };
                let config = CausalWindowConfig::new(2, 1).unwrap();

                assert_eq!(
                    CausalWindowsComparison(encoded_document.windows(&config)),
                    CausalWindowsComparison(CausalWindows {
                        tokens: &token_ids,
                        next_start: Some(0),
                        config,
                        remaining: 1
                    })
                )
            }
        }

        mod fn_incomplete_tail {
            use super::*;

            #[test]
            fn returns_incomplete_tail() {
                let token_ids = vec![1, 2, 3];
                let encoded_document = EncodedDocument {
                    id: "foo".to_owned(),
                    partition: Partition::Train,
                    token_ids: token_ids.clone(),
                };
                let config = CausalWindowConfig::new(2, 1).unwrap();
                assert_eq!(
                    encoded_document.incomplete_tail(&config),
                    Some(IncompleteTail {
                        start: 1,
                        tokens: &[2, 3]
                    })
                )
            }
        }
    }

    mod encoded_corpus_partitions {
        use super::*;

        mod fn_from_partitions {
            use super::*;
            use crate::corpus::{Corpus, SplitManifest};
            use crate::support::{SIMPLE_CORPUS_FILE, SIMPLE_CORPUS_MANIFEST};
            use crate::tokenizer::bpe_trainer::BpeTrainer;

            #[test]
            fn it_encodes_partitions_using_tokenizer() {
                let bpe_trainer = BpeTrainer::new(1);
                let corpus = Corpus::from_file(SIMPLE_CORPUS_FILE).unwrap();
                let manifest = SplitManifest::from_file(SIMPLE_CORPUS_MANIFEST).unwrap();
                let partitions = manifest.partition(&corpus).unwrap();
                let training = bpe_trainer.train(&partitions).unwrap();
                let tokenizer = BpeTokenizer::from_training(&training).unwrap();

                let result = EncodedCorpusPartitions::from_partitions(&partitions, &tokenizer);
                assert_eq!(
                    result.train,
                    vec![EncodedDocument {
                        id: "ru-river-dawn".to_owned(),
                        partition: Partition::Train,
                        token_ids: vec![0, 258, 113, 1]
                    }]
                );
                assert_eq!(
                    result.validation,
                    vec![EncodedDocument {
                        id: "en-river-sunrise".to_owned(),
                        partition: Partition::Validation,
                        token_ids: vec![0, 100, 99, 116, 1]
                    }]
                );
                assert_eq!(
                    result.test,
                    vec![EncodedDocument {
                        id: "es-river".to_owned(),
                        partition: Partition::Test,
                        token_ids: vec![0, 100, 99, 124, 1]
                    }]
                );
            }
        }
    }
}
