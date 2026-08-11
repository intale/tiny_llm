//! Application and exact reversal of a frozne byte-pair rank table.
//!
//! Layout version 1 reserves `0` and `1` for document controls. Content IDs are the [`BpeTraining`]
//! training-space IDs shifted by two, so every possible byt keeps a lossless fallback
//! representation.

use super::bpe_trainer::{
    BYTE_TOKEN_COUNT, BpeTraining, TokenPair, replace_pair_left_to_right, training_vocabulary_base,
};
use log::error;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Serialized layout version
pub const TOKENIZER_LAYOUT_VERSION: u32 = 1;

/// Marker of the beginning of one encoded document
pub const BOS_TOKEN_ID: u32 = 0;
/// Marker of the end of one encoded document
pub const EOS_TOKEN_ID: u32 = 1;

/// Maps every [`BpeTraining`] training-space ID into the content namespace.
pub const CONTENT_ID_OFFSET: u32 = 2;
/// First content ID representing one raw byte.
pub const FIRST_BYTE_TOKEN_ID: u32 = CONTENT_ID_OFFSET;
/// Last content ID representing one raw byte.
pub const LAST_BYTE_TOKEN_ID: u32 = CONTENT_ID_OFFSET + BYTE_TOKEN_COUNT - 1;
/// Content ID assigned to merge rank zero.
pub const FIRST_MERGE_TOKEN_ID: u32 = LAST_BYTE_TOKEN_ID + 1;

/// Stable, typed diagnostics for tokenizer construction and decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BpeTokenizerError {
    /// The requested final namespace cannot be represented `u32` IDs.
    LayoutOverflow { merge_count: usize },
    /// A frozen pair refers to a token unavailable at that rank.
    UnknownMergeOperand { rank: usize, token_id: u32 },
    /// A pair appears at more than one rank.
    DuplicateMergePair { rank: usize, left: u32, right: u32 },
    /// BOS and EOS was passed to content-only decoding.
    ControlTokenInContent { position: usize, token_id: u32 },
    /// A token ID has no byte expansion in this tokenizer.
    UnknownToken { position: usize, token_id: u32 },
    /// A document cannot contain both required endpoint controls.
    DocumentTooShort { length: usize },
    /// The first document token is not BOS.
    ExpectedBos { found: u32 },
    /// The first document token is not EOS.
    ExpectedEos { found: u32 },
    /// A document control appeared between its endpoints.
    InteriorControlToken { position: usize, token_id: u32 },
    /// Exact bytes were recovered but are not a valid UTF-8 string.
    InvalidUtf8 {
        valid_up_to: usize,
        error_len: Option<usize>,
    },
}

impl fmt::Display for BpeTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutOverflow { merge_count } => write!(
                formatter,
                "{merge_count} merge ranks do not fit tokenizer layout version {TOKENIZER_LAYOUT_VERSION}"
            ),
            Self::UnknownMergeOperand { rank, token_id } => write!(
                formatter,
                "merge rank {rank} references unavailable training token {token_id}"
            ),
            Self::DuplicateMergePair { rank, left, right } => write!(
                formatter,
                "merge rank {rank} repeats training pair ({left},{right})"
            ),
            Self::ControlTokenInContent { position, token_id } => write!(
                formatter,
                "control token {token_id} is not allowed in content at position {position}"
            ),
            Self::UnknownToken { position, token_id } => write!(
                formatter,
                "token {token_id} at position {position} is outside this tokenizer vocabulary"
            ),
            Self::DocumentTooShort { length } => write!(
                formatter,
                "document token sequence of length {length} cannot contain BOS and EOS"
            ),
            Self::ExpectedBos { found } => {
                write!(
                    formatter,
                    "expected BOS token 0, found {found} at position 0"
                )
            }
            Self::ExpectedEos { found } => write!(
                formatter,
                "expected EOS token 1, found {found} at the final position"
            ),
            Self::InteriorControlToken { position, token_id } => write!(
                formatter,
                "document control token {token_id} is not allowed at position {position}"
            ),
            Self::InvalidUtf8 { valid_up_to, .. } => write!(
                formatter,
                "decoded bytes are not valid UTF-8 at byte {valid_up_to}"
            ),
        }
    }
}

impl Error for BpeTokenizerError {}

/// The fixed fields and vocabulary extent of tokenizer layout version 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerLayout {
    merge_count: usize,
    vocabulary_size: usize,
}

impl TokenizerLayout {
    /// Validates that every content symbol can be represented by a `u32` ID.
    pub fn new(merge_count: usize) -> Result<Self, BpeTokenizerError> {
        let available_vocabulary_size = u32::MAX - FIRST_MERGE_TOKEN_ID;
        if merge_count > available_vocabulary_size as usize {
            return Err(BpeTokenizerError::LayoutOverflow { merge_count });
        }

        Ok(Self {
            merge_count,
            vocabulary_size: merge_count + FIRST_MERGE_TOKEN_ID as usize,
        })
    }

    /// Returns the serialized layout version.
    pub const fn version(&self) -> u32 {
        TOKENIZER_LAYOUT_VERSION
    }

    /// Returns the number of frozen merge ranks.
    pub fn merge_count(&self) -> usize {
        self.merge_count
    }

    /// Returns the complete number of control and content IDs.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Maps a raw byte to its one-byte content token.
    pub const fn byte_to_token_id(&self, byte: u8) -> u32 {
        FIRST_BYTE_TOKEN_ID + byte as u32
    }

    /// Returns the final content ID assigned to a valid zero-based rank.
    pub fn merge_token_id(&self, rank: usize) -> Option<u32> {
        if rank >= self.merge_count {
            return None;
        }

        usize::try_from(FIRST_MERGE_TOKEN_ID)
            .ok()
            .and_then(|base| base.checked_add(rank))
            .and_then(|token| u32::try_from(token).ok())
    }
}

/// One frozen rule expressed in both trainer and final content namespaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeMergeRule {
    rank: usize,
    training_pair: TokenPair,
    training_token_id: u32,
    content_pair: TokenPair,
    content_token_id: u32,
}

impl BpeMergeRule {
    /// Returns the zero-based application order.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the training pair before the layout offset.
    pub fn training_pair(&self) -> TokenPair {
        self.training_pair
    }

    /// Returns the training token assigned to this rank.
    pub fn training_token_id(&self) -> u32 {
        self.training_token_id
    }

    /// Returns the pair used while encoding final content IDs
    pub fn content_pair(&self) -> TokenPair {
        self.content_pair
    }

    /// Returns the final content ID assigned to this rank.
    pub fn content_token_id(&self) -> u32 {
        self.content_token_id
    }
}

/// One rank that changed a particular input while encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeMergeApplication {
    rank: usize,
    replacements: usize,
    before: Vec<u32>,
    after: Vec<u32>,
}

impl BpeMergeApplication {
    /// Returns the applied rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns how many non-overlapping occurrences changed.
    pub fn replacements(&self) -> usize {
        self.replacements
    }

    /// Returns the input to this rank.
    pub fn before(&self) -> &[u32] {
        &self.before
    }

    /// Returns the output from this rank.
    pub fn after(&self) -> &[u32] {
        &self.after
    }
}

/// Inspectable evidence for one ranked content encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeEncodingTrace {
    initial_tokens: Vec<u32>,
    applications: Vec<BpeMergeApplication>,
    content_tokens: Vec<u32>,
}

impl BpeEncodingTrace {
    /// Returns byte IDs before any merge rank runs.
    pub fn initial_tokens(&self) -> &[u32] {
        &self.initial_tokens
    }

    /// Returns only ranks that changed this input, in application order.
    pub fn applications(&self) -> &[BpeMergeApplication] {
        &self.applications
    }

    /// Returns the canonical content sequence after all ranks.
    pub fn content_tokens(&self) -> &[u32] {
        &self.content_tokens
    }
}

/// An owned, deterministic byte-level BPE tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokenizer {
    layout: TokenizerLayout,
    merge_rules: Vec<BpeMergeRule>,
    training_vocabulary: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    /// Copies the ranks and byte expansins from a validated [`BpeTraining`] result.
    pub fn from_training(training: &BpeTraining) -> Result<Self, BpeTokenizerError> {
        let layout = TokenizerLayout::new(training.rules().len())?;
        let merge_rules = training
            .rules()
            .iter()
            .map(|rule| {
                let training_pair = rule.pair();
                BpeMergeRule {
                    rank: rule.rank(),
                    training_pair,
                    training_token_id: rule.token_id(),
                    content_pair: TokenPair::new(
                        training_pair.left() + CONTENT_ID_OFFSET,
                        training_pair.right() + CONTENT_ID_OFFSET,
                    ),
                    content_token_id: rule.token_id() + CONTENT_ID_OFFSET,
                }
            })
            .collect();

        Ok(Self {
            layout,
            merge_rules,
            training_vocabulary: training.vocabulary().to_vec(),
        })
    }

    /// Builds a frozen tokenizer from ordered training-space pairs.
    pub fn from_merge_pairs(pairs: &[TokenPair]) -> Result<Self, BpeTokenizerError> {
        let layout = TokenizerLayout::new(pairs.len())?;
        let mut training_vocabulary = training_vocabulary_base();
        let mut merge_rules = Vec::with_capacity(pairs.len());
        let mut seen_pairs = BTreeSet::new();

        for (rank, &training_pair) in pairs.iter().enumerate() {
            let rank_id = (rank as u32).checked_add(BYTE_TOKEN_COUNT).ok_or(
                BpeTokenizerError::LayoutOverflow {
                    merge_count: pairs.len(),
                },
            )?;
            for operand in [training_pair.left(), training_pair.right()] {
                if operand >= rank_id {
                    return Err(BpeTokenizerError::UnknownMergeOperand {
                        rank,
                        token_id: operand,
                    });
                }
            }
            if !seen_pairs.insert(training_pair) {
                return Err(BpeTokenizerError::DuplicateMergePair {
                    rank,
                    left: training_pair.left(),
                    right: training_pair.right(),
                });
            }

            let left_bytes = training_vocabulary
                .get(training_pair.left() as usize)
                .ok_or(BpeTokenizerError::UnknownMergeOperand {
                    rank,
                    token_id: training_pair.left(),
                })?;
            let right_bytes = training_vocabulary
                .get(training_pair.right() as usize)
                .ok_or(BpeTokenizerError::UnknownMergeOperand {
                    rank,
                    token_id: training_pair.right(),
                })?;
            let mut merged_bytes = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            merged_bytes.extend_from_slice(left_bytes);
            merged_bytes.extend_from_slice(right_bytes);

            let content_pair = TokenPair::new(
                training_pair.left().checked_add(CONTENT_ID_OFFSET).ok_or(
                    BpeTokenizerError::LayoutOverflow {
                        merge_count: pairs.len(),
                    },
                )?,
                training_pair.right().checked_add(CONTENT_ID_OFFSET).ok_or(
                    BpeTokenizerError::LayoutOverflow {
                        merge_count: pairs.len(),
                    },
                )?,
            );
            let content_token_id = rank_id.checked_add(CONTENT_ID_OFFSET).ok_or(
                BpeTokenizerError::LayoutOverflow {
                    merge_count: pairs.len(),
                },
            )?;

            merge_rules.push(BpeMergeRule {
                rank,
                training_pair,
                training_token_id: rank_id,
                content_pair,
                content_token_id,
            });
            training_vocabulary.push(merged_bytes);
        }

        Ok(Self {
            layout,
            merge_rules,
            training_vocabulary,
        })
    }

    /// Returns the validated layout extent.
    pub fn layout(&self) -> TokenizerLayout {
        self.layout
    }

    /// Returns every rule in ascending rank order.
    pub fn merge_rules(&self) -> &[BpeMergeRule] {
        &self.merge_rules
    }

    /// Returns a content token's byte expansion; controls and unknown IDs have none.
    pub fn token_bytes(&self, content_token_id: u32) -> Option<&[u8]> {
        let training_id = content_token_id.checked_sub(CONTENT_ID_OFFSET)?;
        self.training_vocabulary
            .get(training_id as usize)
            .map(Vec::as_slice)
    }

    /// Converts bytes sequence into content tokens sequence
    fn initial_content_tokens(&self, bytes: &[u8]) -> Vec<u32> {
        bytes
            .iter()
            .map(|byte| self.layout.byte_to_token_id(*byte))
            .collect()
    }

    fn apply_ranked_merges(
        &self,
        mut content_tokens: Vec<u32>,
        mut observe: impl FnMut(&BpeMergeRule, usize, &[u32], &[u32]),
    ) -> Vec<u32> {
        for rule in &self.merge_rules {
            let before = content_tokens;
            let (after, replacements) =
                replace_pair_left_to_right(&before, rule.content_pair, rule.content_token_id);
            if replacements > 0 {
                observe(rule, replacements, &before, &after);
            }
            content_tokens = after;
        }
        content_tokens
    }

    /// Encodes bytes and records every rank that changed the sequence.
    pub fn encode_content_with_trace(&self, bytes: &[u8]) -> BpeEncodingTrace {
        let initial_tokens = self.initial_content_tokens(bytes);
        let mut applications = Vec::new();
        let content_tokens = self.apply_ranked_merges(
            initial_tokens.clone(),
            |rule, replacements, before, after| {
                applications.push(BpeMergeApplication {
                    rank: rule.rank,
                    replacements,
                    before: before.to_vec(),
                    after: after.to_vec(),
                })
            },
        );

        BpeEncodingTrace {
            initial_tokens,
            applications,
            content_tokens,
        }
    }

    /// Encodes arbitrary bytes into the canonical rank-ordered content sequence.
    pub fn encode_content(&self, bytes: &[u8]) -> Vec<u32> {
        let initial_tokens = self.initial_content_tokens(bytes);
        self.apply_ranked_merges(initial_tokens, |_, _, _, _| {})
    }

    /// Encodes a valid UTF-8 string through the same byte boundary.
    pub fn encode_utf8(&self, text: &str) -> Vec<u32> {
        self.encode_content(text.as_bytes())
    }

    /// Encodes content first, then adds controls that never enter a merge pass.
    pub fn encode_document(&self, bytes: &[u8]) -> Vec<u32> {
        let content = self.encode_content(bytes);
        let mut document = Vec::with_capacity(content.len() + 2);
        document.push(BOS_TOKEN_ID);
        document.extend(content);
        document.push(EOS_TOKEN_ID);
        document
    }

    /// Encodes a UTF-8 document and adds its endpoint controls.
    pub fn encode_utf8_document(&self, text: &str) -> Vec<u32> {
        self.encode_document(text.as_bytes())
    }

    fn decode_tokens(
        &self,
        tokens: &[u32],
        position_offset: usize,
    ) -> Result<Vec<u8>, BpeTokenizerError> {
        let mut bytes = Vec::new();
        for (index, &token_id) in tokens.iter().enumerate() {
            let expansion = self
                .token_bytes(token_id)
                .ok_or(BpeTokenizerError::UnknownToken {
                    position: index + position_offset,
                    token_id,
                })?;
            bytes.extend_from_slice(expansion);
        }

        Ok(bytes)
    }

    /// Validates one wrapped document and recovers its exact content bytes.
    pub fn decode_document(&self, document: &[u32]) -> Result<Vec<u8>, BpeTokenizerError> {
        if document.len() < 2 {
            return Err(BpeTokenizerError::DocumentTooShort {
                length: document.len(),
            });
        }
        if document[0] != BOS_TOKEN_ID {
            return Err(BpeTokenizerError::ExpectedBos { found: document[0] });
        }
        let last = document.len() - 1;
        if document[last] != EOS_TOKEN_ID {
            return Err(BpeTokenizerError::ExpectedEos {
                found: document[last],
            });
        }
        for (position, &token_id) in document[1..last].iter().enumerate() {
            if token_id == BOS_TOKEN_ID || token_id == EOS_TOKEN_ID {
                return Err(BpeTokenizerError::InteriorControlToken {
                    position: position + 1,
                    token_id,
                });
            }
        }
        self.decode_tokens(&document[1..last], 1)
    }

    /// Concatenates content-token expansions without interpreting them as text.
    pub fn decode_content(&self, content: &[u32]) -> Result<Vec<u8>, BpeTokenizerError> {
        for (position, &token_id) in content.iter().enumerate() {
            if token_id == BOS_TOKEN_ID || token_id == EOS_TOKEN_ID {
                return Err(BpeTokenizerError::ControlTokenInContent { position, token_id });
            }
        }
        self.decode_tokens(content, 0)
    }

    /// Decodes content bytes, then requires the result to be valid UTF-8.
    pub fn decode_content_utf8(&self, content: &[u32]) -> Result<String, BpeTokenizerError> {
        strict_utf8(self.decode_content(content)?)
    }

    /// Decodes a wrapped document, then requires the result to be valid UTF-8.
    pub fn decode_document_utf8(&self, document: &[u32]) -> Result<String, BpeTokenizerError> {
        strict_utf8(self.decode_document(document)?)
    }
}

fn strict_utf8(bytes: Vec<u8>) -> Result<String, BpeTokenizerError> {
    String::from_utf8(bytes).map_err(|error| {
        let utf8 = error.utf8_error();
        BpeTokenizerError::InvalidUtf8 {
            valid_up_to: utf8.valid_up_to(),
            error_len: utf8.error_len(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    mod tokenizer_layout {
        use super::*;

        mod fn_new {
            use super::*;

            mod when_vocabulary_size_exceeds_the_limit {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = TokenizerLayout::new(u32::MAX as usize);
                    assert_eq!(
                        result,
                        Err(BpeTokenizerError::LayoutOverflow {
                            merge_count: u32::MAX as usize
                        })
                    )
                }
            }

            mod when_vocabulary_does_not_exceed_the_limit {
                use super::*;

                #[test]
                fn it_returns_tokenizer_layout() {
                    let result = TokenizerLayout::new(1234);
                    assert_eq!(
                        result,
                        Ok(TokenizerLayout {
                            merge_count: 1234,
                            vocabulary_size: 1234 + FIRST_MERGE_TOKEN_ID as usize
                        })
                    )
                }
            }
        }

        mod fn_byte_to_token_id {
            use super::*;

            #[test]
            fn it_converts_byte_into_token() {
                let layout = TokenizerLayout::new(1234).unwrap();
                assert_eq!(layout.byte_to_token_id(23), FIRST_BYTE_TOKEN_ID + 23)
            }
        }

        mod fn_merge_token_id {
            use super::*;

            mod when_rank_is_gt_than_merge_count {
                use super::*;

                #[test]
                fn it_returns_none() {
                    let layout = TokenizerLayout::new(12).unwrap();
                    assert_eq!(layout.merge_token_id(13), None)
                }
            }

            mod when_rank_is_gte_than_merge_count {
                use super::*;

                #[test]
                fn it_calculates_rank_content_id() {
                    let merge_count = 12;
                    let rank = 10;
                    let layout = TokenizerLayout::new(merge_count).unwrap();
                    assert_eq!(
                        layout.merge_token_id(rank),
                        Some((FIRST_MERGE_TOKEN_ID as usize + rank) as u32)
                    )
                }
            }
        }
    }

    mod bpe_tokenizer {
        use super::*;
        use crate::corpus::{Corpus, SplitManifest};
        use crate::support::{SIMPLE_CORPUS_FILE, SIMPLE_CORPUS_MANIFEST};
        use crate::tokenizer::bpe_trainer::BpeTrainer;

        // Latest training token id, based on SIMPLE_CORPUS_FILE and SIMPLE_CORPUS_MANIFEST
        const LATEST_TRAINED_TOKEN_ID: u32 = 257;

        fn bpe_training() -> BpeTraining {
            let max_merges = 2;
            let bpe_trainer = BpeTrainer::new(max_merges);
            let corpus = Corpus::from_file(SIMPLE_CORPUS_FILE).unwrap();
            let manifest = SplitManifest::from_file(SIMPLE_CORPUS_MANIFEST).unwrap();
            let partitions = manifest.partition(&corpus).unwrap();
            bpe_trainer.train(&partitions).unwrap()
        }

        mod fn_from_training {
            use super::*;

            #[test]
            fn it_creates_tokenizer_from_training_data() {
                let training = bpe_training();

                let expected_layout = TokenizerLayout::new(training.rules().len()).unwrap();
                let expected_training_vocabulary = training.vocabulary().to_vec();
                let expected_merge_rules = vec![
                    BpeMergeRule {
                        rank: 0,
                        training_pair: TokenPair::new(102, 111),
                        training_token_id: 256,
                        content_pair: TokenPair::new(
                            102 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET,
                        ),
                        content_token_id: 256 + CONTENT_ID_OFFSET,
                    },
                    BpeMergeRule {
                        rank: 1,
                        training_pair: TokenPair::new(256, 111),
                        training_token_id: 257,
                        content_pair: TokenPair::new(
                            256 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET,
                        ),
                        content_token_id: 257 + CONTENT_ID_OFFSET,
                    },
                ];
                let result = BpeTokenizer::from_training(&training);
                assert!(result.is_ok());
                assert_eq!(result.as_ref().unwrap().layout, expected_layout);
                assert_eq!(
                    result.as_ref().unwrap().training_vocabulary,
                    expected_training_vocabulary
                );
                assert_eq!(result.as_ref().unwrap().merge_rules, expected_merge_rules);
            }
        }

        mod fn_from_merge_pairs {
            use super::*;

            #[test]
            fn it_creates_tokenizer_from_token_pairs() {
                let training = bpe_training();
                let training_pairs = training
                    .rules()
                    .iter()
                    .map(|rule| rule.pair())
                    .collect::<Vec<_>>();

                let result = BpeTokenizer::from_merge_pairs(&training_pairs);
                assert!(result.is_ok());
                assert_eq!(result, BpeTokenizer::from_training(&training));
            }
        }

        mod fn_token_bytes {
            use super::*;

            #[test]
            fn it_expands_content_token_into_bytes_sequence() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                assert_eq!(
                    bpe_tokenizer.token_bytes(259),
                    Some([102, 111, 111].as_slice())
                )
            }
        }

        mod fn_initial_content_tokens {
            use super::*;

            #[test]
            fn it_coverts_bytes_sequence_to_content_tokens_sequence() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                assert_eq!(
                    bpe_tokenizer.initial_content_tokens([1, 2, 3].as_slice()),
                    vec![
                        1 + FIRST_BYTE_TOKEN_ID,
                        2 + FIRST_BYTE_TOKEN_ID,
                        3 + FIRST_BYTE_TOKEN_ID
                    ]
                )
            }
        }

        mod fn_apply_ranked_merges {
            use super::*;

            #[test]
            fn apply_ranked_merges() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                assert_eq!(
                    bpe_tokenizer.apply_ranked_merges(
                        vec![
                            102 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET
                        ],
                        |_, _, _, _| {}
                    ),
                    vec![LATEST_TRAINED_TOKEN_ID + CONTENT_ID_OFFSET]
                )
            }
        }

        mod fn_encode_content_with_trace {
            use super::*;

            #[test]
            fn it_encodes_bytes_into_content_with_trace() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                let expected_merge_applications = vec![
                    BpeMergeApplication {
                        rank: 0,
                        replacements: 1,
                        before: vec![
                            102 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET,
                            111 + CONTENT_ID_OFFSET,
                        ],
                        after: vec![256 + CONTENT_ID_OFFSET, 111 + CONTENT_ID_OFFSET],
                    },
                    BpeMergeApplication {
                        rank: 1,
                        replacements: 1,
                        before: vec![256 + CONTENT_ID_OFFSET, 111 + CONTENT_ID_OFFSET],
                        after: vec![257 + CONTENT_ID_OFFSET],
                    },
                ];
                let expected_initial_tokens = vec![
                    102 + CONTENT_ID_OFFSET,
                    111 + CONTENT_ID_OFFSET,
                    111 + CONTENT_ID_OFFSET,
                ];
                let expected_content_tokens = vec![257 + CONTENT_ID_OFFSET];

                let result = bpe_tokenizer.encode_content_with_trace([102, 111, 111].as_slice());
                assert_eq!(result.initial_tokens, expected_initial_tokens);
                assert_eq!(result.applications, expected_merge_applications);
                assert_eq!(result.content_tokens, expected_content_tokens);
            }
        }

        mod fn_encode_content {
            use super::*;

            #[test]
            fn encodes_bytes_into_content() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer.encode_content([102, 111, 111].as_slice()),
                    vec![LATEST_TRAINED_TOKEN_ID + CONTENT_ID_OFFSET]
                )
            }
        }

        mod fb_encode_utf8 {
            use super::*;

            #[test]
            fn it_encodes_text_into_content() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer.encode_utf8("foo"),
                    vec![LATEST_TRAINED_TOKEN_ID + CONTENT_ID_OFFSET]
                )
            }
        }

        mod fn_encode_document {
            use super::*;

            #[test]
            fn it_encodes_bytes_sequence_into_a_document() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer.encode_document([102, 111, 111].as_slice()),
                    vec![0, LATEST_TRAINED_TOKEN_ID + CONTENT_ID_OFFSET, 1]
                )
            }
        }

        mod fn_encode_utf8_document {
            use super::*;

            #[test]
            fn it_encodes_text_into_a_document() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer.encode_utf8_document("foo"),
                    vec![0, LATEST_TRAINED_TOKEN_ID + CONTENT_ID_OFFSET, 1]
                )
            }
        }

        mod fn_decode_tokens {
            use super::*;

            mod when_tokens_are_not_recognizable {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                    let content_offset = 1;

                    let result = bpe_tokenizer.decode_tokens([259, 260].as_slice(), content_offset);
                    assert_eq!(
                        result,
                        Err(BpeTokenizerError::UnknownToken {
                            position: content_offset + 1, // 1 is the index of 260 token
                            token_id: 260
                        })
                    );
                }
            }

            mod when_tokens_are_recognizable {
                use super::*;

                #[test]
                fn it_decodes_content_tokens_sequence_into_bytes_sequence() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                    let content_offset = 0;

                    let result = bpe_tokenizer
                        .decode_tokens(bpe_tokenizer.encode_utf8("foo").as_slice(), content_offset);
                    assert!(result.is_ok());
                    assert_eq!(result.as_ref().unwrap(), &vec![102, 111, 111])
                }
            }
        }

        mod fn_decode_document {
            use super::*;

            mod when_document_length_is_lt_two {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_document([1].as_slice()),
                        Err(BpeTokenizerError::DocumentTooShort { length: 1 })
                    )
                }
            }

            mod when_document_starts_from_non_bos_token {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_document([2, 3, 4].as_slice()),
                        Err(BpeTokenizerError::ExpectedBos { found: 2 })
                    )
                }
            }

            mod when_document_ends_on_non_eos_token {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_document([0, 2, 3, 4].as_slice()),
                        Err(BpeTokenizerError::ExpectedEos { found: 4 })
                    )
                }
            }

            mod when_document_is_valid {
                use super::*;

                #[test]
                fn it_decodes_it_into_bytes_sequence() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();
                    let bytes_sequence = [102, 111, 111].as_slice();
                    let encoded = bpe_tokenizer.encode_document(bytes_sequence);

                    let result = bpe_tokenizer.decode_document(&encoded);
                    assert!(result.is_ok());
                    assert_eq!(result.unwrap(), bytes_sequence)
                }
            }
        }

        mod fn_decode_content {
            use super::*;

            mod when_content_contains_bos_token {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_content([2, BOS_TOKEN_ID, 4].as_slice()),
                        Err(BpeTokenizerError::ControlTokenInContent {
                            position: 1,
                            token_id: BOS_TOKEN_ID
                        })
                    )
                }
            }

            mod when_content_contains_eos_token {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_content([2, EOS_TOKEN_ID, 4].as_slice()),
                        Err(BpeTokenizerError::ControlTokenInContent {
                            position: 1,
                            token_id: EOS_TOKEN_ID
                        })
                    )
                }
            }

            mod when_content_is_valid {
                use super::*;

                #[test]
                fn it_decodes_content_tokens_into_bytes_sequence() {
                    let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                    assert_eq!(
                        bpe_tokenizer.decode_content([259].as_slice()),
                        Ok(vec![102, 111, 111])
                    )
                }
            }
        }

        mod fn_decode_content_utf8 {
            use super::*;

            #[test]
            fn it_decodes_content_tokens_into_utf8() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer.decode_content_utf8([259].as_slice()),
                    Ok("foo".to_string())
                )
            }
        }

        mod fn_decode_document_ut8 {
            use super::*;

            #[test]
            fn it_decodes_document_tokens_into_utf8() {
                let bpe_tokenizer = BpeTokenizer::from_training(&bpe_training()).unwrap();

                assert_eq!(
                    bpe_tokenizer
                        .decode_document_utf8([BOS_TOKEN_ID, 259, EOS_TOKEN_ID].as_slice()),
                    Ok("foo".to_string())
                )
            }
        }
    }

    mod fn_strict_utf8 {
        use super::*;

        mod when_correct_utf8_sequence_is_given {
            use super::*;

            #[test]
            fn it_decodes_it_into_test() {
                let sequence: Vec<u8> = vec![108, 111, 108, 32, 208, 186, 208, 181, 208, 186];

                assert_eq!(strict_utf8(sequence), Ok("lol кек".to_string()))
            }
        }

        mod when_incorrect_utf8_sequence_is_given {
            use super::*;

            #[test]
            fn returns_error() {
                let sequence: Vec<u8> = vec![32, 195, 40];

                assert_eq!(
                    strict_utf8(sequence),
                    Err(BpeTokenizerError::InvalidUtf8 {
                        valid_up_to: 1,
                        error_len: Some(1)
                    })
                )
            }
        }
    }
}
