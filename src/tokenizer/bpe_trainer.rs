//! Deterministic byte-pair merge learning over training documents only.
//!
//! Raw bytes begin as token IDs `0..=255`. Every learned merge receives the
//! next ID (`256 + rank`).

use crate::corpus::CorpusPartitions;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::iter::Iterator;

/// Number of one-byte symbols present before any merge is learned.
pub const BYTE_TOKEN_COUNT: u32 = 256;

pub const TOKENS_VOCABULARY_BASE: [[u8; 1]; 256] = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12],
    [13],
    [14],
    [15],
    [16],
    [17],
    [18],
    [19],
    [20],
    [21],
    [22],
    [23],
    [24],
    [25],
    [26],
    [27],
    [28],
    [29],
    [30],
    [31],
    [32],
    [33],
    [34],
    [35],
    [36],
    [37],
    [38],
    [39],
    [40],
    [41],
    [42],
    [43],
    [44],
    [45],
    [46],
    [47],
    [48],
    [49],
    [50],
    [51],
    [52],
    [53],
    [54],
    [55],
    [56],
    [57],
    [58],
    [59],
    [60],
    [61],
    [62],
    [63],
    [64],
    [65],
    [66],
    [67],
    [68],
    [69],
    [70],
    [71],
    [72],
    [73],
    [74],
    [75],
    [76],
    [77],
    [78],
    [79],
    [80],
    [81],
    [82],
    [83],
    [84],
    [85],
    [86],
    [87],
    [88],
    [89],
    [90],
    [91],
    [92],
    [93],
    [94],
    [95],
    [96],
    [97],
    [98],
    [99],
    [100],
    [101],
    [102],
    [103],
    [104],
    [105],
    [106],
    [107],
    [108],
    [109],
    [110],
    [111],
    [112],
    [113],
    [114],
    [115],
    [116],
    [117],
    [118],
    [119],
    [120],
    [121],
    [122],
    [123],
    [124],
    [125],
    [126],
    [127],
    [128],
    [129],
    [130],
    [131],
    [132],
    [133],
    [134],
    [135],
    [136],
    [137],
    [138],
    [139],
    [140],
    [141],
    [142],
    [143],
    [144],
    [145],
    [146],
    [147],
    [148],
    [149],
    [150],
    [151],
    [152],
    [153],
    [154],
    [155],
    [156],
    [157],
    [158],
    [159],
    [160],
    [161],
    [162],
    [163],
    [164],
    [165],
    [166],
    [167],
    [168],
    [169],
    [170],
    [171],
    [172],
    [173],
    [174],
    [175],
    [176],
    [177],
    [178],
    [179],
    [180],
    [181],
    [182],
    [183],
    [184],
    [185],
    [186],
    [187],
    [188],
    [189],
    [190],
    [191],
    [192],
    [193],
    [194],
    [195],
    [196],
    [197],
    [198],
    [199],
    [200],
    [201],
    [202],
    [203],
    [204],
    [205],
    [206],
    [207],
    [208],
    [209],
    [210],
    [211],
    [212],
    [213],
    [214],
    [215],
    [216],
    [217],
    [218],
    [219],
    [220],
    [221],
    [222],
    [223],
    [224],
    [225],
    [226],
    [227],
    [228],
    [229],
    [230],
    [231],
    [232],
    [233],
    [234],
    [235],
    [236],
    [237],
    [238],
    [239],
    [240],
    [241],
    [242],
    [243],
    [244],
    [245],
    [246],
    [247],
    [248],
    [249],
    [250],
    [251],
    [252],
    [253],
    [254],
    [255],
];

/// A numeric adjacent-token candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenPair {
    left: u32,
    right: u32,
}

impl TokenPair {
    /// Creates a candidate from its left and right token IDs.
    pub fn new(left: u32, right: u32) -> Self {
        Self { left, right }
    }

    /// Returns the left token ID.
    pub fn left(&self) -> u32 {
        self.left
    }

    /// Returns the right token ID.
    pub fn right(&self) -> u32 {
        self.right
    }
}

/// One frozen BPE operation in increasing rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeRule {
    rank: usize,
    pair: TokenPair,
    token_id: u32,
    candidate_count: usize,
    replacement_count: usize,
}

impl MergeRule {
    /// Returns the zero-based order in which the rule was learned.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the pair selected during this round.
    pub fn pair(&self) -> TokenPair {
        self.pair
    }

    /// Returns the new training-space token ID assigned to the merged bytes.
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    /// Returns the overlapping pair count used to rank the candidate.
    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Returns the non-overlapping occurrences replaced in this round.
    pub fn replacement_count(&self) -> usize {
        self.replacement_count
    }
}

/// The learned ranks, vocabulary expansions, and final training sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTraining {
    requested_merges: usize,
    training_document_ids: Vec<String>,
    rules: Vec<MergeRule>,
    vocabulary: Vec<Vec<u8>>,
    final_sequences: Vec<Vec<u32>>,
}

impl BpeTraining {
    /// Returns the configured maximum number of rounds.
    pub fn requested_merge(&self) -> usize {
        self.requested_merges
    }

    /// Returns the exact source IDs that supplied pair statistics.
    pub fn training_document_ids(&self) -> &[String] {
        &self.training_document_ids
    }

    /// Returns rules in frozen application order.
    pub fn rules(&self) -> &[MergeRule] {
        &self.rules
    }

    /// Returns each training document's final token sequence in source order.
    pub fn final_sequences(&self) -> &[Vec<u32>] {
        &self.final_sequences
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Expands one training-space token ID to the bytes it represents.
    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        usize::try_from(token_id)
            .ok()
            .and_then(|index| self.vocabulary.get(index))
            .map(Vec::as_slice)
    }

    /// Contains 0..255 values, 1 value per vec + merged bytes from trained sequences. So, in head
    /// we always have `[0], [1], ..., [255]`. Then each learned merge adds another entry. Let's say
    /// we have "aaa" sequence:
    /// initial sequence:   `[97, 97, 97]`
    /// rule:               (97, 97) → 256
    /// `vocabulary[256]`:  `[0x61, 0x61]`       // "aa"
    /// result sequence:    `[256, 97]`
    ///
    /// If a later rule merged (256, 97):
    ///
    /// rule:               (256, 97) → 257
    /// `vocabulary[257]`:  `[0x61, 0x61, 0x61]` // "aaa"
    ///
    /// So the invariant is:
    ///
    /// IDs 0..255  represent individual bytes
    /// ID 256 + r  represents the concatenated bytes of the pair learned at rank r
    /// The table serves three purposes:
    ///
    ///   - It gives every newly created numeric ID an exact meaning.
    ///   - It lets the trace and later tokenizer retrieve a token’s bytes directly with
    ///     `vocabulary[id]`.
    ///   - It freezes a lossless handoff to BpeTokenizer, where the learned rules are applied to
    ///     arbitrary input and tokens must eventually decode back to bytes.
    ///
    /// The trainer does not need this table merely to count pairs or replace IDs. The ordered merge
    /// rules contain enough information to reconstruct every expansion recursively. Therefore,
    /// vocabulary is derived information stored eagerly for direct lookup and a simpler
    /// BpeTokenizer handoff.
    pub fn vocabulary(&self) -> &[Vec<u8>] {
        &self.vocabulary
    }
}

/// A deterministic tokenizer trainer with a fixed upper bound on merge rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BpeTrainer {
    max_merges: usize,
}

impl BpeTrainer {
    /// Creates a trainer. Zero is a valid request and learns no rules.
    pub fn new(max_merges: usize) -> Self {
        Self { max_merges }
    }

    /// Returns trained tokenizer state plus training evidence. In BPE, "training" means learning
    /// an ordered merge table from corpus statistics. Learns only from the partition object's
    /// training view.
    pub fn train(
        &self,
        partitions: &CorpusPartitions<'_>,
    ) -> Result<BpeTraining, BpeTrainingError> {
        let available_merge_ids = u32::MAX - u8::MAX as u32;
        if self.max_merges > available_merge_ids as usize {
            return Err(BpeTrainingError::new(
                "requested merge count exceeds the u32 token-ID space",
            ));
        }
        let training_documents = partitions.training_documents();
        let document_ids = training_documents
            .iter()
            .map(|document| document.id().to_owned())
            .collect::<Vec<_>>();
        let sequences = training_documents
            .iter()
            .map(|document| text_to_tokens(document.text()))
            .collect::<Vec<_>>();

        learn_from_token_sequences(self.max_merges, document_ids, sequences)
    }
}

/// Converts text to the initial numeric vocabulary.
fn text_to_tokens(text: &str) -> Vec<u32> {
    text.as_bytes()
        .iter()
        .map(|byte| u32::from(*byte))
        .collect()
}

/// Counts adjacent pairs independently inside every supplied token sequence.
fn count_adjacent_pairs(sequences: &[Vec<u32>]) -> BTreeMap<TokenPair, usize> {
    let mut counts = BTreeMap::new();
    for sequence in sequences {
        for window in sequence.windows(2) {
            let pair = TokenPair::new(window[0], window[1]);
            *counts.entry(pair).or_insert(0) += 1;
        }
    }
    counts
}

/// Selects the greatest count, breaking ties by the smallest `(left, right)` IDs.
fn choose_most_frequent_pair(counts: &BTreeMap<TokenPair, usize>) -> Option<(TokenPair, usize)> {
    let mut winner = None;
    for (&pair, &count) in counts {
        match winner {
            None => winner = Some((pair, count)),
            Some((_, best_count)) if count > best_count => winner = Some((pair, count)),
            Some(_) => {}
        }
    }
    winner
}

/// Replaces matches from left to right, so one input token is consumed at most once.
pub fn replace_pair_left_to_right(
    sequence: &[u32],
    pair: TokenPair,
    replacement: u32,
) -> (Vec<u32>, usize) {
    let mut output = Vec::with_capacity(sequence.len());
    let mut replacements = 0;
    let mut index = 0;

    while index < sequence.len() {
        if index + 1 < sequence.len()
            && sequence[index] == pair.left
            && sequence[index + 1] == pair.right
        {
            output.push(replacement);
            replacements += 1;
            index += 2;
        } else {
            output.push(sequence[index]);
            index += 1;
        }
    }

    (output, replacements)
}

pub fn training_vocabulary_base() -> Vec<Vec<u8>> {
    TOKENS_VOCABULARY_BASE
        .iter()
        .map(|bytes| bytes.to_vec())
        .collect()
}

/// Repeatedly count adjacent pairs, select a winner, assign a new token ID, and replace that pair
fn learn_from_token_sequences(
    max_merges: usize,
    document_ids: Vec<String>,
    mut sequences: Vec<Vec<u32>>,
) -> Result<BpeTraining, BpeTrainingError> {
    let mut vocabulary = training_vocabulary_base();
    let mut rules: Vec<MergeRule> = Vec::new();

    for rank in 0..max_merges {
        let counts = count_adjacent_pairs(&sequences);
        let Some((pair, candidate_count)) = choose_most_frequent_pair(&counts) else {
            break;
        };

        let token_id = u32::try_from(vocabulary.len()).map_err(|_| {
            BpeTrainingError::new("learned vocabulary exceeds the u32 token-ID space")
        })?;
        let left_bytes = vocabulary
            .get(pair.left as usize)
            .ok_or_else(|| BpeTrainingError::new("pair contains an unknown left token ID"))?;
        let right_bytes = vocabulary
            .get(pair.right as usize)
            .ok_or_else(|| BpeTrainingError::new("pair contains an unknown right token ID"))?;
        let mut merged_bytes = Vec::with_capacity(left_bytes.len() + right_bytes.len());
        merged_bytes.extend_from_slice(left_bytes);
        merged_bytes.extend_from_slice(right_bytes);

        let mut replacement_count = 0;
        for sequence in &mut sequences {
            let (replaced, count) = replace_pair_left_to_right(sequence, pair, token_id);
            *sequence = replaced;
            replacement_count += count;
        }
        if replacement_count == 0 {
            return Err(BpeTrainingError::new(
                "selected pair had no replaceable occurrence",
            ));
        }

        vocabulary.push(merged_bytes);
        rules.push(MergeRule {
            rank,
            pair,
            token_id,
            candidate_count,
            replacement_count,
        });
    }

    Ok(BpeTraining {
        requested_merges: max_merges,
        training_document_ids: document_ids,
        rules,
        vocabulary,
        final_sequences: sequences,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTrainingError {
    message: String,
}

impl BpeTrainingError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for BpeTrainingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for BpeTrainingError {}

#[cfg(test)]
mod tests {
    use super::*;

    mod bpe_training {
        use super::*;

        mod fn_token_bytes {
            use super::*;

            #[test]
            fn it_returns_bytes_from_vocabulary_the_token_id_corresponds_to() {
                let bpe_training = BpeTraining {
                    requested_merges: 1,
                    training_document_ids: vec![],
                    rules: vec![],
                    vocabulary: vec![vec![1], vec![1, 2]],
                    final_sequences: vec![],
                };
                assert_eq!(bpe_training.token_bytes(1), Some(vec![1, 2].as_slice()));
            }
        }
    }

    mod bpe_trainer {
        use super::*;

        mod fn_train {
            use super::*;

            mod when_number_of_merges_exceeds_the_number_of_available_merges {
                use super::*;
                use crate::corpus::{Corpus, SplitManifest};
                use crate::support::{CORPUS_FILE, CORPUS_MANIFEST};

                #[test]
                fn it_returns_error() {
                    let bpe_trainer = BpeTrainer::new(u32::MAX as usize);
                    let corpus = Corpus::from_file(CORPUS_FILE).unwrap();
                    let manifest = SplitManifest::from_file(CORPUS_MANIFEST).unwrap();
                    let partitions = manifest.partition(&corpus).unwrap();

                    assert_eq!(
                        bpe_trainer.train(&partitions),
                        Err(BpeTrainingError::new(
                            "requested merge count exceeds the u32 token-ID space"
                        ))
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;
                use crate::corpus::{Corpus, SplitManifest};
                use crate::support::{SIMPLE_CORPUS_FILE, SIMPLE_CORPUS_MANIFEST};

                #[test]
                fn returns_trained_tokenizer_state() {
                    let bpe_trainer = BpeTrainer::new(2);
                    let corpus = Corpus::from_file(SIMPLE_CORPUS_FILE).unwrap();
                    let manifest = SplitManifest::from_file(SIMPLE_CORPUS_MANIFEST).unwrap();
                    let partitions = manifest.partition(&corpus).unwrap();

                    let result = bpe_trainer.train(&partitions);

                    let expected_rules = vec![
                        MergeRule {
                            rank: 0,
                            pair: TokenPair {
                                left: 102,
                                right: 111,
                            },
                            token_id: 256,
                            candidate_count: 1,
                            replacement_count: 1,
                        },
                        MergeRule {
                            rank: 1,
                            pair: TokenPair {
                                left: 256,
                                right: 111,
                            },
                            token_id: 257,
                            candidate_count: 1,
                            replacement_count: 1,
                        },
                    ];

                    assert!(result.is_ok());
                    println!("{:?}", result);
                    assert_eq!(
                        &result.as_ref().unwrap().training_document_ids,
                        &vec![String::from("ru-river-dawn")]
                    );
                    assert_eq!(&result.as_ref().unwrap().final_sequences, &vec![vec![257]]);
                    assert_eq!(result.as_ref().unwrap().requested_merges, 2);
                    assert_eq!(&result.as_ref().unwrap().rules, &expected_rules);
                    assert_eq!(
                        &result.as_ref().unwrap().vocabulary[256..],
                        &vec![vec![102, 111], vec![102, 111, 111]]
                    );
                }
            }
        }
    }

    mod fn_text_to_tokens {
        use super::*;

        #[test]
        fn it_converts_text_into_u32_sequence() {
            assert_eq!(
                text_to_tokens("lol kek"),
                Vec::<u32>::from([108, 111, 108, 32, 107, 101, 107])
            )
        }
    }

    mod fn_count_adjacent_pairs {
        use super::*;

        #[test]
        fn it_counts_neighbour_pairs() {
            // [32, 98, 97, 114, 32, 98, 97, 122]
            let sequence1 = text_to_tokens(" bar baz");
            // [98, 97, 102]
            let sequence2 = text_to_tokens("baf");

            assert_eq!(
                count_adjacent_pairs([sequence1, sequence2].as_slice()),
                BTreeMap::from([
                    (TokenPair::new(98, 97), 3),
                    (TokenPair::new(32, 98), 2),
                    (TokenPair::new(97, 114), 1),
                    (TokenPair::new(114, 32), 1),
                    (TokenPair::new(97, 122), 1),
                    (TokenPair::new(97, 102), 1),
                ])
            )
        }
    }

    mod fn_choose_most_frequent_pair {
        use super::*;

        mod when_there_is_pair_with_greatest_count {
            use super::*;

            #[test]
            fn it_choices_most_frequent_pair() {
                let counts = BTreeMap::from([
                    (TokenPair::new(1, 2), 3),
                    (TokenPair::new(2, 3), 2),
                    (TokenPair::new(3, 4), 1),
                ]);

                assert_eq!(
                    choose_most_frequent_pair(&counts),
                    Some((TokenPair::new(1, 2), 3))
                )
            }
        }

        mod when_all_pairs_has_the_same_count {
            use super::*;

            mod when_all_left_values_are_equal {
                use super::*;

                #[test]
                fn it_picks_the_pair_with_lowest_right_value() {
                    let counts = BTreeMap::from([
                        (TokenPair::new(1, 2), 1),
                        (TokenPair::new(1, 3), 1),
                        (TokenPair::new(1, 4), 1),
                    ]);

                    assert_eq!(
                        choose_most_frequent_pair(&counts),
                        Some((TokenPair::new(1, 2), 1))
                    )
                }
            }

            mod when_left_values_are_not_equal {
                use super::*;

                #[test]
                fn it_picks_the_pair_with_lowest_left_value() {
                    let counts = BTreeMap::from([
                        (TokenPair::new(1, 0), 1),
                        (TokenPair::new(2, 0), 1),
                        (TokenPair::new(3, 0), 1),
                    ]);

                    assert_eq!(
                        choose_most_frequent_pair(&counts),
                        Some((TokenPair::new(1, 0), 1))
                    )
                }
            }
        }
    }

    mod fn_replace_pair_left_to_right {
        use super::*;

        #[test]
        fn replaces_matching_pairs_in_sequence_with_the_replacement() {
            let sequence: [u32; 7] = [2, 0, 1, 10, 0, 2, 0];
            let pair = TokenPair::new(2, 0);
            let replacement: u32 = 256;

            assert_eq!(
                replace_pair_left_to_right(&sequence.as_slice(), pair, replacement),
                (vec![replacement, 1, 10, 0, replacement], 2)
            )
        }
    }

    mod fn_learn_from_token_sequences {
        use super::*;

        #[test]
        fn returns_trained_tokenizer_state() {
            let max_merges = 2;
            let document_ids = vec!["foo".to_owned(), "bar".to_owned()];
            // 1st merge
            //      Pairs:
            //          (2, 0)  => 4 // winner
            //          (0, 1)  => 3
            //          (1, 10) => 3
            //          (0, 2) => 2
            //          (10, 3) => 1
            //      Resulting sequences:
            //          [
            //              [256, 1, 10, 0, 256],
            //              [256, 1, 10, 0, 256, 1, 10, 3],
            //          ]
            // 2nd merge
            //      Pairs:
            //          (256, 1) => 3
            //          (1, 10) => 3 // winner
            //          (10, 0) => 2
            //          (0, 256) => 2
            //          (10, 3) => 1
            //      Resulting sequence:
            //          [
            //              [256, 257, 0, 256],
            //              [256, 257, 0, 256, 257, 3],
            //          ]
            let sequences: Vec<Vec<u32>> = vec![
                vec![2, 0, 1, 10, 0, 2, 0],
                vec![2, 0, 1, 10, 0, 2, 0, 1, 10, 3],
            ];

            let expected_rules = vec![
                MergeRule {
                    rank: 0,
                    pair: TokenPair::new(2, 0),
                    token_id: 256,
                    candidate_count: 4,
                    replacement_count: 4,
                },
                MergeRule {
                    rank: 1,
                    pair: TokenPair::new(1, 10),
                    token_id: 257,
                    candidate_count: 3,
                    replacement_count: 3,
                },
            ];
            let mut expected_vocabulary = training_vocabulary_base();
            expected_vocabulary.push(vec![2, 0]);
            expected_vocabulary.push(vec![1, 10]);
            let final_sequences = vec![vec![256, 257, 0, 256], vec![256, 257, 0, 256, 257, 3]];
            let expected_bpe_training = BpeTraining {
                requested_merges: max_merges,
                training_document_ids: document_ids.clone(),
                rules: expected_rules.clone(),
                vocabulary: expected_vocabulary.clone(),
                final_sequences: final_sequences.clone(),
            };

            let result = learn_from_token_sequences(max_merges, document_ids.clone(), sequences);

            assert_eq!(result.as_ref().unwrap().requested_merges, max_merges);
            assert_eq!(
                &result.as_ref().unwrap().training_document_ids,
                &document_ids
            );
            assert_eq!(&result.as_ref().unwrap().rules, &expected_rules);
            assert_eq!(&result.as_ref().unwrap().vocabulary, &expected_vocabulary);
            assert_eq!(&result.as_ref().unwrap().final_sequences, &final_sequences);
            assert_eq!(result, Ok(expected_bpe_training))
        }
    }
}
