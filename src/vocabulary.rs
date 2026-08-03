use std::fmt;
use std::fmt::Formatter;

/// The ID reserved for any scalar value absent from a vocabulary.
pub const UNKNOWN_TOKEN_ID: usize = 0;

/// The text representation of [`UNKNOWN_TOKEN_ID`]
pub const UNKNOWN_TOKEN: &str = "<UNK>";

/// Reports a token ID that has no entry in a particular vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidTokenId {
    id: usize,
    max_id: usize,
}

impl InvalidTokenId {
    /// Returns the invalid ID supplied by the caller.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the largest ID accepted by the vocabulary.
    pub fn max_id(&self) -> usize {
        self.max_id
    }
}

impl fmt::Display for InvalidTokenId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "token ID {} is outside vocabulary range 0..={}",
            self.id, self.max_id
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyUnit {
    Scalar(char),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    known_units: Vec<char>,
}

/// Decodes text into Unicode scalar values in source order.
pub fn unicode_scalars(text: &str) -> Vec<char> {
    text.chars().collect()
}

impl Vocabulary {
    /// Builds a vocabulary from the unique scalar values in `training_text`
    pub fn from_training_text(training_text: &str) -> Self {
        let mut known_units = unicode_scalars(training_text);
        known_units.sort_unstable();
        known_units.dedup();
        Self { known_units }
    }

    /// Returns known units in their deterministic ID order.
    pub fn known_units(&self) -> &[char] {
        &self.known_units
    }

    /// Iterates over `(ID, scalar)` pairs, excluding the reserved unknown ID.
    pub fn entries(&self) -> impl Iterator<Item = (usize, char)> + '_ {
        self.known_units
            .iter()
            .copied()
            .enumerate()
            .map(|(index, unit)| (index + 1, unit))
    }

    /// Find ID by the given scalar unit, returning [`UNKNOWN_TOKEN_ID`] when absent.
    pub fn id_for(&self, unit: char) -> usize {
        self.known_units
            .binary_search(&unit)
            .map_or(UNKNOWN_TOKEN_ID, |index| index + 1)
    }

    /// Find scalar unit by the given ID.
    ///
    /// `Ok(None)` represents the reserved unknown token. An ID above the vocabulary's largest known
    /// ID is an error rather than an unknown token.
    pub fn unit_for_id(&self, id: usize) -> Result<VocabularyUnit, InvalidTokenId> {
        if id == UNKNOWN_TOKEN_ID {
            return Ok(VocabularyUnit::Unknown);
        }

        self.known_units
            .get(id - 1)
            .copied()
            .map(VocabularyUnit::Scalar)
            .ok_or(InvalidTokenId {
                id,
                max_id: self.known_units.len(),
            })
    }

    /// Encodes each scalar unit of `text` into ID.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|unit| self.id_for(unit)).collect()
    }

    /// Decodes IDs, rendering ID [`UNKNOWN_TOKEN_ID`] as the literal [`UNKNOWN_TOKEN`].
    pub fn decode(&self, ids: &[usize]) -> Result<String, InvalidTokenId> {
        let mut text = String::with_capacity(ids.len());

        for &id in ids {
            match self.unit_for_id(id)? {
                VocabularyUnit::Scalar(unit) => text.push(unit),
                VocabularyUnit::Unknown => text.push_str(UNKNOWN_TOKEN),
            }
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRAINING_TEXT: &str = "hi-hi, лол";

    mod from_training_text {
        use super::*;

        #[test]
        fn it_creates_vocabulary() {
            let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

            assert_eq!(
                vocabulary.known_units(),
                vec![' ', ',', '-', 'h', 'i', 'л', 'о']
            )
        }
    }

    mod entries {
        use super::*;

        #[test]
        fn it_computes_id_to_unit_collection() {
            let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

            assert_eq!(
                vocabulary.entries().collect::<Vec<_>>(),
                vec![
                    (1, ' '),
                    (2, ','),
                    (3, '-'),
                    (4, 'h'),
                    (5, 'i'),
                    (6, 'л'),
                    (7, 'о')
                ]
            )
        }
    }

    mod id_for {
        use super::*;

        mod when_unit_exists_in_vocabulary {
            use super::*;

            #[test]
            fn it_finds_id_by_the_given_unit() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

                assert_eq!(vocabulary.id_for('i'), 5)
            }
        }

        mod when_unit_does_not_exist_in_vocabulary {
            use super::*;

            #[test]
            fn it_returns_unknown_token_id() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

                assert_eq!(vocabulary.id_for('z'), UNKNOWN_TOKEN_ID)
            }
        }
    }

    mod unit_for_id {
        use super::*;

        mod when_known_id_is_given {
            use super::*;

            #[test]
            fn it_returns_scalar_vocabulary_unit() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

                assert_eq!(vocabulary.unit_for_id(2), Ok(VocabularyUnit::Scalar(',')))
            }
        }

        mod when_given_id_is_unknown_token_id {
            use super::*;

            #[test]
            fn it_returns_unknown_vocabulary_unit() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

                assert_eq!(
                    vocabulary.unit_for_id(UNKNOWN_TOKEN_ID),
                    Ok(VocabularyUnit::Unknown)
                )
            }
        }

        mod when_id_is_not_existing_id {
            use super::*;

            #[test]
            fn it_returns_error() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

                assert_eq!(
                    vocabulary.unit_for_id(123),
                    Err(InvalidTokenId { id: 123, max_id: 7 })
                )
            }
        }
    }

    mod encode {
        use super::*;

        #[test]
        fn it_encodes_given_text_into_ids() {
            let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);

            assert_eq!(
                vocabulary.encode("лол, hi - лом"),
                vec![6, 7, 6, 2, 1, 4, 5, 1, 3, 1, 6, 7, UNKNOWN_TOKEN_ID]
            )
        }
    }

    mod decode {
        use super::*;

        mod when_the_given_ids_are_all_known_values {
            use super::*;

            #[test]
            fn it_decodes_ids_sequence_into_test() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);
                let ids = vocabulary.encode("лол, hi - лом");

                assert_eq!(vocabulary.decode(&ids), Ok(format!("лол, hi - ло{}", UNKNOWN_TOKEN)))
            }
        }

        mod when_the_given_ids_contains_unknown_values {
            use super::*;

            #[test]
            fn it_returns_error() {
                let vocabulary = Vocabulary::from_training_text(TRAINING_TEXT);
                let ids = [1, 123, 2];

                assert_eq!(vocabulary.decode(&ids), Err(InvalidTokenId { id: 123, max_id: 7 }))
            }
        }
    }
}
