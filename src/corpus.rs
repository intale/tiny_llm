//! Whole-document corpus loading and frozen train/validation/test partitions.
//!
//! This module deliberately runs before tokenization. Stable document IDs,
//! provenance groups, and raw UTF-8 text are validated first so no learned
//! tokenizer or model statistic can move information across a holdout boundary.
//!
//! Validations, implemented here are suitable for limited purpose of this tiny llm and can't be
//! used in complex production solutions, as it is much harder to prepare and validate the quality
//! content. Simple assertions performed here are, obviously, not enough to cover needs of more
//! complex objectives. The implementation does not scale well with the large amount of documents.

use serde::Deserialize;
use std::error::Error;
use std::fmt;

/// Version accepted by [`SplitManifest::from_json`].
pub const SPLIT_SCHEMA_VERSION: u32 = 1;

/// The fixed strategy name recorded in the checked-in split manifest.
pub const SPLIT_STRATEGY: &str = "fixed-paired-document-holdout-v1";

/// One source document whose boundary must survive later tokenization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document {
    id: String,
    language: String,
    provenance_group: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentJson {
    id: String,
    language: String,
    provenance_group: String,
    text: String,
}

impl DocumentJson {
    fn labeled_attrs(&self) -> [(&str, &str); 3] {
        [
            (&self.id, "id"),
            (&self.language, "language"),
            (&self.provenance_group, "provenance_group"),
        ]
    }

    fn into_document(self) -> Document {
        Document {
            id: self.id,
            language: self.language,
            provenance_group: self.provenance_group,
            text: self.text,
        }
    }
}

impl Document {
    /// Returns the stable manifest identity.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the document's BCP-47-style language code.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the group that must remain in one partition.
    pub fn provenance_group(&self) -> &str {
        &self.provenance_group
    }

    /// Returns the decoded source text.
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// A corpus in stable source order plus a checksum of its original bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Corpus {
    documents: Vec<Document>,
    checksum: String,
}

impl Corpus {
    /// Deserializes corpus documents from the given file.
    pub fn from_file(file_path: &str) -> Result<Self, CorpusError> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|error| CorpusError::new(format!("failed to read corpus: {error}")))?;
        Self::from_json(&content)
    }

    /// Deserializes JSON corpus documents
    pub fn from_json(source: &str) -> Result<Self, CorpusError> {
        let decoded: Vec<DocumentJson> = serde_json::from_str(source)
            .map_err(|error| CorpusError::new(format!("invalid corpus JSON: {error}")))?;
        if decoded.is_empty() {
            return Err(CorpusError::new("corpus contains no documents"));
        }

        let mut documents: Vec<Document> = Vec::with_capacity(decoded.len());

        for (index, document_json) in decoded.into_iter().enumerate() {
            for (value, label) in document_json.labeled_attrs() {
                if !is_kebab_identifier(value) {
                    return Err(CorpusError::new(format!(
                        "corpus document at position {}, {:?} must be lowercase ASCII kebab case",
                        index, label
                    )));
                }
            }
            if document_json.text.trim().is_empty() {
                return Err(CorpusError::new(format!(
                    "corpus document at position {index} text is empty"
                )));
            }
            if documents
                .iter()
                .any(|existing| existing.text == document_json.text)
            {
                return Err(CorpusError::new(format!(
                    "duplicate document text at position {index} would leak identical content"
                )));
            }
            documents.push(document_json.into_document());
        }

        Ok(Self {
            documents,
            checksum: format!("fnv1a64:{:016x}", fnv1a64(source.as_bytes())),
        })
    }

    /// Returns documents in their canonical source order.
    pub fn documents(&self) -> &[Document] {
        &self.documents
    }

    /// Find document by id
    pub fn document(&self, id: &str) -> Option<&Document> {
        self.documents.iter().find(|document| document.id == id)
    }

    /// Returns the deterministic checksum of the original corpus bytes.
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
}

/// One of the three mutually exclusive dataset roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    Train,
    Validation,
    Test,
}

impl Partition {
    /// Returns string representation of the partition.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

/// A parsed, but not yet corpus-validated, frozen split manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitManifest {
    schema_version: u32,
    corpus_checksum: String,
    strategy: String,
    train: Vec<String>,
    validation: Vec<String>,
    test: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitManifestJson {
    schema_version: u32,
    corpus_checksum: String,
    strategy: String,
    train: Vec<String>,
    validation: Vec<String>,
    test: Vec<String>,
}

impl SplitManifest {
    /// Deserializes the manifest from the given file.
    pub fn from_file(file_path: &str) -> Result<Self, CorpusError> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|error| CorpusError::new(format!("failed to read split manifest: {error}")))?;
        Self::from_json(&content)
    }

    /// Deserializes the JSON manifest.
    pub fn from_json(source: &str) -> Result<Self, CorpusError> {
        let manifest: SplitManifestJson = serde_json::from_str(source)
            .map_err(|error| CorpusError::new(format!("invalid split manifest JSON: {error}")))?;
        Ok(Self {
            schema_version: manifest.schema_version,
            corpus_checksum: manifest.corpus_checksum,
            strategy: manifest.strategy,
            train: manifest.train,
            validation: manifest.validation,
            test: manifest.test,
        })
    }

    /// Returns the manifest's recorded corpus checksum.
    pub fn corpus_checksum(&self) -> &str {
        &self.corpus_checksum
    }

    /// Returns document IDs assigned to one partition in manifest order.
    pub fn ids(&self, partition: Partition) -> &[String] {
        match partition {
            Partition::Train => &self.train,
            Partition::Validation => &self.validation,
            Partition::Test => &self.test,
        }
    }

    /// Validates checksum, coverage, disjointness, order and provenance groups.
    pub fn partition<'a>(&self, corpus: &'a Corpus) -> Result<CorpusPartitions<'a>, CorpusError> {
        if self.schema_version != SPLIT_SCHEMA_VERSION {
            return Err(CorpusError::new(format!(
                "unsupported split schema version {}",
                self.schema_version
            )));
        }
        if self.strategy != SPLIT_STRATEGY {
            return Err(CorpusError::new(format!(
                "unsupported split strategy {:?}",
                self.strategy
            )));
        }
        if self.corpus_checksum != corpus.checksum {
            return Err(CorpusError::new(format!(
                "corpus checksum mismatch: manifest={:?}, actual={:?}",
                self.corpus_checksum, corpus.checksum
            )));
        }

        for partition in [Partition::Train, Partition::Validation, Partition::Test] {
            let ids = self.ids(partition);
            if ids.is_empty() {
                return Err(CorpusError::new(format!(
                    "{:?} partition is empty",
                    partition.label()
                )));
            }
            validate_source_order(corpus, partition, ids)?;
        }

        let mut seen = Vec::new();
        for partition in [Partition::Train, Partition::Validation, Partition::Test] {
            for id in self.ids(partition) {
                if corpus.document(id).is_none() {
                    return Err(CorpusError::new(format!(
                        "{:?} partition contains unknown document {id:?}",
                        partition.label()
                    )));
                }
                if seen.contains(&id) {
                    return Err(CorpusError::new(format!(
                        "document {id:?} appears in more than one manifest position"
                    )));
                }
                seen.push(id);
            }
        }
        if seen.len() != corpus.documents.len() {
            let missing = corpus
                .documents
                .iter()
                .find(|document| !seen.iter().any(|id| id.as_str() == document.id))
                .map_or("<unknown>", Document::id);
            return Err(CorpusError::new(format!(
                "manifest does not cover corpus document {missing:?}"
            )));
        }

        // Ensure the provenance group appears exactly in one partition
        for document in &corpus.documents {
            let assigned = self.assignment(document.id());
            if let Some(related) = corpus.documents.iter().find(|candidate| {
                candidate.provenance_group == document.provenance_group
                    && self.assignment(candidate.id()) != assigned
            }) {
                return Err(CorpusError::new(format!(
                    "provenance group {:?} is split between {:?} and {:?} partitions",
                    document.provenance_group,
                    assigned.label(),
                    self.assignment(related.id()).label()
                )));
            }
        }

        let mut partitions = CorpusPartitions {
            train: Vec::new(),
            validation: Vec::new(),
            test: Vec::new(),
        };
        for document in &corpus.documents {
            match self.assignment(document.id()) {
                Partition::Train => partitions.train.push(document),
                Partition::Validation => partitions.validation.push(document),
                Partition::Test => partitions.test.push(document),
            }
        }
        Ok(partitions)
    }

    fn assignment(&self, id: &str) -> Partition {
        [Partition::Train, Partition::Validation, Partition::Test]
            .into_iter()
            .find(|&partition| self.ids(partition).iter().any(|candidate| candidate == id))
            .expect(&format!("unknown document id: {:?}", id))
    }
}

/// Borrowed document slices produced only after manifest validation succeeds.
#[derive(Debug, PartialEq, Eq)]
pub struct CorpusPartitions<'a> {
    train: Vec<&'a Document>,
    validation: Vec<&'a Document>,
    test: Vec<&'a Document>,
}

impl<'a> CorpusPartitions<'a> {
    /// Returns documents for one role in original corpus order.
    pub fn documents(&self, partition: Partition) -> &[&'a Document] {
        match partition {
            Partition::Train => &self.train,
            Partition::Validation => &self.validation,
            Partition::Test => &self.test,
        }
    }

    /// Returns stable IDs for display or downstream audit metadata.
    pub fn document_ids(&self, partition: Partition) -> Vec<&'a str> {
        self.documents(partition)
            .iter()
            .map(|document| document.id())
            .collect()
    }

    /// Returns the only documents that may be used to learn tokenizer statistics.
    pub fn training_documents(&self) -> &[&'a Document] {
        &self.train
    }
}

/// One deterministic data-contract violation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusError {
    message: String,
}

impl CorpusError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn at(line: usize, message: impl fmt::Display) -> Self {
        Self::new(format!("line {}: {}", line, message))
    }

    /// Returns the stable diagnostic text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for CorpusError {}

fn is_kebab_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.split('-').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
        && value.as_bytes()[0].is_ascii_lowercase()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn validate_source_order(
    corpus: &Corpus,
    partition: Partition,
    ids: &[String],
) -> Result<(), CorpusError> {
    let mut previous = None;
    for id in ids {
        let position = corpus
            .documents
            .iter()
            .position(|document| document.id == *id);
        let Some(position) = position else {
            continue;
        };
        if previous.is_some_and(|prev_pos| position <= prev_pos) {
            return Err(CorpusError::new(format!(
                "{:?} partition IDs do not preserve corpus source order",
                partition.label()
            )));
        }
        previous = Some(position);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::*;

    mod corpus {
        use super::*;

        mod fn_from_file {
            use super::*;

            mod when_file_exists {
                use super::*;

                #[test]
                fn it_deserializes_it() {
                    assert!(Corpus::from_file(CORPUS_FILE).is_ok())
                }
            }

            mod when_file_does_not_exist {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_file("/123.txt");
                    assert_corpus_error(result, "failed to read corpus:")
                }
            }
        }

        mod fn_from_json {
            use super::*;

            mod when_invalid_json_string_is_provided {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json("asd lol");
                    assert_corpus_error(result, "invalid corpus JSON: expected value")
                }
            }

            mod when_invalid_corpus_format_is_provided {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json("[{\"t\": 1}]");
                    assert_corpus_error(result, "invalid corpus JSON: unknown field")
                }
            }

            mod when_documents_are_empty {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json("[]");
                    assert_corpus_error(result, "corpus contains no documents")
                }
            }

            mod when_document_id_of_a_document_has_arbitrary_format {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en river_dawn",
                        "language": "en",
                        "provenance_group": "pair-river-dawn",
                        "text": "something"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json(CONTENT);
                    assert_corpus_error(
                        result,
                        "corpus document at position 0, \"id\" must be lowercase ASCII kebab case",
                    )
                }
            }

            mod when_language_of_a_document_has_arbitrary_format {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "En",
                        "provenance_group": "pair-river-dawn",
                        "text": "something"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json(CONTENT);
                    assert_corpus_error(
                        result,
                        "corpus document at position 0, \"language\" must be lowercase ASCII kebab case",
                    )
                }
            }

            mod when_provenance_group_of_a_document_has_arbitrary_format {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "pair_river_dawn",
                        "text": "something"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json(CONTENT);
                    assert_corpus_error(
                        result,
                        "corpus document at position 0, \"provenance_group\" must be lowercase ASCII kebab case",
                    )
                }
            }

            mod when_document_text_is_empty {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "pair-river-dawn",
                        "text": ""
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json(CONTENT);
                    assert_corpus_error(result, "corpus document at position 0 text is empty")
                }
            }

            mod when_same_text_appears_in_different_documents {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "pair-river-dawn",
                        "text": "something"
                      },
                      {
                        "id": "ru-river-dawn",
                        "language": "ru",
                        "provenance_group": "pair-river-dawn",
                        "text": "something"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let result = Corpus::from_json(CONTENT);
                    assert_corpus_error(
                        result,
                        "duplicate document text at position 1 would leak identical content",
                    )
                }
            }

            mod when_corpus_documents_are_valid {
                use super::*;

                const CONTENT: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "pair-river-dawn",
                        "text": "something"
                      }
                    ]
                "#;

                #[test]
                fn it_computes_corpus() {
                    let result = Corpus::from_json(CONTENT);
                    let corpus = Corpus {
                        documents: vec![Document {
                            id: "en-river-dawn".to_string(),
                            language: "en".to_string(),
                            provenance_group: "pair-river-dawn".to_string(),
                            text: "something".to_string(),
                        }],
                        checksum: "fnv1a64:28478ced3632aef8".to_string(),
                    };
                    assert_eq!(result.unwrap(), corpus)
                }
            }
        }

        mod fn_document {
            use super::*;

            #[test]
            fn it_returns_document_by_id() {
                let corpus = Corpus::from_file(CORPUS_FILE).unwrap();
                let document = corpus
                    .document("en-bee-garden")
                    .expect("could not find document by the given id");
                assert_eq!(document.id, "en-bee-garden");
                assert_eq!(document.language, "en");
                assert_eq!(document.provenance_group, "pair-bee-garden");
                assert!(document.text.contains("Bees visit the yellow cups first"));
            }
        }
    }

    mod split_manifest {
        use super::*;

        mod fn_from_file {
            use super::*;

            mod when_file_exists {
                use super::*;

                #[test]
                fn it_deserializes_it() {
                    assert!(SplitManifest::from_file(CORPUS_MANIFEST).is_ok())
                }
            }

            mod when_file_does_not_exist {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = SplitManifest::from_file("/123.txt");
                    assert_corpus_error(result, "failed to read split manifest:")
                }
            }
        }

        mod fn_from_json {
            use super::*;

            mod when_invalid_json_string_is_provided {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = SplitManifest::from_json("asd lol");
                    assert_corpus_error(result, "invalid split manifest JSON: expected value")
                }
            }

            mod when_invalid_manifest_format_is_provided {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let result = SplitManifest::from_json("{\"t\": 1}");
                    assert_corpus_error(result, "invalid split manifest JSON: unknown field")
                }
            }

            mod when_manifest_is_valid {
                use super::*;

                const CONTENT: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "123",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["en-river-dawn"],
                      "validation": ["en-night-station"],
                      "test": ["ru-winter-window"]
                    }
                "#;

                #[test]
                fn it_deserializes_it() {
                    let result = SplitManifest::from_json(CONTENT);
                    let manifest = SplitManifest {
                        schema_version: 1,
                        corpus_checksum: "123".to_string(),
                        strategy: "fixed-paired-document-holdout-v1".to_string(),
                        train: vec!["en-river-dawn".to_string()],
                        validation: vec!["en-night-station".to_string()],
                        test: vec!["ru-winter-window".to_string()],
                    };
                    assert_eq!(result.unwrap(), manifest)
                }
            }
        }

        mod fn_ids {
            use super::*;

            const CONTENT: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "123",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["en-river-dawn"],
                      "validation": ["en-night-station"],
                      "test": ["ru-winter-window"]
                    }
            "#;

            #[test]
            fn it_returns_document_ids_by_the_partition() {
                let manifest = SplitManifest::from_json(CONTENT).unwrap();
                assert_eq!(manifest.ids(Partition::Train), vec!["en-river-dawn"]);
                assert_eq!(
                    manifest.ids(Partition::Validation),
                    vec!["en-night-station"]
                );
                assert_eq!(manifest.ids(Partition::Test), vec!["ru-winter-window"]);
            }
        }

        mod fn_partition {
            use super::*;

            fn corpus() -> Corpus {
                Corpus::from_file(CORPUS_FILE).unwrap()
            }

            mod when_manifest_contains_unsupported_schema_version {
                use super::*;

                const CONTENT: &str = r#"
                    {
                      "schema_version": 2,
                      "corpus_checksum": "123",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["en-river-dawn"],
                      "validation": ["en-night-station"],
                      "test": ["ru-winter-window"]
                    }
                "#;

                #[test]
                fn it_returns_error() {
                    let manifest = SplitManifest::from_json(CONTENT).unwrap();
                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(result, "unsupported split schema version 2")
                }
            }

            mod when_manifest_contains_unsupported_split_strategy {
                use super::*;

                const CONTENT: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "123",
                      "strategy": "foo",
                      "train": ["en-river-dawn"],
                      "validation": ["en-night-station"],
                      "test": ["ru-winter-window"]
                    }
                "#;

                #[test]
                fn it_returns_error() {
                    let manifest = SplitManifest::from_json(CONTENT).unwrap();
                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(result, "unsupported split strategy \"foo\"")
                }
            }

            mod when_manifest_checksum_does_not_match_corpus_checksum {
                use super::*;

                const CONTENT: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "123",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["en-river-dawn"],
                      "validation": ["en-night-station"],
                      "test": ["ru-winter-window"]
                    }
                "#;

                #[test]
                fn it_returns_error() {
                    let manifest = SplitManifest::from_json(CONTENT).unwrap();
                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(
                        result,
                        &format!(
                            "corpus checksum mismatch: manifest=\"123\", actual=\"{}\"",
                            "fnv1a64:723b071980ae8a22"
                        ),
                    )
                }
            }

            mod when_partition_ids_are_empty {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let mut manifest = SplitManifest::from_file(CORPUS_MANIFEST).unwrap();
                    // remove validation partition ids
                    manifest.validation.clear();

                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(result, "\"validation\" partition is empty")
                }
            }

            mod when_documents_order_in_corpus_does_not_match_document_ids_order_in_manifest {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let mut manifest = SplitManifest::from_file(CORPUS_MANIFEST).unwrap();
                    // swap first and last element, thus producing inconsistency between corpus
                    // documents order and manifest document ids order
                    let validation_partition_size = manifest.validation.len();
                    let first_el = manifest.validation[0].clone();
                    let last_el = manifest.validation.last().unwrap().clone();
                    manifest.validation[0] = last_el;
                    manifest.validation[validation_partition_size - 1] = first_el;

                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(
                        result,
                        "\"validation\" partition IDs do not preserve corpus source order",
                    )
                }
            }

            mod when_document_id_is_present_in_manifest_but_absent_in_corpus {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let mut manifest = SplitManifest::from_file(CORPUS_MANIFEST).unwrap();
                    // add non-existing document id
                    manifest.validation.push("foo".to_string());

                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(
                        result,
                        "\"validation\" partition contains unknown document \"foo\"",
                    )
                }
            }

            mod when_same_document_id_appears_multiple_times {
                use super::*;

                const MANIFEST: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "fnv1a64:4efb31e29511a2c4",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["en-river-dawn"],
                      "validation": ["en-river-dawn"],
                      "test": ["en-river-dawn"]
                    }
                "#;

                const CORPUS: &str = r#"
                    [
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "pair-river-dawn",
                        "text": "foo"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let manifest = SplitManifest::from_json(MANIFEST).unwrap();
                    let corpus = Corpus::from_json(CORPUS).unwrap();
                    let result = manifest.partition(&corpus);

                    assert_corpus_error(
                        result,
                        &format!(
                            "document {:?} appears in more than one manifest position",
                            "en-river-dawn"
                        ),
                    )
                }
            }

            mod when_manifest_does_not_cover_corpus_documents {
                use super::*;

                #[test]
                fn it_returns_error() {
                    let mut manifest = SplitManifest::from_file(CORPUS_MANIFEST).unwrap();
                    // remove one document id from the tail
                    manifest.validation.pop();

                    let corpus = corpus();
                    let result = manifest.partition(&corpus);
                    assert_corpus_error(
                        result,
                        &format!(
                            "manifest does not cover corpus document {:?}",
                            "ru-night-station"
                        ),
                    )
                }
            }

            mod when_provenance_group_is_split_between_different_partitions {
                use super::*;

                const MANIFEST: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "fnv1a64:bc993bb97daa8d2d",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["ru-river-dawn"],
                      "validation": ["en-river-dawn"],
                      "test": ["es-river-dawn"]
                    }
                "#;

                const CORPUS: &str = r#"
                    [
                      {
                        "id": "ru-river-dawn",
                        "language": "ru",
                        "provenance_group": "river-dawn",
                        "text": "foo"
                      },
                      {
                        "id": "en-river-dawn",
                        "language": "en",
                        "provenance_group": "river-dawn",
                        "text": "bar"
                      },
                      {
                        "id": "es-river-dawn",
                        "language": "es",
                        "provenance_group": "river-dawn",
                        "text": "baz"
                      }
                    ]
                "#;

                #[test]
                fn it_returns_error() {
                    let manifest = SplitManifest::from_json(MANIFEST).unwrap();
                    let corpus = Corpus::from_json(CORPUS).unwrap();
                    let result = manifest.partition(&corpus);

                    assert_corpus_error(
                        result,
                        &format!(
                            "provenance group {:?} is split between {:?} and {:?} partitions",
                            "river-dawn", "train", "validation"
                        ),
                    )
                }
            }

            mod when_all_is_ok {
                use super::*;

                const MANIFEST: &str = r#"
                    {
                      "schema_version": 1,
                      "corpus_checksum": "fnv1a64:c5f392f1f77b65e7",
                      "strategy": "fixed-paired-document-holdout-v1",
                      "train": ["ru-river-dawn"],
                      "validation": ["en-river-sunrise"],
                      "test": ["es-river"]
                    }
                "#;

                const CORPUS: &str = r#"
                    [
                      {
                        "id": "ru-river-dawn",
                        "language": "ru",
                        "provenance_group": "river-dawn",
                        "text": "foo"
                      },
                      {
                        "id": "en-river-sunrise",
                        "language": "en",
                        "provenance_group": "river-sunrise",
                        "text": "bar"
                      },
                      {
                        "id": "es-river",
                        "language": "es",
                        "provenance_group": "river-common",
                        "text": "baz"
                      }
                    ]
                "#;

                #[test]
                fn it_computes_corpus_partitions() {
                    let manifest = SplitManifest::from_json(MANIFEST).unwrap();
                    let corpus = Corpus::from_json(CORPUS).unwrap();
                    let result = manifest.partition(&corpus);

                    let train_doc = Document {
                        id: "ru-river-dawn".to_string(),
                        language: "ru".to_string(),
                        provenance_group: "river-dawn".to_string(),
                        text: "foo".to_string(),
                    };
                    let validation_doc = Document {
                        id: "en-river-sunrise".to_string(),
                        language: "en".to_string(),
                        provenance_group: "river-sunrise".to_string(),
                        text: "bar".to_string(),
                    };
                    let test_doc = Document {
                        id: "es-river".to_string(),
                        language: "es".to_string(),
                        provenance_group: "river-common".to_string(),
                        text: "baz".to_string(),
                    };
                    let expected_corpus_partition = CorpusPartitions {
                        train: vec![&train_doc],
                        validation: vec![&validation_doc],
                        test: vec![&test_doc],
                    };

                    assert_eq!(result.unwrap(), expected_corpus_partition);
                }
            }
        }
    }
}
