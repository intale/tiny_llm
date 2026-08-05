use std::fmt::Debug;
use crate::corpus::CorpusError;

macro_rules! fixture_path {
    ($path:literal) => {
        concat!("tests/fixtures/", $path)
    };
}

pub const CORPUS_FILE: &str = fixture_path!("corpus/corpus.json");
pub const CORPUS_MANIFEST: &str = fixture_path!("corpus/manifest.json");


pub fn assert_corpus_error<R: Debug>(result: Result<R, CorpusError>, err_msg: &str) {
    match result {
        Ok(success) => panic!("Expected to have error, but got: {:?}", success),
        Err(e) => {
            if !e.message().contains(err_msg) {
                panic!("Expected error {:?} to include {:?}, but it didn't.", e.message(), err_msg)
            }
        }
    }
}
