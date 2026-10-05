pub mod gradcheck;

use std::fmt;
use std::fmt::{Debug, Display};
use crate::autograd::tensor_core::{TensorOperation, TensorValue};
use crate::corpus::CorpusError;

macro_rules! fixture_path {
    ($path:literal) => {
        concat!("tests/fixtures/", $path)
    };
}

pub const CORPUS_FILE: &str = fixture_path!("corpus/corpus.json");
pub const CORPUS_MANIFEST: &str = fixture_path!("corpus/manifest.json");

pub const SIMPLE_CORPUS_FILE: &str = fixture_path!("corpus/simple_corpus.json");
pub const SIMPLE_CORPUS_MANIFEST: &str = fixture_path!("corpus/simple_manifest.json");



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

pub fn all_forward_ops(tensor_value: &TensorValue) -> Vec<String> {
    let parents = tensor_value.parents();
    let mut ops = vec![];
    parents.iter().for_each(|edge| {
        ops.extend(all_forward_ops(&edge.parent))
        }
    );
    ops.push(format!("{}", tensor_value.operation()));
    ops
}
