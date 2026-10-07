#[cfg(test)]
#[path = "support/mod.rs"]
mod support;

mod corpus;
mod data;
mod tokenizer;
mod vocabulary;
mod bigram;
mod metrics;
mod tensor;
mod nn;
mod autograd;
pub mod utils;
mod training;

fn main() {
    println!("Hello world!");
}
