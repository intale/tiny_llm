#[cfg(test)]
#[path = "support/mod.rs"]
mod support;

mod corpus;
mod data;
mod tokenizer;
mod vocabulary;
mod bigram;
mod metrics;

fn main() {
    println!("Hello world!");
}
