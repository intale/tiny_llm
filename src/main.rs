#[cfg(test)]
#[path = "support/mod.rs"]
mod support;

mod autograd;
mod bigram;
mod corpus;
mod data;
mod metrics;
mod models;
mod nn;
mod tensor;
mod tokenizer;
mod training;
pub mod utils;
mod vocabulary;

fn main() {
    if std::env::args()
        .collect::<Vec<_>>()
        .contains(&"--ngram-demo".to_owned())
    {
        neural_ngram_demo::neural_ngram()
    } else {
        println!("Hello world!");
    }
}

mod neural_ngram_demo {
    use super::*;
    use crate::autograd::tensor_core::AutogradContext;
    use crate::corpus::{Corpus, Partition, SplitManifest};
    use crate::data::{CausalWindowConfig, EncodedCorpusPartitions};
    use crate::models::neural_ngram::{NeuralNgram, NeuralNgramConfig};
    use crate::nn::init::SplitMix64;
    use crate::tokenizer::bpe::{BOS_TOKEN_ID, BpeTokenizer, EOS_TOKEN_ID};
    use crate::tokenizer::bpe_trainer::{BpeTrainer, TOKENS_VOCABULARY_BASE};
    use crate::training::adam_w::{AdamW, AdamWConfig};
    use crate::training::batch::{BatchDocument, BatchOrder, MiniBatchConfig, MiniBatchEpoch};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::io;
    use std::io::Write;

    const LOREM_CORPUS_FILE: &str = "demo/fixtures/lorem_corpus.json";
    const LOREM_CORPUS_MANIFEST: &str = "demo/fixtures/lorem_manifest.json";

    pub fn neural_ngram() {
        let bpe_trainer = BpeTrainer::new(4);
        let corpus = Corpus::from_file(LOREM_CORPUS_FILE).unwrap();
        let manifest = SplitManifest::from_file(LOREM_CORPUS_MANIFEST).unwrap();
        let partitions = manifest.partition(&corpus).unwrap();
        let training = bpe_trainer.train(&partitions).unwrap();
        let tokenizer = BpeTokenizer::from_training(&training).unwrap();
        let encoded_partitions = EncodedCorpusPartitions::from_partitions(&partitions, &tokenizer);

        let vocabulary_size = tokenizer.layout().vocabulary_size();
        const CONTEXT_LENGTH: usize = 3;
        let embedding_width = 64;
        let hidden_width = 128;
        let config = NeuralNgramConfig::new(
            vocabulary_size,
            CONTEXT_LENGTH,
            embedding_width,
            hidden_width,
        )
        .unwrap();

        let mut rng = SplitMix64::from_seed(0);
        let neural_ngram = NeuralNgram::new(config, &mut rng).unwrap();

        let batch_documents = encoded_partitions
            .documents(Partition::Train)
            .iter()
            .map(|doc| BatchDocument::from_encoded(doc))
            .collect::<Vec<_>>();
        let validation_batch_docs = encoded_partitions
            .documents(Partition::Validation)
            .iter()
            .map(|doc| BatchDocument::from_encoded(doc))
            .collect::<Vec<_>>();
        let window_config = CausalWindowConfig::new(CONTEXT_LENGTH, 1).unwrap();
        let mini_batch_config =
            MiniBatchConfig::new(32, BatchOrder::Shuffled { seed: 10 }).unwrap();
        let mini_batch = MiniBatchEpoch::build(
            Partition::Train,
            &batch_documents,
            window_config,
            mini_batch_config,
        )
        .unwrap();
        let validation_minibatch = MiniBatchEpoch::build(
            Partition::Validation,
            &validation_batch_docs,
            window_config,
            mini_batch_config,
        )
        .unwrap();

        let optimizer_config = AdamWConfig::new(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();
        let mut optimizer = AdamW::new(optimizer_config);
        const EPOCHS: usize = 3;

        println!("Total parameters count: {:?}", config.parameter_count());
        println!("Starting training process...");
        for epoch in 0..EPOCHS {
            println!("Starting epoch {}/{}", epoch + 1, EPOCHS);

            let mut total_loss = 0.0;
            let mut total_examples = 0;

            let mut minibatches_processed = 0;
            let total_minibatches = mini_batch.batches().len();
            for batch in mini_batch.batches().iter() {
                let batch_loss;
                {
                    let loss = neural_ngram.loss(batch).unwrap();
                    batch_loss = loss.value().as_slice()[0];

                    // Calculate gradients
                    loss.backward_with_release().unwrap();
                }

                // Update W using the gradients calculated above
                optimizer.step(neural_ngram.parameters()).unwrap();

                // Clear the gradients before the next minibatch
                for parameter in neural_ngram.parameters() {
                    parameter.tensor().zero_grad().unwrap();
                }

                total_loss += batch_loss * batch.batch_width() as f64;
                total_examples += batch.batch_width();

                minibatches_processed += 1;

                if minibatches_processed % 100 == 0 || minibatches_processed == total_minibatches {
                    print!("\rMinibatches processed: {minibatches_processed}/{total_minibatches}");
                    io::stdout().flush().unwrap();
                }
            }

            let mut validation_total_loss = 0.0;
            let mut total_validation_examples = 0;
            for batch in validation_minibatch.batches().iter() {
                let loss = neural_ngram
                    .loss_with_context(AutogradContext::no_grad(), batch)
                    .unwrap();
                validation_total_loss += loss.value().as_slice()[0] * batch.batch_width() as f64;
                total_validation_examples += batch.batch_width();
            }

            let mut final_training_total_loss = 0.0;
            for batch in mini_batch.batches().iter() {
                let loss = neural_ngram
                    .loss_with_context(AutogradContext::no_grad(), batch)
                    .unwrap();
                final_training_total_loss += loss.value().as_slice()[0] * batch.batch_width() as f64;
            }

            println!(
                "\nFinished epoch {}: online training mean NLL = {:.6}, validation mean NLL = {:.6}, final training mean NLL = {:.6}",
                epoch + 1,
                total_loss / total_examples as f64,
                validation_total_loss / total_validation_examples as f64,
                final_training_total_loss / total_examples as f64
            );
        }

        println!("Training finished.");

        let tokens = tokenizer.encode_utf8("Dolorem");
        let mut masked_tokens = vec![BOS_TOKEN_ID];

        println!("Tokens: {:?}", tokens);

        let mut sequence: Vec<u32> = tokens.get(0..CONTEXT_LENGTH).unwrap().to_vec();
        let mut generated_tokens_count = 0;
        const MAX_TOKENS: usize = 1000;

        let mut tokens_seen = FxHashSet::default();

        loop {
            let context_tokens = sequence.last_chunk::<CONTEXT_LENGTH>().unwrap();
            if tokens_seen.get(context_tokens).is_some() {
                masked_tokens.extend_from_slice(context_tokens);
            } else {
                tokens_seen.insert(context_tokens.clone());
            }
            // println!("context tokens: {:?}", context_tokens);
            let next_token = neural_ngram
                .greedy_next(context_tokens, &masked_tokens)
                .unwrap();
            if next_token == EOS_TOKEN_ID || generated_tokens_count == MAX_TOKENS {
                break;
            }
            generated_tokens_count += 1;
            sequence.push(next_token);
            let decoded = tokenizer.decode_content(&sequence).unwrap();
            println!("Generated: {:?}", String::from_utf8_lossy(&decoded));
        }

        let decoded = tokenizer.decode_content(&sequence).unwrap();
        println!("Final: {:?}", String::from_utf8_lossy(&decoded));
    }
}
