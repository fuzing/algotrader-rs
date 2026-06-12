
use burn::{
    nn::{transformer::TransformerEncoderConfig},
    optim::AdamConfig,
    prelude::*,
};

use super::model::{
    embedder::EmbedderConfig,
    mlp::MLPConfig,
};



// Define configuration struct for the experiment
#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub embedder: EmbedderConfig,
    pub transformer: TransformerEncoderConfig,
    pub mlp: MLPConfig,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub shuffle_seed: u64,
    pub num_epochs: usize,
}

