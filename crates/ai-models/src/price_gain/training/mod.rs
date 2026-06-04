
use burn::{
    nn::{transformer::TransformerEncoderConfig},
    optim::AdamConfig,
    prelude::*,
};


// Define configuration struct for the experiment
#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub shuffle_seed: u64,
    pub num_epochs: usize,
}

