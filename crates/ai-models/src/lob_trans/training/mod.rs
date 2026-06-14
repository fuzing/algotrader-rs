
use burn::{
    nn::{transformer::TransformerEncoderConfig},
    optim::AdamConfig,
    prelude::*,
};
use burn::nn::LstmConfig;
use super::model::{
    embedder::EmbedderConfig,
    mlp::MLPConfig,
    multi_layer_lstm::MultiLayerLstmConfig,
};



// Define configuration struct for the experiment
#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub embedder: EmbedderConfig,
    pub transformer: TransformerEncoderConfig,
    // pub lstm: LstmConfig,
    pub lstm: MultiLayerLstmConfig,
    pub mlp: MLPConfig,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub device_seed: u64,
    pub shuffle_seed: u64,
    pub num_epochs: usize,
    pub lstm_layers: usize,
    pub lstm_hidden_size: usize,
}

