// This is a basic price regression model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

pub mod mlp;
pub mod embedder;

use std::io::Write;
use super::data::batcher::{
    LobTransInferenceBatch,
    LobTransTrainingBatch,
};
use burn::{
    module::Param,
    nn::{
        // Embedding, EmbeddingConfig,
        Linear, LinearConfig,
        attention::SeqLengthOption,
        PositionalEncoding, PositionalEncodingConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        conv::{Conv2dConfig, Conv2d},
        lstm::{LstmConfig, Lstm},
    },
    prelude::*,
    tensor::{
        Distribution,
        activation::softmax
    },
    train::{RegressionOutput, ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn::nn::LstmState;
use self::mlp::{MLPConfig, MLP};
use self::embedder::{EmbedderConfig, Embedder};


// Define the model configuration
#[derive(Config, Debug)]
pub struct LobTransModelConfig {
    sequence_length: usize,          // same as number of patches
    token_size: usize,             // model embedding size
    n_classes: usize,
    loss_weights: Option<Vec<f32>>,
    lstm_layers: usize,
    lstm_hidden_size: usize,

    embedder: EmbedderConfig,
    transformer: TransformerEncoderConfig,
    lstm: LstmConfig,
    mlp: MLPConfig,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct LobTransModel {
    sequence_length: usize,
    token_size: usize,
    n_classes: usize,
    loss_weights: Option<Vec<f32>>,
    lstm_layers: usize,
    lstm_hidden_size: usize,

    embedder: Embedder,
    transformer: TransformerEncoder,
    lstm: Vec<Lstm>,
    output: MLP,
}

// Define functions for model initialization
impl LobTransModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> LobTransModel {
        let embedder = self.embedder.init(&device);
        let transformer = self.transformer.init(device);
        let lstm = (0..self.lstm_layers).map(|index| {
            // if index == 0 {
            //     LstmConfig::new(self.token_size, self.lstm_hidden_size, false)
            //         .with_batch_first(true).init(device)
            // }
            // else {
            //     LstmConfig::new(self.lstm_hidden_size, self.lstm_hidden_size, false)
            //         .with_batch_first(true).init(device)                
            // }
            self.lstm.init(&device)
        }).collect::<Vec<_>>();
        let output = self.mlp.init(device);

        LobTransModel {
            sequence_length: self.sequence_length,
            token_size: self.token_size,
            n_classes: self.n_classes,
            loss_weights: self.loss_weights.clone(),
            lstm_layers: self.lstm_layers,
            lstm_hidden_size: self.lstm_hidden_size,

            embedder,
            transformer,
            lstm,
            output,
        }
    }
}

/// Define model behavior
impl LobTransModel {
    // Defines forward pass for training
    pub fn forward(&self, item: LobTransTrainingBatch) -> ClassificationOutput {
        // Get batch and sequence length, and the device
        let [batch_size, _sequence_length, _token_size] = item.tokens.dims();
        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);

        // formulate the embedding tokens
        let x = self.embedder.forward(tokens);

        // eprintln!("Transformer embeddings shape {}", x.shape());

        // through the transformer
        let x = self
            .transformer
            .forward(TransformerEncoderInput::new(x));

        // eprintln!("Transformer output shape {}", x.shape());

        // we are only interested in the class token from the transformer output
        let x = x.slice([0..batch_size, 0..1]);

        // eprintln!("LSTM input shape {}", x.shape());

        // through the lstm layers
        let mut x = x;
        let mut prev_state: Option<LstmState<2>> = None;
        for layer in self.lstm.iter() {
            let (result, state) = layer.forward(x, prev_state);
            x = result;
            prev_state = Some(state);
        }

        // eprintln!("LSTM output shape {}", x.shape());

        // through the output linear layer
        let x = self.output.forward(x);

        // eprintln!("MLP output shape {}", x.shape());
        // eprintln!("MLP output {}", x);

        // classify, using only the class token
        let x = x
            // .slice([0..batch_size, 0..1, 0..d_model])
            .reshape([batch_size, self.n_classes]);

        // eprintln!("Classification shape {}", x.shape());
        // eprintln!("Classification {}", x);
        // panic!("yep");

        let loss = CrossEntropyLossConfig::new()
            .with_weights(self.loss_weights.clone())
            .init(&x.device())
            .forward(x.clone(), labels.clone());


        // Return the output and loss
        ClassificationOutput {
            loss,
            output: x,
            targets: labels,
        }

    }


    /// Defines forward pass for inference
    pub fn infer(&self, item: LobTransInferenceBatch) -> Tensor<2> {
        let [batch_size, sequence_length, token_size] = item.tokens.dims();

        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);

        // generate the embedding vectors
        let x = self.embedder.forward(tokens);

        // through the transformer
        let x = self
            .transformer
            .forward(TransformerEncoderInput::new(x));

        // we are only interested in the class token from the transformer output
        let x = x.slice([0..batch_size, 0..1]);

        // through the lstm layers
        let mut x = x;
        let mut prev_state: Option<LstmState<2>> = None;
        for layer in self.lstm.iter() {
            let (result, state) = layer.forward(x, prev_state);
            x = result;
            prev_state = Some(state);
        }

        // through the output linear layer
        let x = self.output.forward(x);

        // classify, using only the class token
        let x = x
            // .slice([0..batch_size, 0..1, 0..d_model])
            .reshape([batch_size, self.n_classes]);

        softmax(x, 1)
    }
}

/// Define training step
impl TrainStep for LobTransModel {
    type Input = LobTransTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: LobTransTrainingBatch) -> TrainOutput<ClassificationOutput> {
        // Run forward pass, calculate gradients and return them along with the output
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

/// Define validation step
impl InferenceStep for LobTransModel {
    type Input = LobTransTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: LobTransTrainingBatch) -> ClassificationOutput {
        // Run forward pass and return the output
        self.forward(item)
    }
}
