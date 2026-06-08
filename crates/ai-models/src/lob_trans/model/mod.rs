// This is a basic price regression model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

use std::io::Write;
use std::time::Duration;
use super::data::batcher::{
    LobTransInferenceBatch,
    LobTransTrainingBatch,
};
use burn::{
    module::Param,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::SeqLengthOption,
        PositionalEncoding, PositionalEncodingConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::{
        Distribution,
        activation::softmax
    },
    train::{RegressionOutput, ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

// Define the model configuration
#[derive(Config, Debug)]
pub struct LobTransModelConfig {
    sequence_length: usize,          // same as number of patches
    token_size: usize,             // model embedding size

    transformer: TransformerEncoderConfig,
    n_classes: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct LobTransModel {
    sequence_length: usize,
    token_size: usize,

    // if we're using fixed weight sin/cos positional encodings
    // positional_encoder: PositionalEncoding,

    // [batch_size, class_token]
    class_tokens: Param<Tensor<3>>,

    // [batch_size, seq_length, d_model]
    positional_embeddings: Param<Tensor<3>>,

    transformer: TransformerEncoder,
    output: Linear,
    n_classes: usize,
}

// Define functions for model initialization
impl LobTransModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> LobTransModel {

        // only 1 class token (will be expanded/duplicated in model)
        let class_tokens = Param::from_tensor(
            Tensor::random([1, 1, self.token_size], Distribution::default(), device)
        );

        // learnable position encodings
        let positional_embeddings = Param::from_tensor(
            Tensor::random([1, self.sequence_length + 1, self.token_size], Distribution::default(), device)
        );

        // let positional_encoder = self.positional_encoder.init(device);
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);


        println!("LobTrans init completed");

        LobTransModel {
            sequence_length: self.sequence_length,
            token_size: self.token_size,

            class_tokens,
            positional_embeddings,
            // positional_encoder,
            transformer,
            output,
            n_classes: self.n_classes,
        }
    }
}

/// Define model behavior
impl LobTransModel {
    // Defines forward pass for training
    pub fn forward(&self, item: LobTransTrainingBatch) -> ClassificationOutput {
        // // Get batch and sequence length, and the device
        let [batch_size, seq_length, d_model] = item.tokens.dims();
        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        // eprintln!("Tokens shape: {:?}", tokens.shape());
        let labels = item.labels.to_device(device);
        // eprintln!("Labels shape: {:?}", labels.shape());

        // insert class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32,-1,-1]);
        // eprintln!("Class-tokens shape: {:?}", class_tokens.shape());

        let tokens_with_class = Tensor::cat(vec![class_tokens, tokens], 1);
        // eprintln!("Tokens_with_class shape: {:?}", tokens_with_class.shape());

        // positional encoding
        // eprintln!("positional embeddings shape: {:?}", self.positional_embeddings.val().shape());
        let tokens_with_class_and_pe = tokens_with_class.add(self.positional_embeddings.val());
        // TODO - PMB - should we divide by 2 or not?
        // eprintln!("Tokens_with_class_and_pe shape: {:?}", tokens_with_class_and_pe.shape());

        // through the transformer
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens_with_class_and_pe));
        // eprintln!("encoded shape: {:?}", encoded.shape());

        // we are only interested in the class token from the transformer output
        let encoded_class = encoded.slice([0..batch_size,0..1,0..d_model]);
        // eprintln!("encoded class shape: {:?}", encoded_class.shape());

        // through the output linear layer
        let output = self.output.forward(encoded_class);
        // eprintln!("output shape: {:?}", output.shape());

        // classify, using only the class token
        let output_classification = output
            // .slice([0..batch_size, 0..1, 0..d_model])
            .reshape([batch_size, self.n_classes]);
        // eprintln!("output_classification shape: {:?}", output_classification.shape());


        // let _ = std::io::stderr().flush();
        // panic!("help");


        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());


        // Return the output and loss
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }

    }



    /// Defines forward pass for inference
    // pub fn infer(&self, item: LobTransInferenceBatch) -> Tensor<2> {
    //     let [batch_size, seq_length, d_model] = item.tokens.dims();
    //
    //     let device = &self.transformer.devices()[0];
    //
    //     //
    //     // Move tensors to the correct device
    //     let tokens = item.tokens.to_device(device);
    //
    //     // Perform transformer encoding, calculate output and apply softmax for prediction
    //     let encoded = self
    //         .transformer
    //         .forward(TransformerEncoderInput::new(tokens));
    //     let output = self.output.forward(encoded);
    //     let output = output
    //         .slice([0..batch_size, 0..1])
    //         .reshape([batch_size, self.n_classes]);
    //
    //     softmax(output, 1)
    // }
    pub fn infer(&self, item: LobTransInferenceBatch) -> Tensor<2> {
        let [batch_size, seq_length, d_model] = item.tokens.dims();

        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);

        // insert class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32,-1,-1]);
        // eprintln!("Class-tokens shape: {:?}", class_tokens.shape());

        let tokens_with_class = Tensor::cat(vec![class_tokens, tokens], 1);
        // eprintln!("Tokens_with_class shape: {:?}", tokens_with_class.shape());

        // positional encoding
        // eprintln!("positional embeddings shape: {:?}", self.positional_embeddings.val().shape());
        let tokens_with_class_and_pe = tokens_with_class.add(self.positional_embeddings.val());
        // TODO - PMB - should we divide by 2 or not?
        // eprintln!("Tokens_with_class_and_pe shape: {:?}", tokens_with_class_and_pe.shape());

        // through the transformer
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens_with_class_and_pe));
        // eprintln!("encoded shape: {:?}", encoded.shape());

        // we are only interested in the class token from the transformer output
        let encoded_class = encoded.slice([0..batch_size,0..1,0..d_model]);
        // eprintln!("encoded class shape: {:?}", encoded_class.shape());

        // through the output linear layer
        let output = self.output.forward(encoded_class);
        // eprintln!("output shape: {:?}", output.shape());

        // classify, using only the class token
        let output_classification = output
            // .slice([0..batch_size, 0..1, 0..d_model])
            .reshape([batch_size, self.n_classes]);

        softmax(output_classification, 1)
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
