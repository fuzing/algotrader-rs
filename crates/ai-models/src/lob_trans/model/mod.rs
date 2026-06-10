// This is a basic price regression model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

// mod lob_trans_mlp;

use std::io::Write;
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
        conv::{Conv2dConfig, Conv2d},
    },
    prelude::*,
    tensor::{
        Distribution,
        activation::softmax
    },
    train::{RegressionOutput, ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
// use crate::lob_trans::model::lob_trans_mlp::{LobTransMLP, LobTransMLPConfig};

// Define the model configuration
#[derive(Config, Debug)]
pub struct LobTransModelConfig {
    sequence_length: usize,          // same as number of patches
    token_size: usize,             // model embedding size

    transformer: TransformerEncoderConfig,
    // output_hidden_size: usize,
    n_classes: usize,
    loss_weights: Option<Vec<f32>>,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct LobTransModel {
    sequence_length: usize,
    token_size: usize,

    // [batch_size, class_token]
    class_tokens: Param<Tensor<3>>,

    // [batch_size, seq_length, d_model]
    positional_embeddings: Param<Tensor<3>>,

    transformer: TransformerEncoder,
    output: Linear,
    // output: LobTransMLP,
    n_classes: usize,
    loss_weights: Option<Vec<f32>>,

}

// Define functions for model initialization
impl LobTransModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> LobTransModel {

        // in pytorch
        // patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size= patch_size, stride= patch_size);
        // embed = patch_embed.flatten(2).transpose(1,2);
        //
        // in rust????? (not done below)
        // let patch_size = 24;
        // let init = Initializer::KaimingUniform {
        //     gain: 1.0 / 3.0f64.sqrt(),
        //     fan_out_only: true, // test that fan_out is passed to `init_with()`
        // };
        //
        // let config = Conv2dConfig::new([num_channels, 1], [patch_size, patch_size]).with_initializer(init.clone());
        // let c = config.init(&device);
        // let _ = c.weight.val(); // initializes param


        // only 1 class token (will be expanded/duplicated in model)
        let class_tokens = Param::from_tensor(
            Tensor::random([1, 1, self.token_size], Distribution::default(), device)
        );

        // learnable position encodings
        let positional_embeddings = Param::from_tensor(
            Tensor::random([1, self.sequence_length + 1, self.token_size], Distribution::default(), device)
        );
        
        let transformer = self.transformer.init(device);
        
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        // let output = LobTransMLPConfig::new(
        //     self.transformer.d_model,
        //     self.output_hidden_size,
        //     self.n_classes,
        // ).init(device);


        LobTransModel {
            sequence_length: self.sequence_length,
            token_size: self.token_size,

            class_tokens,
            positional_embeddings,
            // positional_encoder,
            transformer,
            output,
            n_classes: self.n_classes,
            loss_weights: self.loss_weights.clone(),
        }
    }
}

/// Define model behavior
impl LobTransModel {
    // Defines forward pass for training
    pub fn forward(&self, item: LobTransTrainingBatch) -> ClassificationOutput {
        // Get batch and sequence length, and the device
        let [batch_size, sequence_length, token_size] = item.tokens.dims();
        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);

        // insert class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32, -1, -1]);
        let tokens_with_class = Tensor::cat(vec![class_tokens, tokens], 1);

        // positional encoding
        let tokens_with_class_and_positional_encoding = tokens_with_class.add(self.positional_embeddings.val());
        // TODO - PMB - should we divide by 2 or not?

        // through the transformer
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens_with_class_and_positional_encoding));

        // we are only interested in the class token from the transformer output
        // let encoded_class = encoded.slice([0..batch_size, 0..1, 0..token_size]);
        let encoded_class = encoded.slice([0..batch_size, 0..1]);

        // through the output linear layer
        let output = self.output.forward(encoded_class);

        // classify, using only the class token
        let output_classification = output
            // .slice([0..batch_size, 0..1, 0..d_model])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .with_weights(self.loss_weights.clone())
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
    pub fn infer(&self, item: LobTransInferenceBatch) -> Tensor<2> {
        let [batch_size, sequence_length, token_size] = item.tokens.dims();

        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);

        // insert class tokens
        let class_tokens = self.class_tokens.val().expand([batch_size as i32, -1, -1]);

        let tokens_with_class = Tensor::cat(vec![class_tokens, tokens], 1);

        // positional encoding
        let tokens_with_class_and_positional_encoding = tokens_with_class.add(self.positional_embeddings.val());
        // TODO - PMB - should we divide by 2 or not? Probably not given that these are learnable

        // through the transformer
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens_with_class_and_positional_encoding));

        // we are only interested in the class token from the transformer output
        // let encoded_class = encoded.slice([0..batch_size, 0..1, 0..token_size]);
        let encoded_class = encoded.slice([0..batch_size, 0..1]);

        // through the output linear layer
        let output = self.output.forward(encoded_class);

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
