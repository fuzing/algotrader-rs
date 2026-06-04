// This is a basic price regression model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

use super::data::batcher::{
    PriceGainInferenceBatch,
    PriceGainTrainingBatch
};
use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::SeqLengthOption,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::activation::softmax,
    train::{RegressionOutput, ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

// Define the model configuration
#[derive(Config, Debug)]
pub struct PriceGainModelConfig {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PriceGainModel {
    transformer: TransformerEncoder,
    output: Linear,
    n_classes: usize,
}

// Define functions for model initialization
impl PriceGainModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> PriceGainModel {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);

        PriceGainModel {
            transformer,
            output,
            n_classes: self.n_classes,
        }
    }
}

/// Define model behavior
impl PriceGainModel {
    // Defines forward pass for training
    pub fn forward(&self, item: PriceGainTrainingBatch) -> ClassificationOutput {
        // // Get batch and sequence length, and the device
        let [batch_size, seq_length, d_model] = item.tokens.dims();
        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

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
    pub fn infer(&self, item: PriceGainInferenceBatch) -> Tensor<2> {
        let [batch_size, seq_length, d_model] = item.tokens.dims();

        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);

        // Perform transformer encoding, calculate output and apply softmax for prediction
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);
        softmax(output, 1)

    }


}

/// Define training step
impl TrainStep for PriceGainModel {
    type Input = PriceGainTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: PriceGainTrainingBatch) -> TrainOutput<ClassificationOutput> {
        // Run forward pass, calculate gradients and return them along with the output
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

/// Define validation step
impl InferenceStep for PriceGainModel {
    type Input = PriceGainTrainingBatch;
    type Output = ClassificationOutput;

    fn step(&self, item: PriceGainTrainingBatch) -> ClassificationOutput {
        // Run forward pass and return the output
        self.forward(item)
    }
}
