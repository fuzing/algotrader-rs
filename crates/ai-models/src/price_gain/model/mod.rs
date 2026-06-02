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
    vocab_size: usize,
    seq_length: SeqLengthOption,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PriceGainModel {
    transformer: TransformerEncoder,
    embedding_token: Embedding,
    // embedding_pos: Embedding,
    output: Linear,
    n_classes: usize,

}

// Define functions for model initialization
impl PriceGainModelConfig {
    /// Initializes a model with default weights
    pub fn init(&self, device: &Device) -> PriceGainModel {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let max_seq_length = match self.seq_length {
            SeqLengthOption::Fixed(max) | SeqLengthOption::Max(max) => max,
            SeqLengthOption::NoMax => panic!(
                "Price/Volume category requires a max sequence length because of the embedding strategy."
            ),
        };
        let embedding_pos =
            EmbeddingConfig::new(max_seq_length, self.transformer.d_model).init(device);

        PriceGainModel {
            transformer,
            embedding_token,
            // embedding_pos,
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
        // let [batch_size, seq_length] = item.tokens.dims();
        let [batch_size, seq_length, d_model] = item.tokens.dims();

        // println!("Batch size: {}, Sequence length {}, D_Model {}", batch_size, seq_length, d_model);


        // PMB device from transformer instead of from the embedding_token layer
        // let device = &self.embedding_token.devices()[0];
        let device = &self.transformer.devices()[0];

        //
        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);

        // let mask_pad = item.mask_pad.to_device(device);
        //
        // // Calculate token and position embeddings, and combine them
        // let index_positions = Tensor::arange(0..seq_length as i64, device)
        //     .reshape([1, seq_length])
        //     .repeat_dim(0, batch_size);
        // let embedding_positions = self.embedding_pos.forward(index_positions);
        // let embedding_tokens = self.embedding_token.forward(tokens);
        // let embedding = (embedding_positions + embedding_tokens) / 2;
        //
        // Perform transformer encoding, calculate output and loss
        // let encoded = self
        //     .transformer
        //     .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));

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

        // let device = Device::default();
        //
        // // Return the output and loss - PMB
        // ClassificationOutput {
        //     loss: Tensor::<1, Float>::zeros(Shape::new([1]), &device),
        //     output: Tensor::<2, Float>::zeros(Shape::new([1,1]), &device),
        //     targets: Tensor::<1, Int>::zeros(Shape::new([1]), &device),
        // }
    }

    /// Defines forward pass for inference
    pub fn infer(&self, item: PriceGainInferenceBatch) -> Tensor<2> {
        // Get batch and sequence length, and the device
        // let [batch_size, seq_length] = item.tokens.dims();
        // let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        // let tokens = item.tokens.to_device(device);
        // let mask_pad = item.mask_pad.to_device(device);
        // // Calculate token and position embeddings, and combine them
        // let index_positions = Tensor::arange(0..seq_length as i64, device)
        //     .reshape([1, seq_length])
        //     .repeat_dim(0, batch_size);
        // let embedding_positions = self.embedding_pos.forward(index_positions);
        // let embedding_tokens = self.embedding_token.forward(tokens);
        // let embedding = (embedding_positions + embedding_tokens) / 2;

        // // Perform transformer encoding, calculate output and apply softmax for prediction
        // let encoded = self
        //     .transformer
        //     .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        // let output = self.output.forward(encoded);
        // let output = output
        //     .slice([0..batch_size, 0..1])
        //     .reshape([batch_size, self.n_classes]);
        // softmax(output, 1)

        // PMB
        Tensor::<2>::from([1, 1])
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
