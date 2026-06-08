

use super::{dataset::LobTransItem, /*tokenizer::Tokenizer*/};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::{SeqLengthOption, generate_padding_mask},
    prelude::*,
};
use std::sync::Arc;
use derive_new::new;

#[derive(Clone, Debug)]
pub struct LobTransBatcher {}

impl LobTransBatcher {
    pub fn new() -> Self {
        Self {}
    }
}


#[derive(Debug, Clone, new)]
pub struct LobTransTrainingBatch {
    pub tokens: Tensor<3, Float>,           // [batch_size, sequence_length, token_size]
    pub labels: Tensor<1, Int>,             // [batch_size]

    // pub mask_pad: Tensor<2, Bool>,       // padding mask for the tokenized text
}

#[derive (Debug, Clone, new)]
pub struct LobTransInferenceBatch {
    pub tokens: Tensor<3, Float>,           // [batch_size, sequence_length, token_size]
}


impl Batcher<LobTransItem, LobTransTrainingBatch> for LobTransBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<LobTransItem>,
        device: &Device,
    ) -> LobTransTrainingBatch {
        let batch_size = items.len();
        let sequence_length = items.first().map(|i| i.features.len()).unwrap_or(0);
        let token_length = items.first().map(|i| i.features.first().unwrap().len()).unwrap_or(0);

        let flattened_features: Vec<f64> = items
            .iter()
            .map(|item| item.features.clone())
            .flatten()
            .flatten()
            .collect();

        let flattened_labels: Vec<i64> = items.iter().map(|item| item.label.round() as i64).collect();

        // Construct tensors
        let inputs = Tensor::from_floats(
            TensorData::new(flattened_features, vec![batch_size, sequence_length, token_length]),
            device,
        );

        let targets = Tensor::from_ints(
            TensorData::new(flattened_labels, vec![batch_size]),
            device,
        );

        LobTransTrainingBatch {
            tokens: inputs,
            labels: targets,
        }
    }
}


impl Batcher<LobTransItem, LobTransInferenceBatch> for LobTransBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<LobTransItem>,
        device: &Device,
    ) -> LobTransInferenceBatch {
        let batch_size = items.len();
        let sequence_length = items.first().map(|i| i.features.len()).unwrap_or(0);
        let token_length = items.first().map(|i| i.features.first().unwrap().len()).unwrap_or(0);

        let flattened_features: Vec<f64> = items
            .iter()
            .map(|item| item.features.clone())
            .flatten()
            .flatten()
            .collect();


        // Construct tensors
        let inputs = Tensor::from_floats(
            TensorData::new(flattened_features, vec![batch_size, sequence_length, token_length]),
            device,
        );

        LobTransInferenceBatch {
            tokens: inputs,
        }
    }
}


