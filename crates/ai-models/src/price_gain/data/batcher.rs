

use super::{dataset::PriceGainItem, /*tokenizer::Tokenizer*/};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::{SeqLengthOption, generate_padding_mask},
    prelude::*,
};
use std::sync::Arc;
use derive_new::new;

#[derive(Clone, Debug)]
pub struct PriceGainBatcher {}

impl PriceGainBatcher {
    pub fn new() -> Self {
        Self {}
    }
}


#[derive(Debug, Clone, new)]
pub struct PriceGainTrainingBatch {
    pub tokens: Tensor<3, Float>,           // [batch_size, sequence_length, token_size]
    pub labels: Tensor<1, Int>,             // [batch_size]

    // pub mask_pad: Tensor<2, Bool>,       // padding mask for the tokenized text
}

#[derive (Debug, Clone, new)]
pub struct PriceGainInferenceBatch {
    pub tokens: Tensor<3, Float>,           // [batch_size, sequence_length, token_size]
}


impl Batcher<PriceGainItem, PriceGainTrainingBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainTrainingBatch {
        let batch_size = items.len();
        let sequence_length = items.first().map(|i| i.features.len()).unwrap_or(0);
        let token_length = items.first().map(|i| i.features.first().unwrap().len()).unwrap_or(0);

        let flattened_features = items
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

        PriceGainTrainingBatch {
            tokens: inputs,
            labels: targets,
        }
    }
}


impl Batcher<PriceGainItem, PriceGainInferenceBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainInferenceBatch {
        let batch_size = items.len();
        let sequence_length = items.first().map(|i| i.features.len()).unwrap_or(0);
        let token_length = items.first().map(|i| i.features.first().unwrap().len()).unwrap_or(0);

        let flattened_features = items
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

        PriceGainInferenceBatch {
            tokens: inputs,
        }
    }
}


