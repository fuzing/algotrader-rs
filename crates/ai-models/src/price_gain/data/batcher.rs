

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
    pub tokens: Tensor<2, Float>,
    pub labels: Tensor<1, Float>,
}

#[derive (Debug, Clone, new)]
pub struct PriceGainInferenceBatch {
    pub tokens: Tensor<2, Float>,
}

/// Implement Batcher trait for PriceGainBatcher struct for training
// impl Batcher<PriceGainItem, PriceGainTrainingBatch> for PriceGainBatcher
// {
//     /// Batches a vector of price regression items into a training batch
//     fn batch(
//         &self,
//         items: Vec<PriceGainItem>,
//         device: &Device,
//     ) -> PriceGainTrainingBatch {
//         let mut tokens = Vec::with_capacity(items.len());
//         let mut labels = Vec::with_capacity(items.len());
//
//         for item in items {
//
//             // let x = item.patches.ask_price[0].data;
//
//             tokens.push(
//                 Tensor::from_data(
//                     // Fix
//                     TensorData::from([item.item[0], item.item[1]]),
//                     device,
//                 )
//             );
//             labels.push(
//                 Tensor::from_data(
//                     TensorData::from([item.label as f64]),
//                     device,
//                 )
//             );
//         }
//
//
//         PriceGainTrainingBatch {
//             tokens: Tensor::cat(tokens, 0),
//             labels: Tensor::cat(labels, 0),
//         }
//     }
// }

impl Batcher<PriceGainItem, PriceGainTrainingBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainTrainingBatch {
        let batch_size = items.len();
        let feature_dim = items.first().map(|i| i.features.len()).unwrap_or(0);

        // Flatten feature vectors
        let flattened_features: Vec<f64> = items
            .iter()
            .flat_map(|item| item.features.clone())
            .collect();

        let flattened_labels: Vec<f64> = items.iter().map(|item| item.label).collect();

        // Construct tensors
        let inputs = Tensor::from_floats(
            TensorData::new(flattened_features, vec![batch_size, feature_dim]),
            device,
        );

        let targets = Tensor::from_floats(
            TensorData::new(flattened_labels, vec![batch_size]),
            device,
        );

        PriceGainTrainingBatch {
            tokens: inputs,
            labels: targets,
        }
    }
}



/// Implement Batcher trait for PriceGainBatcher struct for inference
// impl Batcher<PriceGainItem, PriceGainInferenceBatch> for PriceGainBatcher
// {
//     /// Batches a vector of price regression items into a inference batch
//     fn batch(
//         &self,
//         items: Vec<PriceGainItem>,
//         device: &Device,
//     ) -> PriceGainInferenceBatch {
//         let mut tokens = Vec::with_capacity(items.len());
//
//         for item in items {
//             tokens.push(
//                 Tensor::from_data(
//                     // Fix
//                     TensorData::from([item.item[0], item.item[1]]),
//                     device,
//                 )
//             );
//         }
//
//         PriceGainInferenceBatch {
//             tokens: Tensor::cat(tokens, 0),
//         }
//     }
// }
impl Batcher<PriceGainItem, PriceGainInferenceBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a inference batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainInferenceBatch {
        let batch_size = items.len();
        let feature_dim = items.first().map(|i| i.features.len()).unwrap_or(0);

        // Flatten feature vectors
        let flattened_features: Vec<f64> = items
            .iter()
            .flat_map(|item| item.features.clone())
            .collect();

        // let flattened_labels: Vec<f64> = items.iter().map(|item| item.label).collect();

        // Construct tensors
        let inputs = Tensor::from_floats(
            TensorData::new(flattened_features, vec![batch_size, feature_dim]),
            device,
        );

        // let targets = Tensor::from_floats(
        //     TensorData::new(flattened_labels, vec![batch_size]),
        //     device,
        // );

        PriceGainInferenceBatch {
            tokens: inputs,
        }
    }
}

