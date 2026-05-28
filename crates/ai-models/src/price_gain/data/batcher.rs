

use super::{dataset::PriceGainItem, tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::{SeqLengthOption, generate_padding_mask},
    prelude::*,
};
use std::sync::Arc;
use derive_new::new;

#[derive(Clone, Debug)]
pub struct PriceGainBatcher {}


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
impl Batcher<PriceGainItem, PriceGainTrainingBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a training batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainTrainingBatch {
        let mut tokens = Vec::with_capacity(items.len());
        let mut labels = Vec::with_capacity(items.len());

        for item in items {
            
            // let x = item.patches.ask_price[0].data;
            
            tokens.push(
                Tensor::from_data(
                    // Fix
                    TensorData::from([item.item[0], item.item[1]]),
                    device,
                )
            );
            labels.push(
                Tensor::from_data(
                    TensorData::from([item.label as f64]),
                    device,
                )
            );
        }


        PriceGainTrainingBatch {
            tokens: Tensor::cat(tokens, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}


/// Implement Batcher trait for PriceGainBatcher struct for inference
impl Batcher<PriceGainItem, PriceGainInferenceBatch> for PriceGainBatcher
{
    /// Batches a vector of price regression items into a inference batch
    fn batch(
        &self,
        items: Vec<PriceGainItem>,
        device: &Device,
    ) -> PriceGainInferenceBatch {
        let mut tokens = Vec::with_capacity(items.len());

        for item in items {
            tokens.push(
                Tensor::from_data(
                    // Fix
                    TensorData::from([item.item[0], item.item[1]]),
                    device,
                )
            );
        }

        PriceGainInferenceBatch {
            tokens: Tensor::cat(tokens, 0),
        }
    }
}


