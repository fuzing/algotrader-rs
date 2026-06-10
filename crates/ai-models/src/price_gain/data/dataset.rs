
use std::{
    path::PathBuf,
    sync::Arc,
};

use burn::data::dataset::{
    Dataset,
};
use derive_new::new;

use crate::price_gain::data::data_spec::PriceGainDataSpec;

use data_handlers::{
    mpk::{
        MpkDataReader,
        AccessType,
    },
    data_handler::DataReader,
};


#[derive(new, Clone, Debug)]
pub struct PriceGainItem {
    pub features: Vec<Vec<f32>>,        // [sequence_length, token_size]
    pub label: f32,
}


#[derive(Debug, Clone)]
pub struct PriceGainDataset {
    file: Arc<MpkDataReader<f32>>,
    // pub spec: PriceGainDataSpec,
    sequence_length: usize,
    token_size: usize,
    gain_threshold: f32,
    loss_threshold: f32,
}



impl PriceGainDataset {
    pub fn new(
        // spec_path: &PathBuf,
        data_path: &PathBuf,
        sequence_length: usize,
        token_size: usize,
        gain_threshold: f32,
        loss_threshold: f32,
    ) -> PriceGainDataset {
        // let spec = PriceGainDataSpec::from_file(spec_path).expect(&format!("Couldn't open spec file {spec_path:?}"));
        let file = MpkDataReader::<f32>::new(data_path.to_str().unwrap(), AccessType::Sequential);

        Self {
            // spec,
            file: Arc::new(file),
            sequence_length,
            token_size,
            gain_threshold,
            loss_threshold,
        }
    }


    pub fn num_classes() -> usize { 3 }

    pub fn class_name(label: usize) -> String {
        match label {
            0 => "Loss",
            1 => "Neutral",
            2 => "Gain",
            _ => panic!("Invalid class label {}", label)
        }.to_string()
    }
}

impl Dataset<PriceGainItem> for PriceGainDataset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        let mut values = self.file.read(index).unwrap();
        if values.is_empty() {
            return None;
        }

        // Last value from the row is the "label"
        let outcome = values.pop()?; // O(1)
        let label = if outcome >= self.gain_threshold {
            2.0     // gain
        }
        else if outcome > -self.loss_threshold {
            1.0     // neutral
        }
        else {
            0.0     // loss
        };

        // now arrange into [sequence_length, d_model]
        if values.len() != self.sequence_length * self.token_size {
            panic!("values is the wrong length: ({}) vs expect size of ({})", values.len(), self.sequence_length * self.token_size);
        }
        let chunks = values
            .chunks(self.token_size)
            .map(|slice| slice.to_vec())
            .collect();

        Some(PriceGainItem {
            features: chunks,
            label,
        })
    }

    fn len(&self) -> usize {
        self.file.len()
    }
}


