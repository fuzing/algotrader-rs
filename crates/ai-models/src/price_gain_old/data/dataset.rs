
use std::{
    path::PathBuf,
    sync::Arc,
};

use burn::data::dataset::{
    Dataset,
};
use derive_new::new;

use crate::price_gain::data::data_spec::DataSpec;

use data_handlers::{
    mpk::{
        MpkDataReader,
        AccessType,
    },
    data_handler::DataReader,
};


#[derive(new, Clone, Debug)]
pub struct PriceGainItem {
    pub features: Vec<Vec<f64>>,        // [sequence_length, token_size]
    pub label: f64,
}


#[derive(Debug, Clone)]
pub struct PriceGainDataset {
    file: Arc<MpkDataReader<f64>>,
    pub spec: DataSpec,
}



impl PriceGainDataset {
    pub fn new(
        spec_path: &PathBuf,
        data_path: &PathBuf,
    ) -> PriceGainDataset {
        let spec = DataSpec::from_file(spec_path).expect(&format!("Couldn't open spec file {spec_path:?}"));
        let file = MpkDataReader::new(data_path.to_str().unwrap(), AccessType::Sequential);

        Self {
            spec,
            file: Arc::new(file),
        }
    }


    pub fn specs(&self) -> DataSpec {
        self.spec.clone()
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
        let label = values.pop()?; // O(1)

        // now arrange into [sequence_length, d_model]
        if values.len() != self.spec.sequence_length * self.spec.token_size {
            panic!("values is the wrong length: ({}) vs expect size of ({})", values.len(), self.spec.sequence_length * self.spec.token_size);
        }
        let chunks = values
            .chunks(self.spec.token_size)
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


