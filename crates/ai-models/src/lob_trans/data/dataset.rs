
use std::{
    path::PathBuf,
    sync::Arc,
};

use burn::data::dataset::{
    Dataset,
};
use derive_new::new;
use serde::de::DeserializeOwned;
use data_handlers::{
    mpk::{
        MpkDataReader,
        AccessType,
    },
    data_handler::DataReader,
};


#[derive(new, Clone, Debug)]
pub struct LobTransItem {
    pub features: Vec<Vec<f32>>,        // [sequence_length, token_size]
    pub label: f32,
}


#[derive(Debug, Clone)]
pub struct LobTransDataset {
    file: Arc<MpkDataReader<f32>>,
    sequence_length: usize,
    token_size: usize,
}


impl LobTransDataset {
    pub fn new(
        data_path: &PathBuf,
        sequence_length: usize,
        token_size: usize,
    ) -> LobTransDataset {
        let file = MpkDataReader::<f32>::new(data_path.to_str().unwrap(), AccessType::Sequential);

        Self {
            file: Arc::new(file),
            sequence_length,
            token_size,
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

impl Dataset<LobTransItem> for LobTransDataset
{
    fn get(&self, index: usize) -> Option<LobTransItem> {
        let mut values = self.file.read(index).unwrap();
        if values.is_empty() {
            return None;
        }

        // Last value from the row is the "label"
        let label = values.pop()?; // O(1)

        // now arrange into [sequence_length, d_model]
        if values.len() != self.sequence_length * self.token_size {
            panic!("values is the wrong length: ({}) vs expect size of ({})", values.len(), self.sequence_length * self.token_size);
        }
        let chunks = values
            .chunks(self.token_size)
            .map(|slice| slice.to_vec())
            .collect();

        Some(LobTransItem {
            features: chunks,
            label,
        })
    }

    fn len(&self) -> usize {
        self.file.len()
    }
}


