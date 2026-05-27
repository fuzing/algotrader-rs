
use std::{
    fs::File,
    io::BufReader,
    path::PathBuf
};
use burn::data::dataset::{
    Dataset,
    InMemDataset,           // PMB in memory dataset
};
use derive_new::new;

use extractors::interval_extractor::{
    ExtractedDataFile,
    IntervalExtractionWithGain
};


#[derive(new, Clone, Debug)]
pub struct PriceGainItem {
    pub item: Vec<f64>,
    pub label: f64,
}


pub struct PriceGainDataset {
    items: Vec<Vec<f64>>,           // the read in data file
    labels: Vec<f64>,
    window: usize,                          // window to aggregate samples over
}

impl PriceGainDataset {
    pub fn new(
        filename: PathBuf,
        window: usize,              // time window in number of consecutive LOB samples
        lob_depth: usize,           // number of bids/asks to include 
    ) -> PriceGainDataset {
        let file = File::open(filename.clone()).expect(&format!("Couldn't open file {filename:?}"));
        let reader = BufReader::new(file);
        let data_file: ExtractedDataFile = serde_json::from_reader(reader).unwrap();
        
        let volume_mean = data_file.

        for i in 0..(data_file.data.len() - window) {

        }


        PriceGainDataset {
            items: vec![],
            labels: vec![],
            window,
        }
    }
}

impl Dataset<PriceGainItem> for PriceGainDataset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        // will panic if index out of range
        Some(PriceGainItem {
            item: self.items[index],
            label: self.labels[index],
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}


