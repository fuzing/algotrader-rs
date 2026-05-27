
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
    prediction_window: usize,                          // temporal window necessary to provide prediction
    patch_window: usize,                                // temporal window for width of patch
    patch_stride: usize,                                // usually set to same as patch_window
}


pub struct PriceGainEmbeddable {
}



impl PriceGainDataset {
    pub fn new(
        filename: PathBuf,
        prediction_window: usize,              // time window in number of consecutive LOB samples
        patch_window: usize,
        patch_stride: usize,
    ) -> PriceGainDataset {
        let file = File::open(filename.clone()).expect(&format!("Couldn't open file {filename:?}"));
        let reader = BufReader::new(file);
        let data_file: ExtractedDataFile = serde_json::from_reader(reader).unwrap();

        // normalization factors for z-score, for price and volume
        let (volume_mean, volume_std_dev) = (data_file.volume_mean, data_file.volume_std_dev);
        let (price_mean, price_std_dev) = (data_file.mid_point_price_mean, data_file.mid_point_price_std_dev);

        // lob depth is the number of bid/ask levels in the extracted data
        let lob_depth = data_file.data[0].bids.len();




        for i in 0..(data_file.data.len() - prediction_window) {
            for j in (0..(prediction_window - patch_window)).step_by(patch_stride) {
                for k in (0..patch_window) {

                }
            }
        }

        PriceGainDataset {
            items: vec![],
            labels: vec![],
            prediction_window,
            patch_window,
            patch_stride,
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


