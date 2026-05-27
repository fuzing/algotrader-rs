
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


const PATCH_TEMPORAL_WINDOW_SIZE: usize = 4;
const PATCH_TEMPORAL_STRIDE: usize = PATCH_TEMPORAL_WINDOW_SIZE;
const LOB_LEVELS: usize = 10;


enum PatchSide {
    Bid,
    Ask,
}

type PatchData = [[f64; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];

pub struct PatchEmbeddable {
    pub side: PatchSide,
    pub data: Box<PatchData>,
}
impl PatchEmbeddable {
    pub fn new(side: PatchSide, data: PatchData) -> Self {
        Self {
            side,
            data: Box::new(data),
        }
    }
}

pub struct PriceGainEmbeddable {
    ask_price_patches: Vec<PatchEmbeddable>,
    bid_price_patches: Vec<PatchEmbeddable>,
    ask_volume_patches: Vec<PatchEmbeddable>,
    bid_volume_patches: Vec<PatchEmbeddable>,
}



impl PriceGainDataset {
    pub fn new(
        filename: PathBuf,
        prediction_temporal_window: usize,              // time window in number of consecutive LOB samples
        patch_temporal_window: usize,
        patch_temporal_stride: usize,
    ) -> PriceGainDataset {
        let file = File::open(filename.clone()).expect(&format!("Couldn't open file {filename:?}"));
        let reader = BufReader::new(file);
        let data_file: ExtractedDataFile = serde_json::from_reader(reader).unwrap();

        // normalization factors for z-score, for price and volume
        let (volume_mean, volume_std_dev) = (data_file.volume_mean, data_file.volume_std_dev);
        let (price_mean, price_std_dev) = (data_file.mid_point_price_mean, data_file.mid_point_price_std_dev);

        // lob depth is the number of bid/ask levels in the extracted data
        let lob_depth = data_file.data[0].bids.len();

        for i in 0..(data_file.data.len() - prediction_temporal_window) {
            for j in (0..(prediction_temporal_window - patch_temporal_window)).step_by(patch_temporal_stride) {
                let mut bid_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
                let mut ask_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
                let mut bid_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
                let mut ask_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
                // for k in (0..PATCH_TEMPORAL_WINDOW_SIZE) {
                //     for (index, bid) in data_file.data[i + j + k].bids.iter().enumerate() {
                //         bid_price_patch[k][index] = (bid.price - price_mean) / price_std_dev;
                //         bid_volume_patch[k][index] = (bid.volume - volume_mean) / volume_std_dev;
                //     }
                //     for (index, ask) in data_file.data[i + j + k].asks.iter().enumerate() {
                //         ask_price_patch[k][index] = (ask.price - price_mean) / price_std_dev;
                //         ask_volume_patch[k][index] = (ask.volume - volume_mean) / volume_std_dev;
                //     }
                // }

                for k in (0..PATCH_TEMPORAL_WINDOW_SIZE) {
                    for l in 0..LOB_LEVELS {
                        bid_price_patch[k][l] = (data_file.data[i + j + k].bids[l].price - price_mean) / price_std_dev;
                        bid_volume_patch[k][l] = (data_file.data[i + j + k].bids[l].volume - volume_mean) / volume_std_dev;
                        ask_price_patch[k][l] = (data_file.data[i + j + k].asks[l].price - price_mean) / price_std_dev;
                        ask_volume_patch[k][l] = (data_file.data[i + j + k].asks[l].volume - volume_mean) / volume_std_dev;
                    }
                }
            }
        }

        PriceGainDataset {
            items: vec![],
            labels: vec![],
            prediction_window: prediction_temporal_window,
            patch_window: patch_temporal_window,
            patch_stride: patch_temporal_stride,
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


