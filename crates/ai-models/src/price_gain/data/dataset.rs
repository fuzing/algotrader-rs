
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
    pub patches: PriceGainPatches,
    pub label: f64,
}


pub struct PriceGainDataset {
    items: Vec<PriceGainItem>,           // the read in data file
}


const PATCH_TEMPORAL_WINDOW_SIZE: usize = 4;
const PATCH_TEMPORAL_STRIDE: usize = PATCH_TEMPORAL_WINDOW_SIZE;
const LOB_LEVELS: usize = 10;


type PatchData = [[f64; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];


pub struct PriceGainPatch {
    pub data: Box<PatchData>,
}

impl PriceGainPatch {
    pub fn new(data: PatchData) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}


pub struct PriceGainPatches {
    ask_price: Vec<PriceGainPatch>,
    bid_price: Vec<PriceGainPatch>,
    ask_volume: Vec<PriceGainPatch>,
    bid_volume: Vec<PriceGainPatch>,
}


impl PriceGainDataset {
    pub fn new(
        filename: PathBuf,
        prediction_temporal_window: usize,              // time window in number of consecutive LOB samples
    ) -> PriceGainDataset {
        let file = File::open(filename.clone()).expect(&format!("Couldn't open file {filename:?}"));
        let reader = BufReader::new(file);
        let data_file: ExtractedDataFile = serde_json::from_reader(reader).unwrap();

        // normalization factors for z-score, for price and volume
        let (volume_mean, volume_std_dev) = (data_file.volume_mean, data_file.volume_std_dev);
        let (price_mean, price_std_dev) = (data_file.mid_point_price_mean, data_file.mid_point_price_std_dev);

        let mut items: Vec<PriceGainItem> = Vec::new();

        for i in 0..=(data_file.data.len() - prediction_temporal_window) {

            // new embeddable
            let mut patches = PriceGainPatches {
                bid_price: Vec::new(),
                bid_volume: Vec::new(),
                ask_price: Vec::new(),
                ask_volume: Vec::new(),
            };

            for j in (0..=(prediction_temporal_window - PATCH_TEMPORAL_WINDOW_SIZE)).step_by(PATCH_TEMPORAL_STRIDE) {
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
                        bid_volume_patch[k][l] = (data_file.data[i + j + k].bids[l].volume as f64 - volume_mean) / volume_std_dev;
                        ask_price_patch[k][l] = (data_file.data[i + j + k].asks[l].price - price_mean) / price_std_dev;
                        ask_volume_patch[k][l] = (data_file.data[i + j + k].asks[l].volume as f64 - volume_mean) / volume_std_dev;
                    }
                }

                // add patches
                patches.bid_price.push(PriceGainPatch::new(bid_price_patch));
                patches.bid_volume.push(PriceGainPatch::new(bid_volume_patch));
                patches.ask_price.push(PriceGainPatch::new(ask_price_patch));
                patches.ask_volume.push(PriceGainPatch::new(ask_volume_patch));
            }

            // let label = data_file.data[i].trade_gain;
            let label = data_file.data[i].mid_point_gain;

            items.push(
                PriceGainItem {
                    patches,
                    label,
                }
            );

        }






        PriceGainDataset {
            items,
        }
    }
}

impl Dataset<PriceGainItem> for PriceGainDataset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        // will panic if index out of range
        Some(self.items[index])
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}


