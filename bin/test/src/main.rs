

use burn::{
    prelude::*,
    nn::{PositionalEncodingConfig, PositionalEncoding},
    tensor::{Tensor, TensorData, Shape},
};
use serde::{ Serialize, Deserialize };
use csv;

const NUMBER_OF_SNAPSHOTS: usize = 400;
const PREDICTION_TEMPORAL_WINDOW_SIZE: usize = 100;



#[derive(Debug, Serialize, Deserialize)]
pub struct Level {
    price: f64,
    volume: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot {
    bids: Vec<Level>,
    asks: Vec<Level>,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct PriceGainItem {
    pub patches: PriceGainPatches,
    pub label: f64,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct PriceGainDataset {
    items: Vec<PriceGainItem>,           // the read in data file
}

const PATCH_TEMPORAL_WINDOW_SIZE: usize = 8;
const PATCH_TEMPORAL_STRIDE: usize = 4;
const LOB_LEVELS: usize = 10;


type PatchData = [[f64; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];


#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct PriceGainPatches {
    ask_price: Vec<PriceGainPatch>,
    bid_price: Vec<PriceGainPatch>,
    ask_volume: Vec<PriceGainPatch>,
    bid_volume: Vec<PriceGainPatch>,
}


fn main() {

    let device = Default::default();
    let values_per_token = 10;
    let pe = PositionalEncodingConfig::new(values_per_token)
        .with_max_sequence_size(100)
        .with_max_timescale(1_000_000)
        .init(&device);

    const BATCH_SIZE: usize = 16;
    let t = Tensor::<2>::zeros(Shape::new([BATCH_SIZE, 10]), &device);
    println!("Tensor {:?}", t);
    // let x = pe.forward();




    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Snapshot windows
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(NUMBER_OF_SNAPSHOTS);
    for _ in 0..NUMBER_OF_SNAPSHOTS {
        snapshots.push(Snapshot{
            bids: (0..LOB_LEVELS).map(|i| Level{price: i as f64, volume: i as u32}).collect(),
            asks: (0..LOB_LEVELS).map(|i| Level{price: i as f64, volume: i as u32}).collect(),
        });
    }

    // just some bogus numbers
    let price_mean: f64 = 5.0;
    let price_std_dev: f64 = 2.0;
    let volume_mean: f64 = 5.0;
    let volume_std_dev: f64 = 2.0;

    let mut items: Vec<PriceGainItem> = Vec::new();

    for i in 0..=(snapshots.len() - PREDICTION_TEMPORAL_WINDOW_SIZE) {
        let mut patches = PriceGainPatches {
            bid_price: Vec::new(),
            bid_volume: Vec::new(),
            ask_price: Vec::new(),
            ask_volume: Vec::new(),
        };

        for j in (0..=(PREDICTION_TEMPORAL_WINDOW_SIZE - PATCH_TEMPORAL_WINDOW_SIZE)).step_by(PATCH_TEMPORAL_STRIDE) {
            let mut bid_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut ask_price_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut bid_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];
            let mut ask_volume_patch: PatchData = [[0.0; LOB_LEVELS]; PATCH_TEMPORAL_WINDOW_SIZE];

            for k in (0..PATCH_TEMPORAL_WINDOW_SIZE) {
                for l in 0..LOB_LEVELS {
                    bid_price_patch[k][l] = (snapshots[i + j + k].bids[l].price - price_mean) / price_std_dev;
                    bid_volume_patch[k][l] = (snapshots[i + j + k].bids[l].volume as f64 - volume_mean) / volume_std_dev;
                    ask_price_patch[k][l] = (snapshots[i + j + k].asks[l].price - price_mean) / price_std_dev;
                    ask_volume_patch[k][l] = (snapshots[i + j + k].asks[l].volume as f64 - volume_mean) / volume_std_dev;
                }
            }

            // add patches
            patches.bid_price.push(PriceGainPatch::new(bid_price_patch));
            patches.bid_volume.push(PriceGainPatch::new(bid_volume_patch));
            patches.ask_price.push(PriceGainPatch::new(ask_price_patch));
            patches.ask_volume.push(PriceGainPatch::new(ask_volume_patch));
        }

        // let label = data_file.data[i].trade_gain;
        let label = 23.0;

        items.push(
            PriceGainItem {
                patches,
                label,
            }
        );


    }

    // number of patches per item
    let predicted_n_patches_per_item = ((PREDICTION_TEMPORAL_WINDOW_SIZE - PATCH_TEMPORAL_WINDOW_SIZE) / PATCH_TEMPORAL_STRIDE) + 1 ;
    println!("predicted_n_patches: {}", predicted_n_patches_per_item);
    // 2 sides by 2 channels (i.e. price and volume)
    let patch_size = LOB_LEVELS * 2 * 2 * PATCH_TEMPORAL_WINDOW_SIZE;
    let token_size = patch_size * predicted_n_patches_per_item;
    println!("token_size: {}", token_size);


    println!("Total number of snapshots: {}", snapshots.len());
    println!("Total number of items: {}", items.len());
    println!("Total number of patches per item: {}", items[0].patches.ask_price.len() /* * 4*/);


    // // Write CSV file
    // // let mut writer = csv::Writer::from_path("./shit.csv").expect("cannot open csv file");
    // let mut writer = csv::WriterBuilder::new()
    //     .has_headers(false)
    //     .from_path("./shit.csv").expect("cannot open csv file");
    //
    // #[derive(Debug, Serialize, Deserialize)]
    // struct Record {
    //     city: String,
    //     state: String,
    //     country: String,
    //     population: Vec<u64>
    // };
    //
    // for item in items {
    //     // writer.serialize(item.patches.ask_price).expect("cannot serialize item");
    //     writer.serialize(item).expect("cannot serialize item");
    //     // writer.serialize(Record {
    //     //     city: "New York".to_string(),
    //     //     state: "NY".to_string(),
    //     //     country: "USA".to_string(),
    //     //     population: vec![1,2,3],
    //     // }).expect("cannot serialize item");
    // }
    // // writer.serialize(&items).expect("cannot write csv file");
    //
    // writer.flush().expect("cannot flush csv file");
}


