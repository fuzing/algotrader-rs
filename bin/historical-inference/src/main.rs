

use extractors::{
    extractor::Extractor,
    interval_extractor::{
        IntervalExtractor,
        IntervalExtraction,
        IntervalExtractionWithGain,
        ExtractedDataFile,
    },
};

use burn::{
    prelude::*,
    nn::{PositionalEncodingConfig, PositionalEncoding},
    tensor::{
        Device,
        DeviceConfig,
        Element,
        Tensor,
        TensorData,
        Shape
    },
};

use statrs::statistics::Statistics;

use serde::{Deserialize, Serialize};
use anyhow::anyhow;

use ai_models::price_gain::data::data_spec::{DataSpec, DataSpecBuilder};

use clap::Parser as ClapParser;
use std::{
    collections::VecDeque,
    env,
    io::{ BufWriter, Read, stderr, stdin, stdout },
    fs::File,
    path::{Path, PathBuf},
    process::exit,
    time::Duration,
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{debug, info, warn, error, Instrument};
use tokio;
use std::error::Error;
use std::fmt::Display;
use std::io::Write;
use dotenv::dotenv;

use databento::{
    dbn::{
        MboMsg,
        decode::{AsyncDbnDecoder},
    },
};

use utilities::date_time::{nanos_to_offset_date_time_with_tz, str_to_offset_date_time};


type Elem = f32;
// type Elem = burn::tensor::f16;


async fn decode_data(
    path: &PathBuf,
    extractor: &mut impl Extractor<IntervalExtraction>,
    spec: &DataSpec,
    holding_time_intervals: usize,
    start_date_nanos: u64,
    end_date_nanos: u64,
) -> Result<(), Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;

    // let mut all_results: Vec<IntervalExtraction> = Vec::new();

    let mut queue: VecDeque<IntervalExtraction> = VecDeque::new();

    println!("Holding for {} intervals", holding_time_intervals);

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        if mbo.ts_recv >= start_date_nanos && mbo.ts_recv <= end_date_nanos {
            let results = extractor.push(mbo).await?;
            if !results.is_empty() {
                // all_results.append(&mut results.clone());
                for result in results {
                    queue.push_back(result);
                }
                while queue.len() > spec.prediction_intervals {
                    queue.pop_front();
                }

                if queue.len() == spec.prediction_intervals {
                    println!("Predict");
                }


            }
        }
    }

    // let mut all_results_mapped: Vec<IntervalExtractionWithGain> = Vec::new();
    // for (index, result) in all_results.iter().enumerate() {
    //     if let Some(future_result) = all_results.get(index + holding_time_intervals) {
    //
    //         //
    //         // Make sure that future data point is from same day
    //         //
    //         let sample_day = nanos_to_offset_date_time_with_tz(result.date_time_nanos as i128, "ET").unwrap().weekday();
    //         let future_day = nanos_to_offset_date_time_with_tz(future_result.date_time_nanos as i128, "ET").unwrap().weekday();
    //
    //         if sample_day == future_day {
    //             // let mid_point_price = (result.bids.get(0).unwrap().price + result.asks.get(0).unwrap().price) / 2.0;
    //             // let future_mid_point_price = (future_result.bids.get(0).unwrap().price + future_result.asks.get(0).unwrap().price) / 2.0;
    //             let mid_point_price = (result.bids[0].price + result.asks[0].price) / 2.0;
    //             let future_mid_point_price = (future_result.bids[0].price + future_result.asks[0].price) / 2.0;
    //
    //             all_results_mapped.push(
    //                 IntervalExtractionWithGain {
    //                     date_time_nanos: result.date_time_nanos,
    //                     last_trade_price: result.last_trade_price,
    //                     future_trade_price: future_result.last_trade_price,
    //                     trade_gain: ((future_result.last_trade_price / result.last_trade_price) - 1.0) * 100.0,
    //
    //                     // depends upon ordering of bids/asks such as BBO must both be at '0' index
    //                     mid_point_price,
    //                     future_mid_point_price,
    //                     mid_point_gain: ((future_mid_point_price / mid_point_price) - 1.0) * 100.00,
    //
    //                     bids: result.bids.clone(),
    //                     asks: result.asks.clone(),
    //                 }
    //             );
    //         }
    //     }
    // }

    Ok(())
}


// struct PriceGainPatches {
//     pub bid_price: Vec<Vec<f64>>,
//     pub bid_volume: Vec<Vec<f64>>,
//     pub ask_price: Vec<Vec<f64>>,
//     pub ask_volume: Vec<Vec<f64>>,
// }
//
// async fn convert_and_write_data(
//     args: &Args,
//     stats: &DataStatistics,
//     data: Vec<IntervalExtractionWithGain>,
// ) -> Result<(), Box<dyn Error>> {
//     let mut csv_filename = File::create(&args.output_csv)?;
//
//     let prediction_temporal_window_size = args.prediction_intervals;
//     let patch_temporal_window_size = args.patch_intervals;
//     let patch_temporal_stride = args.patch_stride;
//     let lob_levels = args.lob_levels;
//
//     let price_mean = stats.price_mean;
//     let price_std_dev = stats.price_std_dev;
//     let volume_mean = stats.volume_mean;
//     let volume_std_dev = stats.volume_std_dev;
//
//     let predicted_patches_per_item = ((prediction_temporal_window_size - patch_temporal_window_size) / patch_temporal_stride) + 1;
//     let n_tokens = predicted_patches_per_item;
//     let patch_size = patch_temporal_window_size * lob_levels;
//     // the model dimension is the sum of the sizes:  ask_price_patch size + ask_volume_patch_size + bid_price_patch size + bid_volume_patch_size
//     println!("patch_size: ----------------------> {}", patch_size);
//     let d_model = patch_size * 4;
//     println!("d_model: ---------------------------> {}", d_model);
//
//     // write the spec file
//     let data_spec = DataSpecBuilder::new()
//         .sequence_length(predicted_patches_per_item)
//         .patch_size(patch_size)
//         .token_size(d_model)
//         .extraction_interval_nanos(args.extraction_interval_nanos)
//         .holding_time_seconds(args.holding_time_seconds)
//         .lob_levels(args.lob_levels)
//         .prediction_intervals(args.prediction_intervals)
//         .patch_intervals(args.patch_intervals)
//         .patch_stride(args.patch_stride)
//         .gain_percentage(args.gain_percentage)
//         .loss_percentage(args.loss_percentage)
//         .price_mean(price_mean)
//         .price_std_dev(price_std_dev)
//         .volume_mean(volume_mean)
//         .volume_std_dev(volume_std_dev)
//         .build();
//     data_spec.to_file(&args.output_spec)?;
//
//
//
//     // CPU based positional encoder
//     let mut device = Device::flex();
//     device
//         .configure(DeviceConfig::default().float_dtype(Elem::dtype()))
//         .unwrap();
//     let positional_encoder = PositionalEncodingConfig::new(d_model)
//         .with_max_sequence_size(n_tokens)
//         .with_max_timescale(1_000_000)
//         .init(&device);
//
//
//     //
//     // send it
//     //
//     for i in 0..=(data.len() - prediction_temporal_window_size) {
//         // new embeddable
//         let mut patches = PriceGainPatches {
//             bid_price: Vec::new(),
//             bid_volume: Vec::new(),
//             ask_price: Vec::new(),
//             ask_volume: Vec::new(),
//         };
//
//         for j in (0..=(prediction_temporal_window_size - patch_temporal_window_size)).step_by(patch_temporal_stride) {
//             let mut bid_price_patch: Vec<f64> = Vec::new();
//             let mut bid_volume_patch: Vec<f64> = Vec::new();
//             let mut ask_price_patch: Vec<f64> = Vec::new();
//             let mut ask_volume_patch: Vec<f64> = Vec::new();
//
//             for k in (0..patch_temporal_window_size) {
//                 for l in 0..lob_levels {
//                     bid_price_patch.push((data[i + j + k].bids[l].price - price_mean) / price_std_dev);
//                     bid_volume_patch.push((data[i + j + k].bids[l].volume as f64 - volume_mean) / volume_std_dev);
//                     ask_price_patch.push((data[i + j + k].asks[l].price - price_mean) / price_std_dev);
//                     ask_volume_patch.push((data[i + j + k].asks[l].volume as f64 - volume_mean) / volume_std_dev);
//                 }
//             }
//
//             assert_eq!(bid_price_patch.len(), patch_size);
//             assert_eq!(bid_volume_patch.len(), patch_size);
//             assert_eq!(ask_price_patch.len(), patch_size);
//             assert_eq!(ask_volume_patch.len(), patch_size);
//
//             // add patches
//             patches.bid_price.push(bid_price_patch);
//             patches.bid_volume.push(bid_volume_patch);
//             patches.ask_price.push(ask_price_patch);
//             patches.ask_volume.push(ask_volume_patch);
//         }
//
//         // our label is found in the snapshot at the end of the prediction temporal window
//         // let label = data[i + prediction_temporal_window - 1].trade_gain;
//         let gain = data[i + prediction_temporal_window_size - 1].mid_point_gain;
//         let label = if gain > args.gain_percentage {
//             2.0
//         } else if (gain > -args.loss_percentage) {
//             1.0
//         }
//         else { 0.0 };
//
//         assert_eq!(patches.bid_price.len(), predicted_patches_per_item);
//         assert_eq!(patches.bid_price.len(), n_tokens);
//
//         let mut tokens: Vec<Vec<f64>> = Vec::with_capacity(n_tokens);
//         for i in 0..n_tokens {
//             let token = [
//                 patches.bid_price[i].clone(),
//                 patches.bid_volume[i].clone(),
//                 patches.ask_price[i].clone(),
//                 patches.ask_volume[i].clone(),
//             ].concat();
//
//             // println!("token length {}", token.len());       // 160
//
//             assert_eq!(token.len(), d_model);
//             tokens.push(token);
//         }
//
//         // build a tensor of [batch_size, n_tokens, d_model] with batch size 1 and then add
//         // positional encodings
//         let flat = tokens.into_iter().flatten().collect::<Vec<_>>();
//         assert_eq!(flat.len(), 1 * n_tokens * d_model);
//
//         let tensor = Tensor::<3, Float>::from_floats(
//             TensorData::new(flat,Shape::new([1, n_tokens, d_model])),
//             &device
//         );
//
//         // add positional encodings and divide by 2.0 to normalize
//         let tensor_with_positions = positional_encoder.forward(tensor).div_scalar(2.0);
//         let vec_with_positions = tensor_with_positions.to_data().iter::<f64>().collect::<Vec<_>>();
//         assert_eq!(vec_with_positions.len(), 1 * n_tokens * d_model);
//         let mut final_vector = vec_with_positions;
//
//         // add label as last position for vector
//         final_vector.push(label);
//
//         // format the line as a comma separated list of floats and write to the file
//         let line = final_vector.into_iter().map(|v| format_float(v)).collect::<Vec<_>>();
//         writeln!(csv_filename, "{}", line.join(","))?;
//     }
//
//     Ok(())
// }



#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // get .env variables into environment
    dotenv().ok();
    let root_folder = env::var("ROOT_FOLDER").expect("no ROOT_FOLDER found in environment");

    // tracing format
    fmt()
        // .with_level(true)
        // .without_time()
        // .with_file(false)
        // .with_line_number(false)
        // .with_thread_ids(true)
        // .with_thread_names(true)
        // .pretty()
        .with_ansi(false)   // turns off display characters that change color etc.
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Building order book");

    // Parse the command line arguments
    let args = Args::parse();

    // Canonicalize all input files, to ensure that the files exists and that
    // the path is valid. Store it in a vector for further processing.
    let inputs = args
        .inputs.clone()
        .into_iter()
        .map(|p| Path::new(&root_folder).join("data").join(p).canonicalize())
        .collect::<Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    println!("inputs: {:?}", inputs);


    // read in the spec file
    let specs = DataSpec::from_file(&args.spec_file)?;



    // number of intervals that we're presuming holding for
    let holding_time_intervals: usize = (specs.holding_time_seconds as u64 * 1_000_000_000 / &specs.extraction_interval_nanos) as usize;

    let start_date_nanos = str_to_offset_date_time(&format!("{} 00:00:00 UTC", &args.start_date)).expect("Invalid start date").unix_timestamp_nanos() as u64;
    let end_date_nanos = str_to_offset_date_time(&format!("{} 23:59:59 UTC", &args.end_date)).expect("Invalid end date").unix_timestamp_nanos() as u64;

    // let mut all_data: Vec<IntervalExtractionWithGain> = Vec::new();

    for input in inputs {
        let mut extractor = IntervalExtractor::builder()
            .nbr_lob_levels(&specs.lob_levels)
            .extraction_interval_nanos(&specs.extraction_interval_nanos)
            .build();

        decode_data(
            &input,
            &mut extractor,
            &specs,
            holding_time_intervals,
            start_date_nanos,
            end_date_nanos,
        ).await?;
        // all_data.append(&mut data);

        println!("Stats: {}", extractor.stats());
    }

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // start/end dates to extract from/to
    #[arg(long)]
    start_date: String,

    #[arg(long)]
    end_date: String,

    #[arg(long)]
    spec_file: PathBuf,

    #[arg(long)]
    artifacts_folder: PathBuf,    
    
    #[arg()]
    inputs: Vec<PathBuf>,
}

