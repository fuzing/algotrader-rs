

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

use ai_models::{
    lob_trans::data::{
        data::{LobTransPatchType, LobTransPatchSide},
        data_spec::{LobTransDataSpec, LobTransDataSpecBuilder}
    }
};

use clap::Parser as ClapParser;
use std::{
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

use data_handlers::{
    mpk::MpkDataWriter,
    data_handler::DataWriter,
};


type Elem = f32;
// type Elem = burn::tensor::f16;


// how elements are to be stored when extracted
type StorageElem = f32;


async fn decode_data(
    path: &PathBuf,
    extractor: &mut impl Extractor<IntervalExtraction>,
    holding_time_intervals: usize,
    start_date_nanos: u64,
    end_date_nanos: u64,
) -> Result<Vec<IntervalExtractionWithGain>, Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    let mut all_results: Vec<IntervalExtraction> = Vec::new();

    println!("Holding for {} intervals", holding_time_intervals);

    println!("Reading MBO data");
    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        if mbo.ts_recv >= start_date_nanos && mbo.ts_recv <= end_date_nanos {
            let results = extractor.push(mbo).await?;
            if !results.is_empty() {
                all_results.append(&mut results.clone());
            }
        }
    }

    println!("Generating Samples");
    let mut all_results_mapped: Vec<IntervalExtractionWithGain> = Vec::new();
    for (index, result) in all_results.iter().enumerate() {
        if let Some(future_result) = all_results.get(index + holding_time_intervals) {

            //
            // Make sure that future data point is from same day
            //
            let sample_day = nanos_to_offset_date_time_with_tz(result.date_time_nanos as i128, "ET")?.weekday();
            let future_day = nanos_to_offset_date_time_with_tz(future_result.date_time_nanos as i128, "ET")?.weekday();

            if sample_day == future_day {
                // let mid_point_price = (result.bids.get(0).unwrap().price + result.asks.get(0).unwrap().price) / 2.0;
                // let future_mid_point_price = (future_result.bids.get(0).unwrap().price + future_result.asks.get(0).unwrap().price) / 2.0;
                let mid_point_price = (result.bids[0].price + result.asks[0].price) / 2.0;
                let future_mid_point_price = (future_result.bids[0].price + future_result.asks[0].price) / 2.0;

                all_results_mapped.push(
                    IntervalExtractionWithGain {
                        date_time_nanos: result.date_time_nanos,
                        last_trade_price: result.last_trade_price,
                        future_trade_price: future_result.last_trade_price,
                        trade_gain: ((future_result.last_trade_price / result.last_trade_price) - 1.0) * 100.0,

                        // depends upon ordering of bids/asks such as BBO must both be at '0' index
                        mid_point_price,
                        future_mid_point_price,
                        mid_point_gain: ((future_mid_point_price / mid_point_price) - 1.0) * 100.00,

                        bids: result.bids.clone(),
                        asks: result.asks.clone(),
                    }
                );
            }
        }
    }

    Ok(all_results_mapped)
}

async fn convert_and_write_data(
    args: &Args,
    stats: &DataStatistics,
    data: Vec<IntervalExtractionWithGain>,
) -> Result<(), Box<dyn Error>> {
    println!("Convert and write data");

    let mut writer = MpkDataWriter::<StorageElem>::new(&args.output_data.to_string_lossy());

    let prediction_temporal_window_size = args.prediction_intervals;
    let patch_temporal_window_size = args.patch_intervals;
    let patch_temporal_stride = args.patch_stride;
    let lob_levels = args.lob_levels;

    let price_mean = stats.price_mean;
    let price_std_dev = stats.price_std_dev;
    let volume_mean = stats.volume_mean;
    let volume_std_dev = stats.volume_std_dev;

    let predicted_patches_per_item = ((prediction_temporal_window_size - patch_temporal_window_size) / patch_temporal_stride) + 1;
    let n_tokens = predicted_patches_per_item;

    let patch_size = patch_temporal_window_size * lob_levels + 2;

    // the model dimension is the sum of the sizes:  ask_price_patch size + ask_volume_patch_size + bid_price_patch size + bid_volume_patch_size
    println!("patch_size: ----------------------> {}", patch_size);
    let d_model = patch_size * 4;
    println!("d_model: ---------------------------> {}", d_model);

    let mut n_gains: usize = 0;
    let mut n_neutrals: usize = 0;
    let mut n_losses: usize = 0;

    //
    // send it
    //
    for i in 0..=(data.len() - prediction_temporal_window_size) {

        let mut bid_price_patches: Vec<Vec<StorageElem>> = Vec::new();
        let mut bid_volume_patches: Vec<Vec<StorageElem>> = Vec::new();
        let mut ask_price_patches: Vec<Vec<StorageElem>> = Vec::new();
        let mut ask_volume_patches: Vec<Vec<StorageElem>> = Vec::new();


        for j in (0..=(prediction_temporal_window_size - patch_temporal_window_size)).step_by(patch_temporal_stride) {
            // // create each patch - starting with each patch header value pair
            let mut bid_price_patch: Vec<StorageElem> = Vec::with_capacity(patch_size);
            bid_price_patch.extend(vec![LobTransPatchType::Price.value() as StorageElem, LobTransPatchSide::Bid.value() as StorageElem]);
            let mut bid_volume_patch: Vec<StorageElem> = Vec::with_capacity(patch_size);
            bid_volume_patch.extend(vec![LobTransPatchType::Volume.value() as StorageElem, LobTransPatchSide::Bid.value() as StorageElem]);
            let mut ask_price_patch: Vec<StorageElem> = Vec::with_capacity(patch_size);
            ask_price_patch.extend(vec![LobTransPatchType::Price.value() as StorageElem, LobTransPatchSide::Ask.value() as StorageElem]);
            let mut ask_volume_patch: Vec<StorageElem> = Vec::with_capacity(patch_size);
            ask_volume_patch.extend(vec![LobTransPatchType::Volume.value() as StorageElem, LobTransPatchSide::Ask.value() as StorageElem]);

            for k in 0..patch_temporal_window_size {
                for l in 0..lob_levels {
                    bid_price_patch.push(((data[i + j + k].bids[l].price - price_mean) / price_std_dev) as StorageElem);
                    bid_volume_patch.push(((data[i + j + k].bids[l].volume as f64 - volume_mean) / volume_std_dev) as StorageElem);
                    ask_price_patch.push(((data[i + j + k].asks[l].price - price_mean) / price_std_dev) as StorageElem);
                    ask_volume_patch.push(((data[i + j + k].asks[l].volume as f64 - volume_mean) / volume_std_dev) as StorageElem);
                }
            }

            assert_eq!(bid_price_patch.len(), patch_size);
            assert_eq!(bid_volume_patch.len(), patch_size);
            assert_eq!(ask_price_patch.len(), patch_size);
            assert_eq!(ask_volume_patch.len(), patch_size);

            // add patches
            bid_price_patches.push(bid_price_patch);
            bid_volume_patches.push(bid_volume_patch);
            ask_price_patches.push(ask_price_patch);
            ask_volume_patches.push(ask_volume_patch);
        }

        // our label is found in the snapshot at the end of the prediction temporal window
        // let label = data[i + prediction_temporal_window - 1].trade_gain;
        let gain = data[i + prediction_temporal_window_size - 1].mid_point_gain;

        assert_eq!(bid_price_patches.len(), predicted_patches_per_item);
        assert_eq!(bid_price_patches.len(), n_tokens);

        let mut tokens: Vec<Vec<StorageElem>> = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let token = [
                bid_price_patches[i].clone(),
                bid_volume_patches[i].clone(),
                ask_price_patches[i].clone(),
                ask_volume_patches[i].clone(),
            ].concat();

            assert_eq!(token.len(), d_model);
            tokens.push(token);
        }

        // build a tensor of [batch_size, n_tokens, d_model] with batch size 1 and then add
        // positional encodings
        let mut final_vector = tokens.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(final_vector.len(), 1 * n_tokens * d_model);

        // add gain as last position for vector
        final_vector.push(gain as StorageElem);

        // write to the data file
        writer.write(&final_vector)?;

        if (i % 10_000) == 0 {
            println!("Written {} of {} samples", i, data.len());
        }
    }

    println!("Gains({n_gains}), Neutrals({n_neutrals}), Losses({n_losses})");

    // write the spec file
    let data_spec = LobTransDataSpecBuilder::new()
        .sequence_length(predicted_patches_per_item)
        .patch_size(patch_size)
        .token_size(d_model)
        .extraction_interval_nanos(args.extraction_interval_nanos)
        .holding_time_seconds(args.holding_time_seconds)
        .lob_levels(args.lob_levels)
        .prediction_intervals(args.prediction_intervals)
        .patch_intervals(args.patch_intervals)
        .patch_stride(args.patch_stride)
        .price_mean(price_mean)
        .price_std_dev(price_std_dev)
        .volume_mean(volume_mean)
        .volume_std_dev(volume_std_dev)
        .start_date(&args.start_date)
        .end_date(&args.end_date)
        .build();
    data_spec.to_file(&args.output_spec)?;

    Ok(())
}




#[derive(Debug)]
pub struct DataStatistics {
    price_mean: f64,
    price_std_dev: f64,
    volume_mean: f64,
    volume_std_dev: f64,
}


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

    // number of intervals that we're presuming holding for
    let holding_time_intervals: usize = (args.holding_time_seconds as u64 * 1_000_000_000 / &args.extraction_interval_nanos) as usize;

    let start_date_nanos = str_to_offset_date_time(&format!("{} 00:00:00 UTC", &args.start_date)).expect("Invalid start date").unix_timestamp_nanos() as u64;
    let end_date_nanos = str_to_offset_date_time(&format!("{} 23:59:59 UTC", &args.end_date)).expect("Invalid end date").unix_timestamp_nanos() as u64;

    let mut all_data: Vec<IntervalExtractionWithGain> = Vec::new();

    for input in inputs {
        let mut extractor = IntervalExtractor::builder()
            .nbr_lob_levels(&args.lob_levels)
            .extraction_interval_nanos(&args.extraction_interval_nanos)
            .build();

        let mut data = decode_data(
            &input,
            &mut extractor,
            holding_time_intervals,
            start_date_nanos,
            end_date_nanos,
        ).await?;
        all_data.append(&mut data);

        println!("Stats: {}", extractor.stats());
    }

    // calculate statistics for z-score manipulation
    let _last_trade_price_mean = all_data.iter().map(|i| i.last_trade_price).mean();
    let _last_trade_price_std_dev = all_data.iter().map(|i| i.last_trade_price).std_dev();
    let mid_point_price_mean = all_data.iter().map(|i| i.mid_point_price).mean();
    let mid_point_price_std_dev = all_data.iter().map(|i| i.mid_point_price).std_dev();

    // All volumes mean and std_dev
    let mut all_volumes: Vec::<f64> = Vec::new();
    for d in all_data.iter() {
        for bid in d.bids.iter() {
            all_volumes.push(bid.volume as f64)
        }
        for ask in d.asks.iter() {
            all_volumes.push(ask.volume as f64)
        }
    }
    let volume_mean = all_volumes.iter().mean();
    let volume_std_dev = all_volumes.iter().std_dev();

    let stats = DataStatistics {
        price_mean: mid_point_price_mean,
        price_std_dev: mid_point_price_std_dev,
        volume_mean,
        volume_std_dev
    };


    convert_and_write_data(&args, &stats, all_data).await?;

    // write_data(&args, &stats, all_data).await?;


    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // nanoseconds between intervals - to discretize the snapshots
    #[arg(long)]
    extraction_interval_nanos: u64,

    // presumed holding time for the data - offset used to compute gain/loss
    #[arg(long)]
    holding_time_seconds: u16,

    // levels of each side of the order book to capture (e.g. 5, 10 etc.)
    #[arg(long)]
    lob_levels: usize,

    // number of intervals used for each prediction
    #[arg(long)]
    prediction_intervals: usize,

    // number of intervals to include per patch
    #[arg(long)]
    patch_intervals: usize,

    // number of intervals to include per patch
    #[arg(long)]
    patch_stride: usize,

    // start/end dates to extract from/to
    #[arg(long)]
    start_date: String,

    #[arg(long)]
    end_date: String,

    #[arg(long)]
    output_data: PathBuf,

    #[arg(long)]
    output_spec: PathBuf,

    #[arg(long)]
    pretty: bool,

    #[arg()]
    inputs: Vec<PathBuf>,
}

