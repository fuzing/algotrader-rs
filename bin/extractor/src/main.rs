

use extractors::{
    extractor::Extractor,
    interval_extractor::{
        IntervalExtractor,
        IntervalExtraction,
        IntervalExtractionWithGain,
        ExtractedDataFileFormat,
    },
};

use serde::{Deserialize, Serialize};
use anyhow::anyhow;

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
use dotenv::dotenv;

use databento::{
    dbn::{
        MboMsg,
        decode::{AsyncDbnDecoder},
    },
};


async fn decode_data(path: &PathBuf, extractor: &mut impl Extractor<IntervalExtraction>, holding_time_intervals: usize) -> Result<Vec<IntervalExtractionWithGain>, Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    let mut all_results: Vec<IntervalExtraction> = Vec::new();

    println!("Holding for {} intervals", holding_time_intervals);

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        let results = extractor.push(mbo).await?;
        if !results.is_empty() {
            // println!("{:?}\n", results);
            all_results.append(&mut results.clone());
        }
    }

    let mut all_results_mapped: Vec<IntervalExtractionWithGain> = Vec::new();
    for (index, result) in all_results.iter().enumerate() {
        if let Some(future_result) = all_results.get(index + holding_time_intervals) {

            let mid_point_price = (result.bids.get(0).unwrap().price + result.asks.get(0).unwrap().price) / 2.0;
            let future_mid_point_price = (future_result.bids.get(0).unwrap().price + future_result.asks.get(0).unwrap().price) / 2.0;

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

    Ok(all_results_mapped)
}


// #[derive(Debug, Serialize, Deserialize)]
// struct ExtractedDataFileFormat {
//     holding_time_seconds: u16,
//     interval_nanos: u64,
//     data: Vec<IntervalExtractionWithGain>
// }

async fn write_data(
    pretty: bool,

    path: PathBuf,
    holding_time_seconds: u16,
    interval_nanos: u64,
    data: Vec<IntervalExtractionWithGain>,

    last_trade_price_mean: f64,
    last_trade_price_std_dev: f64,
    mid_point_price_mean: f64,
    mid_point_price_std_dev: f64,
) -> Result<(), Box<dyn Error>> {
    let out_data = ExtractedDataFileFormat {
        holding_time_seconds,
        interval_nanos,

        last_trade_price_mean,
        last_trade_price_std_dev,
        mid_point_price_mean,
        mid_point_price_std_dev,
        data,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    if pretty {
        serde_json::to_writer_pretty(writer, &out_data)?;
    }
    else {
        serde_json::to_writer(writer, &out_data)?;
    }
    Ok(())
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
        .inputs
        .into_iter()
        .map(|p| Path::new(&root_folder).join("data").join(p).canonicalize())
        .collect::<Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    println!("inputs: {:?}", inputs);

    // number of intervals that we're presuming holding for
    let holding_time_intervals: usize = (args.holding_time_seconds as u64 * 1_000_000_000 / args.extraction_interval_nanos) as usize;

    let mut all_data: Vec<IntervalExtractionWithGain> = Vec::new();

    for input in inputs {
        let mut extractor = IntervalExtractor::builder()
            .nbr_lob_levels(args.levels)
            .extraction_interval_nanos(args.extraction_interval_nanos)
            .build();

        let mut data = decode_data(&input, &mut extractor, holding_time_intervals).await?;
        all_data.append(&mut data);

        println!("Stats: {}", extractor.stats());
    }

    // calculate statistics
    let last_trade_price_mean = 0.0;
    let last_trade_price_std_dev = 0.0;
    let mid_point_price_mean = 0.0;
    let mid_point_price_std_dev = 0.0;

    write_data(args.pretty, args.output, args.holding_time_seconds, args.extraction_interval_nanos, all_data,
            last_trade_price_mean, last_trade_price_std_dev, mid_point_price_mean, mid_point_price_std_dev).await?;

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // nanoseconds between intervals
    #[arg(long)]
    extraction_interval_nanos: u64,

    // presumed holding time for the data
    #[arg(long)]
    holding_time_seconds: u16,

    // levels of each side of the order book to capture (e.g. 5, 10 etc.)
    #[arg(long)]
    levels: usize,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    pretty: bool,

    #[arg()]
    inputs: Vec<PathBuf>,
}

