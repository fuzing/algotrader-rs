

use extractors::{
    extractor::Extractor,
    interval_extractor::{
        IntervalExtractor,
        IntervalExtraction,
        IntervalExtractionWithGain,
        ExtractedDataFile,
    },
};

use statrs::statistics::Statistics;

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
use std::io::Write;
use dotenv::dotenv;

use databento::{
    dbn::{
        MboMsg,
        decode::{AsyncDbnDecoder},
    },
};

use utilities::date_time::{nanos_to_offset_date_time_with_tz, str_to_offset_date_time};


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

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        if mbo.ts_recv >= start_date_nanos && mbo.ts_recv <= end_date_nanos {
            let results = extractor.push(mbo).await?;
            if !results.is_empty() {
                // println!("{:?}\n", results);
                all_results.append(&mut results.clone());
            }
        }
    }

    let mut all_results_mapped: Vec<IntervalExtractionWithGain> = Vec::new();
    for (index, result) in all_results.iter().enumerate() {
        if let Some(future_result) = all_results.get(index + holding_time_intervals) {

            //
            // Make sure that future data point is from same day
            //
            let sample_day = nanos_to_offset_date_time_with_tz(result.date_time_nanos as i128, "ET").unwrap().weekday();
            let future_day = nanos_to_offset_date_time_with_tz(future_result.date_time_nanos as i128, "ET").unwrap().weekday();

            if sample_day == future_day {

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
    }

    Ok(all_results_mapped)
}

fn format_float(val: f64) -> String {
    // format!("{:.6}", val)
    format!("{}", val)
        // .trim_end_matches('0')
        .to_string()
}

async fn convert_and_write_data(
    args: &Args,
    stats: &DataStatistics,
    data: Vec<IntervalExtractionWithGain>,
) -> Result<(), Box<dyn Error>> {
    // let out_data = ExtractedDataFile {
    //     holding_time_seconds: args.holding_time_seconds,
    //     interval_nanos: args.extraction_interval_nanos,
    //
    //     price_mean: stats.price_mean,
    //     price_std_dev: stats.price_std_dev,
    //
    //     volume_mean: stats.volume_mean,
    //     volume_std_dev: stats.volume_std_dev,
    //     data,
    // };

    let mut file = File::create(&args.output)?;
    // let mut writer = BufWriter::new(file);
    //
    // writer.write_fmt()?;

    writeln!(file, "{}", format_float(10.0))?;






    Ok(())
}



async fn write_data(
    args: &Args,
    stats: &DataStatistics,
    data: Vec<IntervalExtractionWithGain>,
) -> Result<(), Box<dyn Error>> {
    let out_data = ExtractedDataFile {
        holding_time_seconds: args.holding_time_seconds,
        interval_nanos: args.extraction_interval_nanos,

        price_mean: stats.price_mean,
        price_std_dev: stats.price_std_dev,

        volume_mean: stats.volume_mean,
        volume_std_dev: stats.volume_std_dev,
        data,
    };

    let file = File::create(&args.output)?;
    let writer = BufWriter::new(file);
    if args.pretty {
        serde_json::to_writer_pretty(writer, &out_data)?;
    }
    else {
        serde_json::to_writer(writer, &out_data)?;
    }
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
    // nanoseconds between intervals
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

    // Governs how the prediction is classified.
    //   If future price >= current_price + gain_percentage then "buy"
    //   If future price is in the band gain_percentage to loss_percentage then "neutral" (i.e. don't buy)
    //   If future price is <= loss_percentage then "sell" (i.e. don't buy)
    //  Use values such as 0.1 (0.1%), meaning gain of 0.1% at the end of the holding_time
    //
    #[arg(long)]
    gain_percentage: f64,

    #[arg(long)]
    loss_percentage: f64,

    // start/end dates to extract from/to
    #[arg(long)]
    start_date: String,

    #[arg(long)]
    end_date: String,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    pretty: bool,

    #[arg()]
    inputs: Vec<PathBuf>,
}

