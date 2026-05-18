

use extractors::{
    extractor::Extractor,
    interval_extractor::{ IntervalExtractor, IntervalExtraction },
};

use anyhow::anyhow;

use clap::Parser as ClapParser;
use std::{
    env,
    io::{ Read, stderr, stdin, stdout },
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




// #[derive(Clone, Debug, Serialize, Deserialize)]
// pub struct IntervalExtraction {
//     date_time_nanos: u64,                     // nanos past unix epoch
//     last_trade_price: f64,
//     bids: Vec<PriceVolumeLevel>,
//     asks: Vec<PriceVolumeLevel>,
// }


#[derive(Debug)]
struct IntervalExtractionWithGain {
    extract: IntervalExtraction,
    gain: f64,
}
impl Display for IntervalExtractionWithGain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IntervalExtraction with gain (gain: {})", self.gain)?;
        Ok(())
    }
}


async fn decode_data(path: &PathBuf, extractor: &mut impl Extractor<IntervalExtraction>, holding_time_intervals: usize) -> Result<(), Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;

    let mut all_results: Vec<IntervalExtraction> = Vec::new();

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        let results = extractor.push(mbo).await?;
        if !results.is_empty() {
            println!("{:?}\n", results);
            all_results.append(&mut results.clone());
        }
    }

    let mut all_results_mapped: Vec<IntervalExtractionWithGain> = Vec::new();
    for (index, result) in all_results.iter().enumerate() {
        if let Some(future_result) = all_results.get(index + holding_time_intervals) {
            all_results_mapped.push(
                IntervalExtractionWithGain {
                    extract: result.clone(),
                    gain: ((future_result.last_trade_price / result.last_trade_price) - 1.0) * 100.0,
                }
            );
        }
    }

    println!("\n\nGot {} results.", all_results.len());

    for rm in all_results_mapped {
        println!("{}", rm);
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
    // info!("Run with arguments: {args:#?}");

    // Canonicalize all input files, to ensure that the files exists and that
    // the path is valid. Store it in a vector for further processing.
    let inputs = args
        .inputs
        .into_iter()
        .map(|p| Path::new(&root_folder).join("data").join(p).canonicalize())
        .collect::<Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    println!("inputs: {:?}", inputs);
    
    let mut extractor = IntervalExtractor::builder()
        .nbr_lob_levels(5)
        .extraction_interval_nanos(args.extraction_interval_nanos)
        .build();

    // println!("Extractor is {:?}", extractor);

    // number of intervals that we're presuming holding for
    let holding_time_intervals: usize = (args.holding_time_seconds * 1_000_000_000 / args.extraction_interval_nanos) as usize;

    decode_data(inputs.get(0).unwrap(), &mut extractor, holding_time_intervals).await?;
    // println!("Extractor is {:?}", extractor);

    println!("Stats: {}", extractor.stats());

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // /// Write additional debut output in the output directory.
    // #[arg(short, long)]
    // enable_debug_output: bool,

    #[arg(long, value_delimiter = ',')]
    symbol: String,

    #[arg(long)]
    start_date: String,

    #[arg(short, long)]
    end_date: String,

    #[arg(long)]
    extraction_interval_nanos: u64,

    #[arg(long)]
    holding_time_seconds: u64,

    // Path to settings file
    // #[arg(short, long)]
    // settings: PathBuf,
    // /// Path to write the generated code to.
    // #[arg()]
    // output: PathBuf,
    //
    // /// Paths to read the schemas files from.
    #[arg()]
    inputs: Vec<PathBuf>,
}

