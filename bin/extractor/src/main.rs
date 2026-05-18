

use extractors::{
    extractor::Extractor,
    interval_extractor::{ IntervalExtractor, IntervalExtractorBuilder, IntervalExtraction},
};

use anyhow::anyhow;

// use anyhow::anyhow;
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
use dotenv::dotenv;

use databento::{
    dbn::{
        MboMsg,
        decode::{AsyncDbnDecoder, DbnMetadata},
    },
};
use time::{
    macros::{date, datetime},
};


async fn decode_data(path: &PathBuf, extractor: &mut impl Extractor<IntervalExtraction>) -> Result<(), Box<dyn Error>> {
    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        let results = extractor.push(mbo).await?;
        if !results.is_empty() {
            println!("{:?}\n", results)
        }
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
        .nbr_lob_levels(10)
        .extraction_interval_nanos(1_000_000_000)
        .build();

    println!("Extractor is {:?}", extractor);
    decode_data(inputs.get(0).unwrap(), &mut extractor).await?;
    println!("Extractor is {:?}", extractor);

    println!("Stats: {:?}", extractor.stats());

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

