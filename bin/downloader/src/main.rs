// use anyhow::anyhow;
use clap::Parser as ClapParser;
use std::{
    env,
    io::{ Read, stderr, stdin, stdout },
    path::PathBuf,
    process::exit,
    time::Duration,
};
use tracing_subscriber::{EnvFilter, fmt};
use tokio;
use std::error::Error;

use databento::{
    dbn::{decode::DbnMetadata, Dataset, SType, Schema, TradeMsg},
    historical::timeseries::GetRangeParams,
    HistoricalClient,
};
use time::macros::{date, datetime};

mod errors;


const USER_NAME: &str = "username";
const PASSWORD: &str = "password";
const DATABENTO_API_KEY: &str = "API_KEY";



#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    fmt()
        .without_time()
        .with_file(true)
        .with_level(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .pretty()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse the command line arguments
    let args = Args::parse();
    tracing::info!("Run with arguments: {args:#?}");

    // // Canonicalize all input files, to ensure that the files exists and that
    // // the path is valid. Store it in a vector for further processing.
    // let inputs = args
    //     .inputs
    //     .into_iter()
    //     .map(|p| p.canonicalize())
    //     .collect::<::errors::Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    // Canonicalize settings file
    let settings = args.settings.canonicalize().unwrap();
    println!("{:?}", settings);
    // let settings = SessionSettings::try_from_path(&settings).map_err(|e| anyhow!("{:?}", e))?;

    // Databento stuff
    let mut client = HistoricalClient::builder().key_from_env()?.build()?;
    // let mut decoder = client
    //     .timeseries()
    //     .get_range(
    //         &GetRangeParams::builder()
    //             .dataset(Dataset::GlbxMdp3)
    //             .date_time_range(datetime!(2022-06-10 14:30 UTC)..datetime!(2022-06-10 14:40 UTC))
    //             .symbols("ES.FUT")
    //             .stype_in(SType::Parent)
    //             .schema(Schema::Trades)
    //             .build(),
    //     )
    //     .await?;
    // let symbol_map = decoder
    //     .metadata()
    //     .symbol_map_for_date(date!(2022 - 06 - 10))?;
    // while let Some(trade) = decoder.decode_record::<TradeMsg>().await? {
    //     let symbol = &symbol_map[trade];
    //     println!("Received trade for {symbol}: {trade:?}");
    // }


    println!("Hello, world!");

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    /// Write additional debut output in the output directory.
    #[arg(short, long)]
    enable_debug_output: bool,

    /// Path to settings file
    #[arg(short, long)]
    settings: PathBuf,
    // /// Path to write the generated code to.
    // #[arg()]
    // output: PathBuf,
    //
    // /// Paths to read the schemas files from.
    // #[arg()]
    // inputs: Vec<PathBuf>,
}