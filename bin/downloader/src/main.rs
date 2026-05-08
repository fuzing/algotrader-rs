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
use tracing::{info, warn, error};
use tokio;
use std::error::Error;
use dotenv::dotenv;

use databento::{
    dbn::{
        decode::DbnMetadata,
        Dataset,
        PitSymbolMap,
        SType,
        Schema,
        TradeMsg
    },
    live::Subscription,
    historical::timeseries::GetRangeParams,
    HistoricalClient,
    ReferenceClient,
    LiveClient,
};
use time::macros::{date, datetime};

mod errors;


// const USER_NAME: &str = "username";
// const PASSWORD: &str = "password";
// const DATABENTO_API_KEY: &str = "API_KEY";


async fn get_history() -> Result<(), Box<dyn Error>>
{
    info!("Downloading history");

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


    Ok(())
}


async fn get_live() -> Result<(), Box<dyn Error>>
{
    // Databento stuff
    let mut client = LiveClient::builder()
        .key_from_env()?
        .dataset(Dataset::GlbxMdp3)
        .build()
        .await?;
    client
        .subscribe(
            Subscription::builder()
                .symbols("ES.FUT")
                .schema(Schema::Trades)
                .stype_in(SType::Parent)
                .build(),
        )
        .await
        .unwrap();
    // client.start().await?;
    //
    // let mut symbol_map = PitSymbolMap::new();
    // // Get the next trade
    // while let Some(rec) = client.next_record().await? {
    //     if let Some(trade) = rec.get::<TradeMsg>() {
    //         let symbol = &symbol_map[trade];
    //         println!("Received trade for {symbol}: {trade:?}");
    //         break;
    //     }
    //     symbol_map.on_record(rec)?;
    // }

    Ok(())
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // get .env variables into environment
    dotenv().ok();

    // tracing format
    fmt()
        // .with_level(true)
        // .without_time()
        // .with_file(false)
        // .with_line_number(false)
        // .with_thread_ids(true)
        // .with_thread_names(true)
        // .pretty()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Starting downloader");

    // Parse the command line arguments
    let args = Args::parse();
    // info!("Run with arguments: {args:#?}");

    // // Canonicalize all input files, to ensure that the files exists and that
    // // the path is valid. Store it in a vector for further processing.
    // let inputs = args
    //     .inputs
    //     .into_iter()
    //     .map(|p| p.canonicalize())
    //     .collect::<::errors::Result<Vec<_>, _>>().map_err(|e| anyhow!(e))?;

    // Canonicalize settings file
    // let settings = args.settings.canonicalize().unwrap();
    // println!("{:?}", settings);
    // let settings = SessionSettings::try_from_path(&settings).map_err(|e| anyhow!("{:?}", e))?;

    get_history().await?;
    // get_live().await?;

    println!("Hello, world!");

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    /// Write additional debut output in the output directory.
    #[arg(short, long)]
    enable_debug_output: bool,

    // Path to settings file
    // #[arg(short, long)]
    // settings: PathBuf,
    // /// Path to write the generated code to.
    // #[arg()]
    // output: PathBuf,
    //
    // /// Paths to read the schemas files from.
    // #[arg()]
    // inputs: Vec<PathBuf>,
}

