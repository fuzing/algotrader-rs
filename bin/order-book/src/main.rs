mod market;
mod book;
mod price_level;
mod level;

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
use tokio::{self, fs};
use std::error::Error;
use std::num::NonZeroU64;
use dotenv::dotenv;

use databento::{
    HistoricalClient,
    ReferenceClient,
    LiveClient,
    dbn::{
        Action,
        BidAskPair,
        Dataset,
        MboMsg,
        Publisher,
        Record,
        Schema,
        Side,
        SymbolIndex,
        UNDEF_PRICE,
        decode::{AsyncDbnDecoder, DbnMetadata},
        pretty,
    },
    historical::timeseries::GetRangeToFileParams,
};
use time::macros::{date, datetime};



async fn download_to_file() -> Result<(), Box<dyn Error>> {

    let path = "mbo.dbn.zst";

    if (!fs::try_exists(path).await?) {
        let mut client = HistoricalClient::builder().key_from_env()?.build()?;
        client
            .timeseries()
            .get_range_to_file(
                &GetRangeToFileParams::builder()
                    .dataset(Dataset::DbeqBasic)
                    .symbols(vec!["GOOG", "GOOGL"])
                    .date_time_range(
                        datetime!(2024-04-03 08:00:00 UTC)..datetime!(2024-04-03 14:00:00 UTC),
                    )
                    .schema(Schema::Mbo)
                    .path(path)
                    .build(),
            )
            .await?;
    }

    Ok(())
}


async fn decode_data() -> Result<(), Box<dyn Error>> {
    let path = "mbo.dbn.zst";

    let mut market = Market::default();

    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    let symbol_map = decoder.metadata().symbol_map()?;

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
        market.apply(mbo.clone());
        // If it's the last update in an event, print the state of the aggregated book
        if mbo.flags.is_last() {
            let symbol = symbol_map.get_for_rec(mbo).unwrap();
            let (best_bid, best_offer) = market.aggregated_bbo(mbo.hd.instrument_id);
            println!("{symbol} Aggregated BBO | {}", mbo.ts_recv().unwrap());
            if let Some(best_offer) = best_offer {
                println!("    {best_offer}");
            } else {
                println!("    None");
            }
            if let Some(best_bid) = best_bid {
                println!("    {best_bid}");
            } else {
                println!("    None");
            }
        }
    }

    Ok(())
}



async fn build_order_book() -> Result<(), Box<dyn Error>>
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
    //             // .symbols(Symbols::All)
    //             .stype_in(SType::Parent)
    //             // .limit(NonZeroU64::new(100).unwrap())
    //             .schema(Schema::Trades)Starting
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

    info!("Building order book");

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

    build_order_book().await?;
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

