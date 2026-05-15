
use order_book::{
    market::Market,
    date_time::to_offset_date_time,
};

use strategies::strategy::{Strategy};

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
use tracing::{info, warn, error, Instrument};
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
        ErrorMsg,
        MboMsg,
        Publisher,
        Record,
        Schema,
        SType,
        Side,
        SymbolIndex,
        SymbolMappingMsg,
        TsSymbolMap,
        UNDEF_PRICE,
        decode::{AsyncDbnDecoder, DbnMetadata},
        pretty,

    },
    historical::timeseries::GetRangeToFileParams,
    live::Subscription,
};
use time::{
    OffsetDateTime,
    format_description::well_known::{Rfc3339, Iso8601},
    macros::{date, datetime},
};

use chrono::{ DateTime, Utc};
use databento::reference::Country::Is;
use strategies::dummy_strategy::DummyStrategy;


async fn decode_data(dataset: &str, symbols: &Vec<String>, strategy: &mut impl Strategy) -> Result<(), Box<dyn Error>> {
// async fn decode_data(symbols: &Vec<String>) -> Result<(), Box<dyn Error>> {

    // turn Vec<String> into Vec<&str>
    let symbols_str = symbols.iter().map(|s| s.as_str()).collect::<Vec<&str>>();

    let mut market = Market::default();

    // let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    // let symbol_map = decoder.metadata().symbol_map()?;


    // First, create a live client and connect
    let mut client = LiveClient::builder()
        .key_from_env()?
        .dataset(dataset)
        .build()
        .await?;

    // Next, we will subscribe to the MBO snapshot and start the session
    client
        .subscribe(
            Subscription::builder()
                .schema(Schema::Mbo)
                .symbols(symbols_str)
                .stype_in(SType::Continuous)
                .use_snapshot(true)
                .build(),
        )
        .await?;
    client.start().await?;

    // historical records
    // while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {
    //     market.apply(mbo.clone());
    //     // If it's the last update in an event, print the state of the aggregated book
    //     if mbo.flags.is_last() {
    //         let symbol = symbol_map.get_for_rec(mbo).unwrap();
    //         let (best_bid, best_offer) = market.aggregated_bbo(mbo.hd.instrument_id);
    //         println!("{symbol} Aggregated BBO | {}", mbo.ts_recv().unwrap());
    //         if let Some(best_offer) = best_offer {
    //             println!("    {best_offer}");
    //         } else {
    //             println!("    None");
    //         }
    //         if let Some(best_bid) = best_bid {
    //             println!("    {best_bid}");
    //         } else {
    //             println!("    None");
    //         }
    //
    //         println!("{}", market);
    //     }
    // }


    // start with an empty symbol map
    // https://github.com/databento/databento-rs/blob/main/src/historical/symbology.rs
    let mut symbol_map = TsSymbolMap::new();


    // Then, we will process all snapshot records, and stop at the first record
    // with F_LAST flag, which indicates that the snapshot is complete and the
    // order book is in a valid state
    while let Some(record) = client.next_record().await? {
        if let Some(mbo) = record.get::<MboMsg>() {

            strategy.pre_apply(mbo, &symbol_map, &market).await?;
            market.apply(mbo.clone());
            strategy.post_apply(mbo, &symbol_map, &market).await?;

            if mbo.flags.is_snapshot() {
                println!("Snapshot: {mbo:?}");
            } else {
                println!("Live: {mbo:?}");
            }
            if mbo.flags.is_last() {
                println!("Snapshot is complete");
                // break;
            }
        } else if let Some(symbol) = record.get::<SymbolMappingMsg>() {
            info!("Symbol mapping: {symbol:?}");
            // insert the symbol into the symbol table
            // symbol_map.insert(
            //     symbol.instrument(),
            // ).await?;
        } else if let Some(error) = record.get::<ErrorMsg>() {
            eprintln!("{}", error.err()?);
            break;
        }
    }


    client.close().await?;

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
    let mut strategy = DummyStrategy::new();
    decode_data(&args.dataset, &args.symbols, &mut strategy).await?;
    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // /// Write additional debut output in the output directory.
    // #[arg(short, long)]
    // enable_debug_output: bool,

    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,

    #[arg(short, long)]
    dataset: String,


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

