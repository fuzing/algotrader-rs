
use order_book::{
    market::Market,
    date_time::to_offset_date_time,
};

use strategies::{
    strategy::Strategy,
    dummy_strategy::DummyStrategy,
    test_strategy::{TestStrategy, TestStrategyBuilder},
};

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
use tracing::{debug, info, warn, error, Instrument};
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
use time::{
    OffsetDateTime,
    format_description::well_known::{Rfc3339, Iso8601},
    macros::{date, datetime},
};

use chrono::{ DateTime, Utc};
use clap::builder::Str;
use databento::reference::Country::Is;

async fn build_from_snapshot() -> Result<Market, Box<dyn Error>> {
    let mut market = Market::default();

    Ok(market)
}

//
// Datasets:
//   Nasdaq -> XNAS.ITCH
//   NYSE -> ARCX.PILLAR
//
//
async fn download_to_file(path: &PathBuf, dataset: &str, symbols: &Vec<String>, start_time: &str, end_time: &str) -> Result<(), Box<dyn Error>> {
    info!("Download to file");

    let start_t = to_offset_date_time(start_time)?;
    let end_t = to_offset_date_time(end_time)?;

    println!("DTRange {:?}", start_t..end_t);

    if !fs::try_exists(path).await? {
        let mut client = HistoricalClient::builder().key_from_env()?.build()?;
        client
            .timeseries()
            .get_range_to_file(
                &GetRangeToFileParams::builder()
                    .dataset(dataset)
                    .symbols(symbols.to_owned())
                    .date_time_range(
                        // datetime!(2024-04-03 08:00:00 UTC)..datetime!(2024-04-03 14:00:00 UTC),
                        start_t..end_t,
                    )
                    .schema(Schema::Mbo)
                    .path(path)
                    .build(),
            )
            .await?;
    }

    Ok(())
}


async fn decode_data(path: &PathBuf, strategy: &mut impl Strategy) -> Result<(), Box<dyn Error>> {

    let mut market = Market::default();

    let mut decoder = AsyncDbnDecoder::from_zstd_file(path).await?;
    let symbol_map = decoder.metadata().symbol_map()?;

    while let Some(mbo) = decoder.decode_record::<MboMsg>().await? {

        strategy.pre_apply(mbo, &symbol_map, &market).await?;
        market.apply(mbo.clone());
        strategy.post_apply(mbo, &symbol_map, &market).await?;

        // If it's the last update in an event, print the state of the aggregated book
        if mbo.flags.is_last() {
            // let symbol = symbol_map.get_for_rec(mbo).unwrap();
            // let (best_bid, best_offer) = market.aggregated_bbo(mbo.hd.instrument_id);
            // println!("{symbol} Aggregated BBO | {}", mbo.ts_recv().unwrap());
            // if let Some(best_offer) = best_offer {
            //     println!("    Ask -> {best_offer}");
            // } else {
            //     println!("    Ask -> None");
            // }
            // if let Some(best_bid) = best_bid {
            //     println!("    Bid -> {best_bid}");
            // } else {
            //     println!("    Bid -> None");
            // }

            // println!("{}", market);
        }

    }

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
        .with_ansi(false)   // turns off display characters that change color etc.
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
    let path: PathBuf = PathBuf::from(std::format!("/run/media/peter/genetics/algotrader/data/{}-{}-{}-{}-mbo.dbn.zst", args.symbols.join(":"), args.dataset, args.start_time, args.end_time));
    download_to_file(&path, &args.dataset, &args.symbols, &args.start_time, &args.end_time).await?;

    let mut strategy = TestStrategyBuilder::default()
        .purchase_shares(100)
        .minimum_ask_shares_in_book(1_000)
        .maximum_holding_time(24 * 60 * 60)
        .bid_ask_volume_ratio(1.5)
        .desired_gain_percentage(0.35)
        .stop_loss_percentage(2.0)
        .build();

    println!("Strategy is {:?}", strategy);

    decode_data(&path, &mut strategy).await?;


    println!("Total Profit/Loss {:.2}", strategy.profit_loss());
    println!("Total Shares Traded for the Period {}", strategy.total_shares_traded());

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    // /// Write additional debut output in the output directory.
    // #[arg(short, long)]
    // enable_debug_output: bool,

    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,

    #[arg(long)]
    start_time: String,

    #[arg(short, long)]
    end_time: String,

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

