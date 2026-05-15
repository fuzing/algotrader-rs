
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
use dotenv::dotenv;

use databento::{
    HistoricalClient,
    dbn::{Schema, },
    historical::timeseries::GetRangeToFileParams,
};

use utilities::date_time::str_to_offset_date_time;

//
// Datasets:
//   Nasdaq -> XNAS.ITCH
//   NYSE -> ARCX.PILLAR
//
//
async fn download_to_file(path: &PathBuf, dataset: &str, symbols: &Vec<String>, start_time: &str, end_time: &str) -> Result<(), Box<dyn Error>> {
    info!("Download to file");

    let start_t = str_to_offset_date_time(start_time)?;
    let end_t = str_to_offset_date_time(end_time)?;

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

    // Parse the command line arguments
    let args = Args::parse();
    let root_folder = env::var("ROOT_FOLDER").expect("no ROOT_FOLDER found in environment");
    let path: PathBuf = PathBuf::from(std::format!("{}/data/{}-{}-{}-{}-mbo.dbn.zst", root_folder, args.symbols.join(":"), args.dataset, args.start_time, args.end_time));

    download_to_file(&path, &args.dataset, &args.symbols, &args.start_time, &args.end_time).await?;

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,

    #[arg(long)]
    start_time: String,

    #[arg(short, long)]
    end_time: String,

    #[arg(short, long)]
    dataset: String,
}


