//
// Takes a .csv file and generates an index file with start and length of each line
//

use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use clap::Parser as ClapParser;
use std::{
    error::Error,
    env,
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{debug, info, warn, error, Instrument};
use tokio;
use dotenv::dotenv;
use data_handlers::{
    data_handler::{DataWriter, DataReader},
    mpk::{MpkDataReader, MpkDataWriter, AccessType}
};


type StorageElem = f32;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // get .env variables into environment
    dotenv().ok();
    // let root_folder = env::var("ROOT_FOLDER").expect("no ROOT_FOLDER found in environment");

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

    println!("Shuffling {}", args.input);

    let mut rng = StdRng::seed_from_u64(args.seed);

    let reader: MpkDataReader<StorageElem> = MpkDataReader::new(&args.input, AccessType::Random);
    let mut writer = MpkDataWriter::<StorageElem>::new(&args.output);

    let n_items = reader.len();
    println!("Shuffling {} items", n_items);

    // vector of indices (initially in order)
    let mut remaining: Vec<usize> = (0..n_items).collect();

    // shuffle the indices
    remaining.shuffle(&mut rng);

    // read randomly and write to the new index file
    while remaining.len() > 0 {
        let index =remaining.pop().unwrap();
        let data = reader.read(index)?;
        writer.write(&data)?;
        if (remaining.len() % 1_000) == 0 {
            println!("Shuffle remaining Items: {}", remaining.len());
        }
    }

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    #[arg(long)]
    seed: u64,

    #[arg(long)]
    input: String,

    #[arg(long)]
    output: String,
}

