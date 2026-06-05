//
// Takes a .csv file and generates an index file with start and length of each line
//


use serde::{Deserialize, Serialize};
use anyhow::anyhow;


use clap::Parser as ClapParser;
use std::{
    error::Error,
    env,
    io::{
        BufRead,
        BufReader, BufWriter, Read, Write, stderr, stdin, stdout },
    fs::File,
    path::{Path, PathBuf},
    process::exit,
    time::Duration,
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{debug, info, warn, error, Instrument};
use tokio;
use dotenv::dotenv;


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


    // Parse the command line arguments
    let args = Args::parse();

    println!("Generating index for {}", args.input.clone().to_string_lossy());

    //
    // now process data
    //
    let file = File::open(args.input.clone()).expect(&format!("Couldn't open data file {:?}", args.input));

    // let reader = BufReader::new(&file);
    let reader = BufReader::with_capacity(64 * 1_024, &file);

    // let mut index: Vec<(usize, usize)> = Vec::new();
    let mut index: Vec<usize> = Vec::new();
    let mut cursor = 0;

    println!("Dataset - mapping line starting positions");

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.expect("Error reading csv file during indexing");
        let len = line.bytes().len();

        // no need to skip because we have no header
        // // skip header - first row
        // if (i == 0) {
        //     cursor += len + 1;      // +1 for newline
        //     continue;
        // }

        if (i % 1_000) == 0 {
            println!("Processed {} lines", i);
        }
        if i == 1_000 {
            break;
        }

        index.push(cursor);
        // index.push((cursor, len));
        cursor += len + 1;      // +1 for newline
    }
    println!("Indexed {} lines during mapping", index.len());

    let filename = format!("{}.json", &args.output.to_string_lossy());
    let file = File::create(&filename).expect(&format!("Couldn't open output file {:?}", &args.output));
    let writer = BufWriter::with_capacity(64 * 1_024, &file);
    // let writer = BufWriter::new(&file);
    serde_json::to_writer_pretty(writer, &index)?;

    let filename = format!("{}.mpk", &args.output.to_string_lossy());
    let file = File::create(&filename).expect(&format!("Couldn't open output file {:?}", &filename));
    let mut writer = BufWriter::with_capacity(64 * 1_024, &file);
    // let mut writer = BufWriter::new(&file);
    // let msgpack_bytes = rmp_serde::to_vec(&index)?;
    rmp_serde::encode::write(&mut writer, &index)?;
    writer.flush()?;
    file.sync_all()?;


    // println!("Sleeping");
    // tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;


    println!("Reading messagepack");

    let filename = format!("{}.mpk", &args.output.to_string_lossy());
    println!("filename {}", &filename);
    let file = File::open(&filename).expect(&format!("Couldn't open data file {:?}", &filename));
    // let reader = BufReader::with_capacity(10 * 1_024 * 1_024, &file);
    let reader = BufReader::new(&file);
    // let values: Vec<(usize, usize)> = rmp_serde::decode::from_read(reader)?;
    let values: Vec<usize> = rmp_serde::decode::from_read(reader)?;
    // for (start, len) in values.iter() {
    //     println!("({start}, {len})");
    // }
    for start in values.iter() {
        println!("({start})");
    }
    assert_eq!(index, values);

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

}

