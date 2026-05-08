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



const USER_NAME: &str = "username";
const PASSWORD: &str = "password";
const DATABENTO_API_KEY: &str = "API_KEY";



#[tokio::main]
async fn main()
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



    println!("Hello, world!");
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