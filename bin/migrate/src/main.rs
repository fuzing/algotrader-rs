// use anyhow::anyhow;
use clap::Parser as ClapParser;
use std::{
    env,
    io::{ Read, stderr, stdin, stdout },
    path::PathBuf,
    process::exit,
    time::Duration,
};
use sqlx::{
    Pool,
    Postgres,
    postgres::{PgPoolOptions},
    migrate
};
use tracing_subscriber::{EnvFilter, fmt};
use tracing::{info, warn, error};
use tokio;
use std::error::Error;
use dotenv::dotenv;



async fn migrate() -> Result<(), Box<dyn Error>>
{

    info!("migrating database");

    let connection_string = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(connection_string.as_str())
        .await?;

    sqlx::migrate!("../../migrations").run(&pool).await?;

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

    info!("Starting migrator");

    // Parse the command line arguments
    let args = Args::parse();
    // info!("Run with arguments: {args:#?}");

    migrate().await?;

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

