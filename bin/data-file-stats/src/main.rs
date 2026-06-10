//
// Generate some statistics for a file
//

use clap::Parser as ClapParser;
use std::{
    error::Error,
    env,
};
use tokio;
use data_handlers::{
    data_handler::{DataWriter, DataReader},
    mpk::{MpkDataReader, MpkDataWriter, AccessType}
};

type StorageElem = f32;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>
{
    // Parse the command line arguments
    let args = Args::parse();
    let reader: MpkDataReader<StorageElem> = MpkDataReader::new(&args.input, AccessType::Sequential);

    let n_items = reader.len();
    println!("{} Total items", n_items);

    let mut n_gains = 0;
    let mut n_neutrals = 0;
    let mut n_losses = 0;

    for i in 0..n_items {
        let mut data = reader.read(i).unwrap();
        let gain = data.pop().unwrap();
        if gain >= args.gain_threshold {
            n_gains += 1;
        }
        else if gain > -args.loss_threshold {
            n_neutrals += 1;
        }
        else {
            n_losses += 1;
        }

        if (i % 1_000) == 0 {
            println!("Processed {} of {}", i, n_items);
        }
    }

    println!("Total samples: {}", reader.len());
    // balanced formula for weights
    assert_eq!(n_gains + n_neutrals + n_losses, n_items);
    let n_classes = 3;
    let bf_loss_weight = n_items as f64 / (n_classes as f64 * n_losses as f64);
    let bf_neutral_weight = n_items as f64 / (n_classes as f64 * n_neutrals as f64);
    let bf_gain_weight = n_items as f64 / (n_classes as f64 * n_gains as f64);
    println!("Balanced Formula Weights Matrix: [{:.3},{:.3},{:.3}]", bf_loss_weight, bf_neutral_weight, bf_gain_weight);

    let if_loss_weight = n_items as f64 / n_losses as f64;
    let if_neutral_weight = n_items as f64 / n_neutrals as f64;
    let if_gain_weight = n_items as f64 / n_gains as f64;
    println!("Inverse Frequency Weights Matrix: [{:.3},{:.3},{:.3}]", if_loss_weight, if_neutral_weight, if_gain_weight);

    Ok(())
}


#[derive(Debug, ClapParser)]
struct Args {
    #[arg(long)]
    gain_threshold: f32,

    #[arg(long)]
    loss_threshold: f32,

    #[arg(long)]
    input: String,
}

