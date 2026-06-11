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

        if (i % 10_000) == 0 {
            println!("Processed {} of {}", i, n_items);
        }
    }

    println!("Total samples: {}", reader.len());
    println!("Absolute => Gains: ({}).  Neutrals ({}).  Losses ({})", n_gains, n_neutrals, n_losses);
    println!("Percentage => Gains: ({:.3}%).  Neutrals ({:.3}%).  Losses ({:.3}%)",
             n_gains as f64 / n_items as f64 * 100.0,
             n_neutrals as f64 / n_items as f64 * 100.00,
             n_losses as f64 / n_items as f64 * 100.00
    );
    // balanced formula for weights
    assert_eq!(n_gains + n_neutrals + n_losses, n_items);
    let n_classes = 3;
    let bf_loss_weight = n_items as f64 / (n_classes as f64 * n_losses as f64);
    let bf_neutral_weight = n_items as f64 / (n_classes as f64 * n_neutrals as f64);
    let bf_gain_weight = n_items as f64 / (n_classes as f64 * n_gains as f64);
    println!("Inverse Frequency Weights Matrix: [{:.3},{:.3},{:.3}]", bf_loss_weight, bf_neutral_weight, bf_gain_weight);

    // another kind
    let if_loss_weight = n_items as f64 / n_losses as f64;
    let if_neutral_weight = n_items as f64 / n_neutrals as f64;
    let if_gain_weight = n_items as f64 / n_gains as f64;
    println!("Simplified Inverse Frequency Weights Matrix: [{:.3},{:.3},{:.3}]", if_loss_weight, if_neutral_weight, if_gain_weight);

    // log scaled inverse frequency
    let lsln_loss_weight = (1.0 + (n_items as f64 / n_losses as f64)).ln();
    let lsln_neutral_weight = (1.0 + (n_items as f64 / n_neutrals as f64)).ln();
    let lsln_gain_weight = (1.0 + (n_items as f64 / n_gains as f64)).ln();
    println!("Log Scaled Ln - Inverse Frequency Weights Matrix: [{:.3},{:.3},{:.3}]", lsln_loss_weight, lsln_neutral_weight, lsln_gain_weight);

    // log scaled inverse frequency
    let lslog10_loss_weight = 1.0 / (n_classes as f64 + n_losses as f64 / n_items as f64).log10();
    let lslog10_neutral_weight = 1.0 / (n_classes as f64 + n_neutrals as f64 / n_items as f64).log10();
    let lslog10_gain_weight = 1.0 / (n_classes as f64 + n_gains as f64 / n_items as f64).log10();
    println!("Log Scaled Log10 - Inverse Frequency Weights Matrix: [{:.3},{:.3},{:.3}]", lslog10_loss_weight, lslog10_neutral_weight, lslog10_gain_weight);

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

