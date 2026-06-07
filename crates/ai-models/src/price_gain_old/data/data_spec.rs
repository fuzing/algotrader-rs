//
// specifies the details of the csv data
//

use std::{
    fs::File,
    io::BufWriter,
    path::PathBuf,
};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataSpec {
    pub sequence_length: usize,         // number of items in temporal sequence
    pub patch_size: usize,              // number of values in a patch (lob_depth * patch_temporal_window)
    pub token_size: usize,              // in our use case it's 4 * patch_size (bid_price, bid_volume, ask_price, ask_volume)

    pub extraction_interval_nanos: u64, // to discretize snapshots, how many nanons between emitted samples
    pub holding_time_seconds: u16,      // number of seconds to hold the stock prior to selling
    pub lob_levels: usize,              // number of levels within the LOB to use for patches
    pub prediction_intervals: usize,    // number of intervals/snapshots to use for predictions
    pub patch_intervals: usize,         // number of intervals to use per patch
    pub patch_stride: usize,            // number of intervals to skip for each patch (usually same as patch_intervals)
    pub gain_percentage: f64,           // percentage gain to achieve by holding_time for "buy" recommendation
    pub loss_percentage: f64,           // percentage loss to achieve by holding_time for "sell" recommendation
    
    pub price_mean: f64,                // mean price for all of the samples
    pub price_std_dev: f64,             // price standard deviation
    pub volume_mean: f64,               // mean volume for all of the samples
    pub volume_std_dev: f64,            // volume standard deviation

    pub positional_max_timescale: usize,    // parameter for positional encoder

    pub gain_repeats: usize,
    pub neutral_repeats: usize,
    pub loss_repeats: usize,

    pub n_gains: usize,
    pub n_neutrals: usize,
    pub n_losses: usize,
}


impl DataSpec {
    pub fn new(
        sequence_length: usize,
        patch_size: usize,
        token_size: usize,
        extraction_interval_nanos: u64,
        holding_time_seconds: u16,
        lob_levels: usize,
        prediction_intervals: usize,
        patch_intervals: usize,
        patch_stride: usize,
        gain_percentage: f64,
        loss_percentage: f64,
        
        price_mean: f64,
        price_std_dev: f64,
        volume_mean: f64,
        volume_std_dev: f64,

        positional_max_timescale: usize,

        gain_repeats: usize,
        neutral_repeats: usize,
        loss_repeats: usize,

        n_gains: usize,
        n_neutrals: usize,
        n_losses: usize,
    ) -> Self {
        Self {
            sequence_length,
            patch_size,
            token_size,
            extraction_interval_nanos,
            holding_time_seconds,
            lob_levels,
            prediction_intervals,
            patch_intervals,
            patch_stride,
            gain_percentage,
            loss_percentage,
            
            price_mean,
            price_std_dev,
            volume_mean,
            volume_std_dev,

            positional_max_timescale,

            gain_repeats,
            neutral_repeats,
            loss_repeats,

            n_gains,
            n_neutrals,
            n_losses,
        }
    }

    pub fn to_file(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let spec_file = File::create(path)?;
        let spec_writer = BufWriter::new(spec_file);
        serde_json::to_writer_pretty(spec_writer, self)?;
        Ok(())
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let spec_file_content = std::fs::read_to_string(path)?;
        let spec: Self = serde_json::from_str(&spec_file_content)?;
        Ok(spec)
    }
}


#[derive(Debug)]
pub struct DataSpecBuilder {
    pub sequence_length: usize,         // number of items in temporal sequence
    pub patch_size: usize,              // number of values in a patch (lob_depth * patch_temporal_window)
    pub token_size: usize,              // in our use case it's 4 * patch_size (bid_price, bid_volume, ask_price, ask_volume)

    pub extraction_interval_nanos: u64, // to discretize snapshots, how many nanons between emitted samples
    pub holding_time_seconds: u16,      // number of seconds to hold the stock prior to selling
    pub lob_levels: usize,              // number of levels within the LOB to use for patches
    pub prediction_intervals: usize,    // number of intervals/snapshots to use for predictions
    pub patch_intervals: usize,         // number of intervals to use per patch
    pub patch_stride: usize,            // number of intervals to skip for each patch (usually same as patch_intervals)
    pub gain_percentage: f64,           // percentage gain to achieve by holding_time for "buy" recommendation
    pub loss_percentage: f64,           // percentage loss to achieve by holding_time for "sell" recommendation

    pub price_mean: f64,                // mean price for all of the samples
    pub price_std_dev: f64,             // price standard deviation
    pub volume_mean: f64,               // mean volume for all of the samples
    pub volume_std_dev: f64,            // volume standard deviation

    pub positional_max_timescale: usize,    // parameter for positional encoder

    pub gain_repeats: usize,
    pub neutral_repeats: usize,
    pub loss_repeats: usize,

    pub n_gains: usize,
    pub n_neutrals: usize,
    pub n_losses: usize,
}


impl DataSpecBuilder {
    pub fn new() -> Self {
        Self {
            sequence_length: 128,
            patch_size: 40,
            token_size: 160,
            extraction_interval_nanos: 250_000_000_000,
            holding_time_seconds: 10,
            lob_levels: 10,
            prediction_intervals: 64,
            patch_intervals: 8,
            patch_stride: 8,
            gain_percentage: 0.1,
            loss_percentage: 0.1,
            price_mean: 0.0,
            price_std_dev: 0.0,
            volume_mean: 0.0,
            volume_std_dev: 0.0,
            positional_max_timescale: 1_000_000,

            gain_repeats: 1,
            neutral_repeats: 1,
            loss_repeats: 1,

            n_gains: 0,
            n_neutrals: 0,
            n_losses: 0,
        }
    }

    pub fn sequence_length(mut self, sequence_length: usize) -> Self {
        self.sequence_length = sequence_length;
        self
    }


    pub fn patch_size(mut self, patch_size: usize) -> Self {
        self.patch_size = patch_size;
        self
    }

    pub fn token_size(mut self, token_size: usize) -> Self {
        self.token_size = token_size;
        self
    }

    pub fn extraction_interval_nanos(mut self, extraction_interval_nanos: u64) -> Self {
        self.extraction_interval_nanos = extraction_interval_nanos;
        self
    }


    pub fn holding_time_seconds(mut self, holding_time_seconds: u16) -> Self {
        self.holding_time_seconds = holding_time_seconds;
        self
    }

    pub fn lob_levels(mut self, lob_levels: usize) -> Self {
        self.lob_levels = lob_levels;
        self
    }

    pub fn prediction_intervals(mut self, prediction_intervals: usize) -> Self {
        self.prediction_intervals = prediction_intervals;
        self
    }

    pub fn patch_intervals(mut self, patch_intervals: usize) -> Self {
        self.patch_intervals = patch_intervals;
        self
    }

    pub fn patch_stride(mut self, patch_stride: usize) -> Self {
        self.patch_stride = patch_stride;
        self
    }
    pub fn gain_percentage(mut self, gain_percentage: f64) -> Self {
        self.gain_percentage = gain_percentage;
        self
    }

    pub fn loss_percentage(mut self, loss_percentage: f64) -> Self {
        self.loss_percentage = loss_percentage;
        self
    }

    pub fn price_mean(mut self, price_mean: f64) -> Self {
        self.price_mean = price_mean;
        self
    }

    pub fn price_std_dev(mut self, price_std_dev: f64) -> Self {
        self.price_std_dev = price_std_dev;
        self
    }

    pub fn volume_mean(mut self, volume_mean: f64) -> Self {
        self.volume_mean = volume_mean;
        self
    }

    pub fn volume_std_dev(mut self, volume_std_dev: f64) -> Self {
        self.volume_std_dev = volume_std_dev;
        self
    }

    pub fn positional_max_timescale(mut self, positional_max_timescale: usize) -> Self {
        self.positional_max_timescale = positional_max_timescale;
        self
    }


    pub fn gain_repeats(mut self, gain_repeats: usize) -> Self {
        self.gain_repeats = gain_repeats;
        self
    }

    pub fn neutral_repeats(mut self, neutral_repeats: usize) -> Self {
        self.neutral_repeats = neutral_repeats;
        self
    }

    pub fn loss_repeats(mut self, loss_repeats: usize) -> Self {
        self.loss_repeats = loss_repeats;
        self
    }


    pub fn n_gains(mut self, n_gains: usize) -> Self {
        self.n_gains = n_gains;
        self
    }

    pub fn n_neutrals(mut self, n_neutrals: usize) -> Self {
        self.n_neutrals = n_neutrals;
        self
    }

    pub fn n_losses(mut self, n_losses: usize) -> Self {
        self.n_losses = n_losses;
        self
    }




    pub fn build(&self) -> DataSpec {
        DataSpec::new(
            self.sequence_length,
            self.patch_size,
            self.token_size,
            self.extraction_interval_nanos,
            self.holding_time_seconds,
            self.lob_levels,
            self.prediction_intervals,
            self.patch_intervals,
            self.patch_stride,
            self.gain_percentage,
            self.loss_percentage,
            self.price_mean,
            self.price_std_dev,
            self.volume_mean,
            self.volume_std_dev,
            self.positional_max_timescale,

            self.gain_repeats,
            self.neutral_repeats,
            self.loss_repeats,
            
            self.n_gains,
            self.n_neutrals,
            self.n_losses,
        )
    }
}

