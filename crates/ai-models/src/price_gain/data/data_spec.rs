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
pub struct PriceGainDataSpec {
    pub type_: String,
    pub sequence_length: usize,         // number of items in temporal sequence
    pub patch_size: usize,              // number of values in a patch (lob_depth * patch_temporal_window)
    pub token_size: usize,              // in our use case it's 4 * patch_size (bid_price, bid_volume, ask_price, ask_volume)

    pub extraction_interval_nanos: u64, // to discretize snapshots, how many nanons between emitted samples
    pub holding_time_seconds: u16,      // number of seconds to hold the stock prior to selling
    pub lob_levels: usize,              // number of levels within the LOB to use for patches
    pub prediction_intervals: usize,    // number of intervals/snapshots to use for predictions
    pub patch_intervals: usize,         // number of intervals to use per patch
    pub patch_stride: usize,            // number of intervals to skip for each patch (usually same as patch_intervals)
    
    pub price_mean: f64,                // mean price for all of the samples
    pub price_std_dev: f64,             // price standard deviation
    pub volume_mean: f64,               // mean volume for all of the samples
    pub volume_std_dev: f64,            // volume standard deviation

    pub start_date: String,
    pub end_date: String,
}


impl PriceGainDataSpec {
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
        
        price_mean: f64,
        price_std_dev: f64,
        volume_mean: f64,
        volume_std_dev: f64,

        start_date: String,
        end_date: String,
    ) -> Self {
        Self {
            type_: "PriceGain".to_string(),
            sequence_length,
            patch_size,
            token_size,
            extraction_interval_nanos,
            holding_time_seconds,
            lob_levels,
            prediction_intervals,
            patch_intervals,
            patch_stride,
            
            price_mean,
            price_std_dev,
            volume_mean,
            volume_std_dev,

            start_date,
            end_date,
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
pub struct PriceGainDataSpecBuilder {
    pub sequence_length: usize,         // number of items in temporal sequence
    pub patch_size: usize,              // number of values in a patch (lob_depth * patch_temporal_window)
    pub token_size: usize,              // in our use case it's 4 * patch_size (bid_price, bid_volume, ask_price, ask_volume)

    pub extraction_interval_nanos: u64, // to discretize snapshots, how many nanons between emitted samples
    pub holding_time_seconds: u16,      // number of seconds to hold the stock prior to selling
    pub lob_levels: usize,              // number of levels within the LOB to use for patches
    pub prediction_intervals: usize,    // number of intervals/snapshots to use for predictions
    pub patch_intervals: usize,         // number of intervals to use per patch
    pub patch_stride: usize,            // number of intervals to skip for each patch (usually same as patch_intervals)

    pub price_mean: f64,                // mean price for all of the samples
    pub price_std_dev: f64,             // price standard deviation
    pub volume_mean: f64,               // mean volume for all of the samples
    pub volume_std_dev: f64,            // volume standard deviation

    pub start_date: String,
    pub end_date: String,
}


impl PriceGainDataSpecBuilder {
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
            price_mean: 0.0,
            price_std_dev: 0.0,
            volume_mean: 0.0,
            volume_std_dev: 0.0,

            start_date: "".to_string(),
            end_date: "".to_string(),
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

    pub fn start_date(mut self, start_date: &str) -> Self {
        self.start_date = start_date.to_string();
        self
    }

    pub fn end_date(mut self, end_date: &str) -> Self {
        self.end_date = end_date.to_string();
        self
    }


    pub fn build(&self) -> PriceGainDataSpec {
        PriceGainDataSpec::new(
            self.sequence_length,
            self.patch_size,
            self.token_size,
            self.extraction_interval_nanos,
            self.holding_time_seconds,
            self.lob_levels,
            self.prediction_intervals,
            self.patch_intervals,
            self.patch_stride,
            self.price_mean,
            self.price_std_dev,
            self.volume_mean,
            self.volume_std_dev,

            self.start_date.clone(),
            self.end_date.clone(),
        )
    }
}

