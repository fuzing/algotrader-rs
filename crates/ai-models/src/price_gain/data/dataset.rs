
use std::{
    fs::File,
    io::BufReader,
    path::PathBuf,
    sync::Arc,
};
use std::io::BufRead;
use memmap2::Mmap;
use burn::data::dataset::{
    Dataset,
    InMemDataset,           // PMB in memory dataset
};
use derive_new::new;

use extractors::interval_extractor::{
    ExtractedDataFile,
    IntervalExtractionWithGain
};


#[derive(new, Clone, Debug)]
pub struct PriceGainItem {
    pub features: Vec<f64>,
    pub label: f64,
}



#[derive(Debug)]
pub struct PriceGainDataset {
    // memory mapped file, shareable across threads
    mmap: Arc<Mmap>,

    train_split: (usize, usize),
    test_split: (usize, usize),
    validation_split: (usize, usize),

    // Row index of each line
    // (start byte, byte_len)
    index: Vec<(usize, usize)>,
}



impl PriceGainDataset {
    pub fn new(
        path: PathBuf,
        test_train_validate_shares: (usize, usize, usize),
    ) -> PriceGainDataset {
        let file = File::open(path.clone()).expect(&format!("Couldn't open file {path:?}"));

        let mmap = unsafe {
            Mmap::map(&file).expect("failed to memory map file")
        };


        let reader = BufReader::new(&file);

        let mut index = Vec::new();
        let mut cursor = 0;

        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result.expect("Error reading csv file during indexing");
            let len = line.len();

            // no need to skip because we have no header
            // // skip header - first row
            // if (i == 0) {
            //     cursor += len + 1;      // +1 for newline
            //     continue;
            // }
            
            index.push((cursor, len));
            cursor += len + 1;      // +1 for newline
        }
        println!("Indexed {} lines during mapping", index.len());


        let n_samples = index.len();
        let (train_share, test_share, validation_share) = test_train_validate_shares;
        let total_share = train_share + test_share + validation_share;
        let test_size = ((test_share as f64 / total_share as f64) * n_samples as f64).floor() as usize;
        let validation_size = ((validation_share as f64 / total_share as f64) * n_samples as f64).floor() as usize;
        let train_size = n_samples - test_size - validation_size;

        Self {
            train_split: (0, train_size),
            test_split: (train_size, test_size),
            validation_split: (train_size + test_size, validation_size),
            mmap: Arc::new(mmap),
            index,
        }
    }

    pub fn train_set(&mut self) -> PriceGainDatasetSubset {
        PriceGainDatasetSubset {
            dataset: self,
            offset: self.train_split.0,
            size: self.train_split.1,
        }
    }

    pub fn test_set(&self) -> PriceGainDatasetSubset {
        PriceGainDatasetSubset {
            dataset: self,
            offset: self.test_split.0,
            size: self.test_split.1,
        }
    }

    pub fn validation_set(&self) -> PriceGainDatasetSubset {
        PriceGainDatasetSubset {
            dataset: self,
            offset: self.validation_split.0,
            size: self.validation_split.1,
        }
    }
}

impl Dataset<PriceGainItem> for PriceGainDataset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        // Get byte offset -> slice metadata
        let (start, len) = *self.index.get(index)?;

        // Slice the mmap (looks like reading from RAM)
        let bytes = &self.mmap[start..start + len];

        // Convert CSV line from UTF-8 bytes
        let line_str = std::str::from_utf8(bytes).ok()?;

        // Split by comma; fast and simple (no quoting support)
        let mut values: Vec<f64> = line_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();

        if values.is_empty() {
            return None;
        }

        // Last value = label
        let label = values.pop()?; // O(1)

        Some(PriceGainItem {
            features: values,
            label,
        })
    }

    fn len(&self) -> usize {
        self.index.len()
    }
}


#[derive(Debug)]
pub struct PriceGainDatasetSubset {
    dataset: PriceGainDataset,
    offset: usize,
    size: usize,
}

impl Dataset<PriceGainItem> for PriceGainDatasetSubset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        if index >= self.size {
            return None;
        }
        self.dataset.get(self.offset + index)
    }

    fn len(&self) -> usize {
        self.size
    }
}



