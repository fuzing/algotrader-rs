
use std::{
    fs::File,
    io::BufReader,
    path::PathBuf,
    sync::Arc,
};
use std::io::BufRead;
use memmap2::{
    Mmap,
    Advice,
};
use burn::data::dataset::{
    Dataset,
    InMemDataset,           // PMB in memory dataset
};
use derive_new::new;

use crate::price_gain::data::data_spec::DataSpec;


#[derive(new, Clone, Debug)]
pub struct PriceGainItem {
    pub features: Vec<Vec<f64>>,        // [sequence_length, token_size]
    pub label: f64,
}


#[derive(Debug, Clone)]
pub struct PriceGainDataset {
    // memory mapped file, shareable across threads
    mapped_file: Arc<Mmap>,

    // Row index of each line
    // (start byte, byte_len)
    index: Vec<(usize, usize)>,

    pub spec: DataSpec,
}



impl PriceGainDataset {
    pub fn new(
        spec_path: &PathBuf,
        data_path: &PathBuf,
    ) -> PriceGainDataset {

        //
        // access specs
        //
        let spec = DataSpec::from_file(spec_path).expect(&format!("Couldn't open spec file {spec_path:?}"));

        //
        // now process data
        //
        let file = File::open(&data_path).expect(&format!("Couldn't open data file {:?}", &data_path));
        let mapped_file = unsafe {
            Mmap::map(&file).expect("failed to memory map file")
        };
        mapped_file.advise(Advice::Sequential).expect("failed to advise mmap of sequential");


        // let reader = BufReader::new(&file);
        let reader = BufReader::with_capacity(128 * 1_024, &file);

        let mut index = Vec::new();
        let mut cursor = 0;

        println!("Dataset - mapping line starting positions");
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

        Self {
            spec,
            mapped_file: Arc::new(mapped_file),
            index,
        }
    }


    pub fn specs(&self) -> DataSpec {
        self.spec.clone()
    }


    pub fn num_classes() -> usize { 3 }

    pub fn class_name(label: usize) -> String {
        match label {
            0 => "Loss",
            1 => "Neutral",
            2 => "Gain",
            _ => panic!("Invalid class label {}", label)
        }.to_string()
    }
}

impl Dataset<PriceGainItem> for PriceGainDataset {
    fn get(&self, index: usize) -> Option<PriceGainItem> {
        // Get byte offset -> slice metadata
        let (start, len) = *self.index.get(index)?;

        // Slice the mmap (looks like reading from RAM)
        let bytes = &self.mapped_file[start..start + len];

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

        // Last value from the row is the "label"
        let label = values.pop()?; // O(1)

        // now arrange into [sequence_length, d_model]
        if values.len() != self.spec.sequence_length * self.spec.token_size {
            panic!("values is the wrong length: ({}) vs expect size of ({})", values.len(), self.spec.sequence_length * self.spec.token_size);
        }
        let chunks = values
            .chunks(self.spec.token_size)
            .map(|slice| slice.to_vec())
            .collect();

        Some(PriceGainItem {
            features: chunks,
            label,
        })
    }

    fn len(&self) -> usize {
        self.index.len()
    }
}


