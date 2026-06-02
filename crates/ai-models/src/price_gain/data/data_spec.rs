//
// specifies the details of the csv data
//

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataSpec {
    pub sequence_length: usize,         // number of items in temporal sequence
    pub patch_size: usize,              // number of values in a patch (lob_depth * patch_temporal_window)
    pub token_size: usize,              // in our use case it's 4 * patch_size (bid_price, bid_volume, ask_price, ask_volume)
}


impl DataSpec {
    pub fn new(sequence_length: usize, patch_size: usize, token_size: usize) -> Self {
        Self {
            sequence_length,
            patch_size,
            token_size,
        }
    }
}



