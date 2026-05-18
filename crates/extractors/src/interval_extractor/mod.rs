
use std::error::Error;
use databento::{
    dbn::{
        Action,
        MboMsg,
        TsSymbolMap,
        pretty,
    },
};
use order_book::book::Book;
use crate::extractor::{Extractor};
use tracing::{debug, info};
use serde::{ Serialize, Deserialize };
use utilities::date_time::nanos_to_offset_date_time_with_tz;


// a price level
#[derive(Debug, Serialize, Deserialize)]
struct PriceVolumeLevel {
    price: f64,
    volume: u64,
}


#[derive(Debug, Serialize, Deserialize)]
struct IntervalExtraction {
    date_time_nanos: u64,                     // nanos past unix epoch
    last_trade_price: Option<i64>,
    bids: Vec<PriceVolumeLevel>,
    asks: Vec<PriceVolumeLevel>,
}



#[derive(Debug)]
pub struct IntervalExtractor {
    nbr_lob_levels: usize,              // number of LOB bid/ask levels to capture per extraction
    extraction_interval_nanos: u64,            // the interval between extractions (in nanos)


    book: Book,                     // LOB
    last_trade_price: Option<i64>,


    // current_state: IntervalExtractorState,
    // profit_loss: i64,
    // total_shares_traded: u32,
    //
    // purchase_shares: u32,
    // minimum_ask_shares_in_book: u32,
    // bid_ask_volume_ratio: f32,      // e.g. 2.0 would mean that the buy is triggered when bid volume is 2x ask volume
    // maximum_holding_time: u32,         // duration to wait for success in seconds, otherwise fail
    // desired_gain_percentage: f32,   // when this upside price is breached then exit the trade
    // stop_loss_percentage: f32,      // when the price hits the loss point then do this
}

impl IntervalExtractor {
    pub fn new(
        nbr_lob_levels: usize,
        extraction_interval_nanos: u64,
    ) -> Self {
        Self {
            nbr_lob_levels,
            extraction_interval_nanos,
            book: Book::new(),
            last_trade_price: None,
        }
    }

    pub fn builder() -> IntervalExtractorBuilder {
        IntervalExtractorBuilder::default()
    }

}

impl Extractor<f64> for IntervalExtractor {
    async fn push(&mut self, mbo: &MboMsg) -> Result<Vec<f64>, Box<dyn Error>> {

        // apply the MBO message to the order book
        self.book.apply(mbo.clone());

        // let received_date_time = mbo.ts_recv().unwrap();
        let received_date_Time = nanos_to_offset_date_time_with_tz(mbo.ts_recv as i128, "ET")?;


        Ok(vec![])
    }
}

#[derive(Debug)]
pub struct IntervalExtractorBuilder {
    nbr_lob_levels: usize,
    extraction_interval_nanos: u64,
}


impl Default for IntervalExtractorBuilder {
    fn default() -> Self {
        Self {
            nbr_lob_levels: 10,                // minimum ask shares in book
            extraction_interval_nanos: 1_000_000_000,       // 1 second intervals
        }        
    }
}

impl IntervalExtractorBuilder {
    pub fn build(&self) -> IntervalExtractor {
        IntervalExtractor::new(
            self.nbr_lob_levels,
            self.extraction_interval_nanos,
        )
    }

    pub fn nbr_lob_levels(&mut self, value: usize) -> &mut Self {
        self.nbr_lob_levels = value;
        self
    }

    pub fn extraction_interval_nanos(&mut self, value: u64) -> &mut Self {
        self.extraction_interval_nanos = value;
        self
    }
}

