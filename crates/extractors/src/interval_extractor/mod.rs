
use std::error::Error;
use databento::{
    dbn::{
        Action,
        MboMsg,
        TsSymbolMap,
        pretty,
    },
};
use order_book::{
    book::Book,
    price_level::PriceLevel,
};
use crate::extractor::{Extractor};
use tracing::{debug, info};
use serde::{ Serialize, Deserialize };
use utilities::date_time::nanos_to_offset_date_time_with_tz;


// a price level
#[derive(Debug, Serialize, Deserialize)]
struct PriceVolumeLevel {
    price: f64,
    volume: u32,
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

        let action = mbo.action().unwrap();

        // Presume Nasdaq or NYSE, so local time is eastern - either EST or EDT depending upon time of year
        let received_date_time = nanos_to_offset_date_time_with_tz(mbo.ts_recv as i128, "ET")?;

        let day_of_week = received_date_time.monday_based_week();
        let local_hour = received_date_time.hour();
        let local_minute = received_date_time.minute();

        //
        // Monday through Friday between 09:40:00 and 15:50:00
        //   i.e. on exchange day between 10 minutes after the open and 10 minutes prior to the close
        //
        if day_of_week < 5 &&
            (local_hour > 9 || (local_hour == 9 && local_minute >= 10)) &&
            (local_hour < 15 || (local_hour == 15 && local_minute <= 50)) {

            // continue processing
            let mut bid_levels: Vec<PriceLevel> = self.book.bid_levels(self.nbr_lob_levels).collect();
            let mut ask_levels: Vec<PriceLevel> = self.book.ask_levels(self.nbr_lob_levels).collect();

            let mut bid_price_volume_levels: Vec<PriceVolumeLevel> = Vec::new();
            let mut ask_price_volume_levels: Vec<PriceVolumeLevel> = Vec::new();

            if bid_levels.len() > 0 && ask_levels.len() > 0 {
                for i in 0..self.nbr_lob_levels {
                    // bids
                    if let Some(bid_level) = bid_levels.get(i) {
                        bid_price_volume_levels.push(PriceVolumeLevel {
                            price: bid_level.price as f64 / 1_000_000_000_f64,
                            volume: bid_level.size,
                        });
                    }
                    else {
                        let level = bid_price_volume_levels.get(i - 1).unwrap();
                        bid_price_volume_levels.push(PriceVolumeLevel {
                            price: level.price as f64 / 1_000_000_000_f64,
                            volume: 0,
                        });
                    }

                    // asks
                    if let Some(ask_level) = ask_levels.get(i) {
                        ask_price_volume_levels.push(PriceVolumeLevel {
                            price: ask_level.price as f64 / 1_000_000_000_f64,
                            volume: ask_level.size,
                        });
                    }
                    else {
                        let level = ask_price_volume_levels.get(i - 1).unwrap();
                        ask_price_volume_levels.push(PriceVolumeLevel {
                            price: level.price as f64 / 1_000_000_000_f64,
                            volume: 0,
                        });
                    }
                }
            }
        }


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

