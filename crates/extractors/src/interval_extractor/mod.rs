
use std::error::Error;
use std::fmt::Display;
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
use time::Weekday;
use utilities::date_time::nanos_to_offset_date_time_with_tz;


// a price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceVolumeLevel {
    pub price: f64,
    pub volume: u32,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntervalExtraction {
    pub date_time_nanos: u64,                     // nanos past unix epoch
    pub last_trade_price: f64,
    pub bids: Vec<PriceVolumeLevel>,
    pub asks: Vec<PriceVolumeLevel>,
}

#[derive(Debug)]
pub struct IntervalExtractorStats {
    total_mbo_messages: usize,
    total_trades: usize,
    total_emitted_intervals: usize,
}

impl Display for IntervalExtractorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Stats:")?;
        writeln!(f, "  Total MBO Messages: {}", self.total_mbo_messages)?;
        writeln!(f, "  Total Trades: {}", self.total_trades)?;
        writeln!(f, "  Total Emitted Intervals: {}", self.total_emitted_intervals)?;

        Ok(())
    }
}


#[derive(Debug)]
pub struct IntervalExtractor {
    // Parameters
    nbr_lob_levels: usize,              // number of LOB bid/ask levels to capture per extraction
    extraction_interval_nanos: u64,            // the interval between extractions (in nanos)

    // Local state
    book: Book,                     // LOB
    last_trade_price: Option<f64>,
    next_extraction_time: Option<u64>,

    // Statistics
    total_mbo_messages: usize,
    total_trades: usize,
    total_emitted_intervals: usize,

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
            next_extraction_time: None,

            total_mbo_messages: 0,
            total_trades: 0,
            total_emitted_intervals: 0,
        }
    }

    pub fn builder() -> IntervalExtractorBuilder {
        IntervalExtractorBuilder::default()
    }
    pub fn stats(&self) -> IntervalExtractorStats {
        IntervalExtractorStats {
            total_mbo_messages: self.total_mbo_messages,
            total_trades: self.total_trades,
            total_emitted_intervals: self.total_emitted_intervals,
        }
    }
}


impl Extractor<IntervalExtraction> for IntervalExtractor {
    async fn push(&mut self, mbo: &MboMsg) -> Result<Vec<IntervalExtraction>, Box<dyn Error>> {
        self.total_mbo_messages += 1;

        // apply the MBO message to the order book
        self.book.apply(mbo.clone());

        
        let mut results: Vec<IntervalExtraction> = Vec::new();
        
        
        // Presume Nasdaq or NYSE, so local time is eastern - either EST or EDT depending upon time of year
        let received_date_time = nanos_to_offset_date_time_with_tz(mbo.ts_recv as i128, "ET")?;

        let day = received_date_time.weekday();
        let valid_day = match day {
            Weekday::Saturday | Weekday::Sunday => false,
            _ => true,
        };
        let local_hour = received_date_time.hour();
        let local_minute = received_date_time.minute();

        //
        // Monday through Friday between 09:40:00 and 15:50:00
        //   i.e. on exchange day between 10 minutes after the open and 10 minutes prior to the close
        //
        if valid_day &&
            (local_hour > 9 || (local_hour == 9 && local_minute > 45)) &&
            (local_hour < 15 || (local_hour == 15 && local_minute < 45)) {

            // if this is a trade action then use it to set the last trade price
            // println!("yep");
            let action = mbo.action().unwrap();
            if action == Action::Trade {
                self.total_trades += 1;
                self.last_trade_price = Some(mbo.price_f64());
                // println!("Last trade price: {:?}", self.last_trade_price.unwrap());

                if self.next_extraction_time.is_none() {
                    let net = mbo.ts_recv + self.extraction_interval_nanos;
                    self.next_extraction_time = Some(net - (net % self.extraction_interval_nanos));
                }
            }

            //
            // will only have a next_extraction_time if there's a valid last_trade_price
            //
            if let Some(mut next_extraction_time) = self.next_extraction_time {
                if mbo.ts_recv > next_extraction_time {
                    // main processing
                    let mut bid_levels = self.book.bid_levels(self.nbr_lob_levels);
                    let mut ask_levels = self.book.ask_levels(self.nbr_lob_levels);

                    let mut bid_price_volume_levels: Vec<PriceVolumeLevel> = Vec::new();
                    let mut ask_price_volume_levels: Vec<PriceVolumeLevel> = Vec::new();

                    let mut last_valid_bid: f64 = 0.0;
                    let mut last_valid_ask: f64 = 0.0;

                    for i in 0..self.nbr_lob_levels {
                        // bids
                        if let Some(bid_level) = bid_levels.next() {
                            last_valid_bid = bid_level.price as f64 / 1_000_000_000_f64;
                            bid_price_volume_levels.push(PriceVolumeLevel {
                                price: last_valid_bid,
                                volume: bid_level.size,
                            });
                        }
                        else {
                            bid_price_volume_levels.push(PriceVolumeLevel {
                                price: last_valid_bid,
                                volume: 0,
                            });
                        }

                        // asks
                        if let Some(ask_level) = ask_levels.next() {
                            last_valid_ask = ask_level.price as f64 / 1_000_000_000_f64;
                            ask_price_volume_levels.push(PriceVolumeLevel {
                                price: last_valid_ask,
                                volume: ask_level.size,
                            });
                        }
                        else {
                            ask_price_volume_levels.push(PriceVolumeLevel {
                                price: last_valid_ask,
                                volume: 0,
                            });
                        }
                    }

                    let last_trade_price = self.last_trade_price.unwrap();
                    while mbo.ts_recv > next_extraction_time {
                        results.push(IntervalExtraction {
                            date_time_nanos: next_extraction_time,
                            last_trade_price,
                            bids: bid_price_volume_levels.clone(),
                            asks: ask_price_volume_levels.clone(),
                        });

                        next_extraction_time += self.extraction_interval_nanos;
                    }

                    self.next_extraction_time = Some(next_extraction_time);
                    self.total_emitted_intervals += results.len();
                }
            }

        }
        else {
            // outside business day/hour
            self.last_trade_price = None;
            self.next_extraction_time = None;
        }
        Ok(results)
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

