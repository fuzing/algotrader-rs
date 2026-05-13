
use std::error::Error;
use databento::dbn::{MboMsg, TsSymbolMap};
use order_book::market::Market;
use crate::strategy::Strategy;
use tracing::{debug};

#[derive(Debug)]
pub struct TestStrategy {}

impl TestStrategy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Strategy for TestStrategy {
    async fn pre_apply(&self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        // if let Some(book) = market.find_book_from_mbo(msg) {
        //
        // }
        Ok(())
    }

    async fn post_apply(&self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {

        if let Some(book) = market.find_book_from_mbo(msg) {
            debug!("=====================> found book");
        }

        Ok(())
    }
}


