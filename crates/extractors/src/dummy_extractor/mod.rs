use std::error::Error;
use databento::dbn::{MboMsg, TsSymbolMap};
use order_book::market::Market;
use crate::extractpr::{Strategy};

#[derive(Debug)]
pub struct DummyStrategy {}

impl DummyStrategy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Strategy for DummyStrategy {
    async fn pre_apply(&mut self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    async fn post_apply(&mut self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}


