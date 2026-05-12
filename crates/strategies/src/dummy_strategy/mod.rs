use std::error::Error;
use databento::dbn::{MboMsg, TsSymbolMap};
use order_book::market::Market;
use crate::strategy::Strategy;

#[derive(Debug)]
pub struct DummyStrategy {}

impl Strategy for DummyStrategy {
    async fn pre_apply(msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    async fn post_apply(msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}


