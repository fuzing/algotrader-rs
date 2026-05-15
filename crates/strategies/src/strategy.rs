
use std::error::Error;
use databento::{
    dbn::{
        TsSymbolMap,
        MboMsg,
    },
};

use order_book::market::Market;



#[derive(Debug)]
pub enum StrategyMode {
    Training,               // Mode is training - e.g. for AI model training
    Live,                   // Mode is live - e.g. for AI model inference
}


pub trait Strategy {
    // called prior to the Mbo being applied to the order book
    async fn pre_apply(&mut self, mode: StrategyMode, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> { Ok(()) }

    // called after the Mbo has been applied to the order book
    async fn post_apply(&mut self, mode: StrategyMode, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> { Ok(()) } 
}

