
use std::error::Error;
use databento::{
    dbn::{
        TsSymbolMap,
        MboMsg,
    },
};

use order_book::market::Market;


pub trait Extractor<M> {
    async fn push(&mut self, msg: &MboMsg) -> Result<Vec<M>, Box<dyn Error>> { Ok(vec![]) }
}

