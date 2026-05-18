use std::error::Error;
use databento::dbn::{MboMsg, TsSymbolMap};
use order_book::market::Market;
use crate::extractor::{Extractor};

#[derive(Debug)]
pub struct DummyExtractor {}

impl DummyExtractor {
    pub fn new() -> Self {
        Self {}
    }
}

impl Extractor<f64> for DummyExtractor {
    async fn push(&mut self, msg: &MboMsg) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(vec![])
    }

}


