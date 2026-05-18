
use std::error::Error;
use databento::{
    dbn::{
        MboMsg,
    },
};
use order_book::book::Book;


// generic M denotes the format of the messages that are emitted from the extractor
pub trait Extractor<M> {
    async fn push(&mut self, msg: &MboMsg) -> Result<Vec<M>, Box<dyn Error>> { Ok(vec![]) }
}

