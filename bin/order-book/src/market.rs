
use std::collections::HashMap;
use crate::book::Book;
use databento::dbn::Publisher;

#[derive(Debug, Default)]
pub struct Market {
    books: HashMap<u32, Vec<(Publisher, Book)>>,
}

