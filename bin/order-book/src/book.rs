
use std::collections::{HashMap, BTreeMap};
use crate::level::Level;
use databento::{dbn::Side};

#[derive(Debug, Default)]
pub struct Book {
    orders_by_id: HashMap<u64, (Side, i64)>,
    offers: BTreeMap<i64, Level>,
    bids: BTreeMap<i64, Level>,
}


