
use std::fmt::Display;
use databento::{dbn::{MboMsg, pretty}};


#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: i64,
    pub size: u32,
    pub count: u32,
}

impl PriceLevel {
    pub fn new<'a>(price: i64, orders: impl Iterator<Item = &'a MboMsg>) -> Self {
        orders.fold(
            PriceLevel {
                price,
                size: 0,
                count: 0,
            },
            |mut level, order| {
                if !order.flags.is_tob() {
                    level.count += 1;
                }
                level.size += order.size;
                level
            },
        )
    }
}

impl Display for PriceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:4} @ {:6.2} | {:2} order(s)",
            self.size,
            pretty::Px(self.price),
            self.count
        )
    }
}

