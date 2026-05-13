
use std::error::Error;
use databento::dbn::{Action, MboMsg, TsSymbolMap};
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

    async fn post_apply(&self, mbo: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {

        let action = mbo.action().unwrap();
        match action {
            Action::Modify => {
                debug!("Post Modify");
                // self.modify(mbo)
            },
            // Action::Trade | Action::Fill | Action::None => {}
            Action::Trade => {
                debug!("Post Trade");
            },
            Action::Fill => {
                debug!("Post Fill");
            },
            Action::None => {
                debug!("Post None");
            },
            Action::Cancel => {
                debug!("Post Cancel");
                // self.cancel(mbo)
            },
            Action::Add => {
                debug!("Post Add");
                // self.add(mbo)
            },
            Action::Clear => {
                debug!("Post Clear");
                // self.clear()
            },
        }


        if let Some(book) = market.find_book_from_mbo(mbo) {
            let bid_levels = book.bid_levels();
            let (total_bid_orders, total_bid_shares) = bid_levels.fold((0, 0), |(total_orders, total_shares), level| {
                (total_orders + level.count, total_shares + level.size)
            });

            let ask_levels = book.ask_levels();
            let (total_ask_orders, total_ask_shares) = ask_levels.fold((0, 0), |(total_orders, total_shares), level| {
                (total_orders + level.count, total_shares + level.size)
            });

            debug!("\n\n=======> Total Bid Orders ({total_bid_orders}), Total Bid Shares ({total_bid_shares}) => Total Ask Orders ({total_ask_orders}), Total Ask Shares ({total_ask_shares})");

            if total_ask_shares > 0 && total_bid_shares / total_ask_shares > 3 {
                debug!("========> BINGO!!!!");
            }
        }

        Ok(())
    }
}


