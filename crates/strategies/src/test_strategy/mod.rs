
use std::error::Error;
use databento::{
    dbn::{
        Action,
        MboMsg,
        TsSymbolMap,
        pretty,
    },
};
use order_book::market::Market;
use crate::strategy::Strategy;
use tracing::{debug, info};



#[derive(Debug)]
enum TestStrategyState {
    Waiting,
    // processing with (start_time, purchase_price, success_price, stop_loss_price)
    Processing(
        u64,        // start_time
        u64,        // end_time
        i64,        // purchase_price
        i64,        // success_price
        i64,        // stop_loss price
    ),
}

#[derive(Debug)]
pub struct TestStrategy {
    last_trade_price: Option<i64>,
    current_state: TestStrategyState,

    bid_ask_volume_ratio: f32,      // e.g. 2.0 would mean that the buy is triggered when bid volume is 2x ask volume
    holding_wait_time: u32,                     // duration to wait for success in seconds, otherwise fail
    gain_success_percentage: f32,            // when this upside price is breached then exit the trade
    stop_loss_percentage: f32,          // when the price hits the loss point then do this
}

impl TestStrategy {
    pub fn new() -> Self {
        Self {
            last_trade_price: None,
            current_state: TestStrategyState::Waiting,
            bid_ask_volume_ratio: 2.0,
            holding_wait_time: 3_600,                           // 1 hour
            gain_success_percentage: 0.25,
            stop_loss_percentage: 1.00,
        }
    }
}

impl Strategy for TestStrategy {
    async fn pre_apply(&mut self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
        // if let Some(book) = market.find_book_from_mbo(msg) {
        //
        // }
        Ok(())
    }

    async fn post_apply(&mut self, mbo: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {

        let action = mbo.action().unwrap();
        match action {
            Action::Modify => {
                debug!("Post Modify");
                // self.modify(mbo)
            },
            // Action::Trade | Action::Fill | Action::None => {}
            Action::Trade => {
                info!("Post Trade at ${} @ {}", pretty::Px(mbo.price), mbo.ts_recv().unwrap());
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

        let action = mbo.action().unwrap();

        if action == Action::Trade {
            self.last_trade_price = Some(mbo.price);
        }

        match self.current_state {
            TestStrategyState::Waiting => {
                if let Some(last_trade_price) = self.last_trade_price {
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

                        let (best_bid, best_offer) = market.aggregated_bbo(mbo.hd.instrument_id);
                        if let Some(best_bid) = best_bid && let Some(best_offer) = best_offer {
                            // buy at the mid-point of bid/ask
                            if total_ask_shares > 0 && (total_bid_shares as f32 / total_ask_shares as f32) > self.bid_ask_volume_ratio {
                                let limit_price = (best_bid.price + best_offer.price) / 2;

                                let stop_loss_price = limit_price - (limit_price as f32 * self.stop_loss_percentage / 100.00) as i64;
                                let success_price = limit_price + (limit_price as f32 * self.gain_success_percentage / 100.00) as i64;
                                let end_time = mbo.ts_recv + (self.holding_wait_time as u64 * 1_000_000_000);
                                info!("========> BINGO!!!!   buy at ${}", pretty::Px(limit_price));
                                self.current_state = TestStrategyState::Processing(mbo.ts_recv, end_time, limit_price, success_price, stop_loss_price);
                            }
                        }
                    }
                }
            },
            TestStrategyState::Processing(start_time, end_time, purchase_price, success_price, stop_loss_price) => {
                // 0.1% move
                if action == Action::Trade && mbo.price >= success_price {
                    info!("========> Success Trade at Paid(${}), Sold At(${}) @ {}", pretty::Px(purchase_price), pretty::Px(mbo.price), mbo.ts_recv().unwrap());
                    self.current_state = TestStrategyState::Waiting;
                }
                else if action == Action::Trade && mbo.price <= stop_loss_price {
                    info!("========> Failed Stop Loss Trade at Paid(${}), Sold At(${}) @ {}", pretty::Px(purchase_price), pretty::Px(mbo.price), mbo.ts_recv().unwrap());
                    self.current_state = TestStrategyState::Waiting;
                }
                else if mbo.ts_recv >= end_time {
                    info!("=======> Failed Time trade Paid(${}) {start_time} -> {}", pretty::Px(purchase_price), mbo.ts_recv);
                    self.current_state = TestStrategyState::Waiting;
                }

            }
        }




        Ok(())
    }
}


