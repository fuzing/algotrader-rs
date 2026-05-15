
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
use crate::strategy::{Strategy};
use tracing::{debug, info};




#[derive(Debug)]
enum AiStrategyState {
    Waiting,
    // processing with (start_time, purchase_price, success_price, stop_loss_price)
    Processing(
        u64,        // start_time
        u64,        // end_time
        u32,        // purchase_shares
        i64,        // purchase_price
        i64,        // success_price
        i64,        // stop_loss price
    ),
}

#[derive(Debug)]
pub struct AiStrategy {
    last_trade_price: Option<i64>,
    current_state: AiStrategyState,
    profit_loss: i64,
    total_shares_traded: u32,

    purchase_shares: u32,
    minimum_ask_shares_in_book: u32,
    bid_ask_volume_ratio: f32,      // e.g. 2.0 would mean that the buy is triggered when bid volume is 2x ask volume
    maximum_holding_time: u32,         // duration to wait for success in seconds, otherwise fail
    desired_gain_percentage: f32,   // when this upside price is breached then exit the trade
    stop_loss_percentage: f32,      // when the price hits the loss point then do this
}

impl AiStrategy {
    pub fn new(
        purchase_shares: u32,
        minimum_ask_shares_in_book: u32,
        bid_ask_volume_ratio: f32,
        maximum_holding_time: u32,
        desired_gain_percentage: f32,
        stop_loss_percentage: f32
    ) -> Self {
        Self {
            last_trade_price: None,
            current_state: AiStrategyState::Waiting,
            profit_loss: 0,
            total_shares_traded: 0,
            purchase_shares,
            minimum_ask_shares_in_book,
            bid_ask_volume_ratio,
            maximum_holding_time,
            desired_gain_percentage,
            stop_loss_percentage,
        }
    }

    pub fn builder() -> AiStrategyBuilder {
        AiStrategyBuilder::default()
    }

    pub fn profit_loss(&self) -> f32 {
        (self.profit_loss / 1_000_000_000) as f32
    }
    pub fn total_shares_traded(&self) -> u32 {
        self.total_shares_traded
    }
}

impl Strategy for AiStrategy {
    // async fn pre_apply(&mut self, msg: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {
    //     Ok(())
    // }

    async fn post_apply(&mut self, mbo: &MboMsg, symbol_map: &TsSymbolMap, market: &Market) -> Result<(), Box<dyn Error>> {

        // let action = mbo.action().unwrap();
        // match action {
        //     Action::Modify => {
        //         debug!("Post Modify");
        //         // self.modify(mbo)
        //     },
        //     // Action::Trade | Action::Fill | Action::None => {}
        //     Action::Trade => {
        //         info!("Post Trade at ${} @ {}", pretty::Px(mbo.price), mbo.ts_recv().unwrap());
        //     },
        //     Action::Fill => {
        //         debug!("Post Fill");
        //     },
        //     Action::None => {
        //         debug!("Post None");
        //     },
        //     Action::Cancel => {
        //         debug!("Post Cancel");
        //         // self.cancel(mbo)
        //     },
        //     Action::Add => {
        //         debug!("Post Add");
        //         // self.add(mbo)
        //     },
        //     Action::Clear => {
        //         debug!("Post Clear");
        //         // self.clear()
        //     },
        // }

        let action = mbo.action().unwrap();

        if action == Action::Trade {
            self.last_trade_price = Some(mbo.price);
            self.total_shares_traded += mbo.size;
        }

        match self.current_state {
            AiStrategyState::Waiting => {
                if let Some(last_trade_price) = self.last_trade_price {
                    if let Some(book) = market.find_book_from_mbo(mbo) {
                        // println!("----------------------------------------------------------------------------------------------");
                        // for x in book.ask_levels(5) {
                        //     println!("Ask {}", pretty::Px(x.price));
                        // }
                        // println!("");
                        // for x in book.bid_levels(5) {
                        //     println!("Bid {}", pretty::Px(x.price));
                        // }
                        // let (bid, ask) = book.bbo();
                        // if let Some(bid) = bid {
                        //     println!("Best Bid {}", pretty::Px(bid.price));
                        // }
                        // if let Some(ask) = ask {
                        //     println!("Best Ask {}", pretty::Px(ask.price));
                        // }


                        let bid_levels = book.bid_levels(usize::MAX);
                        let (total_bid_orders, total_bid_shares) = bid_levels.fold((0, 0), |(total_orders, total_shares), level| {
                            (total_orders + level.count, total_shares + level.size)
                        });

                        let ask_levels = book.ask_levels(usize::MAX);
                        let (total_ask_orders, total_ask_shares) = ask_levels.fold((0, 0), |(total_orders, total_shares), level| {
                            (total_orders + level.count, total_shares + level.size)
                        });

                        debug!("\n\n=======> Total Bid Orders ({total_bid_orders}), Total Bid Shares ({total_bid_shares}) => Total Ask Orders ({total_ask_orders}), Total Ask Shares ({total_ask_shares})");

                        let (best_bid, best_offer) = market.aggregated_bbo(mbo.hd.instrument_id);
                        if let Some(best_bid) = best_bid && let Some(best_offer) = best_offer {
                            // buy at the mid-point of bid/ask
                            if total_ask_shares >= self.minimum_ask_shares_in_book && (total_bid_shares as f32 / total_ask_shares as f32) > self.bid_ask_volume_ratio {
                                let limit_price = self.last_trade_price.unwrap(); // best_bid.price; // (best_bid.price + best_offer.price) / 2;

                                let stop_loss_price = limit_price - (limit_price as f32 * self.stop_loss_percentage / 100.00) as i64;
                                let success_price = limit_price + (limit_price as f32 * self.desired_gain_percentage / 100.00) as i64;
                                let end_time = mbo.ts_recv + (self.maximum_holding_time as u64 * 1_000_000_000);
                                info!("========> Purchase => Buy at ${} @ {}", pretty::Px(limit_price), mbo.ts_recv().unwrap());
                                self.current_state = AiStrategyState::Processing(mbo.ts_recv, end_time, self.purchase_shares, limit_price, success_price, stop_loss_price);
                            }
                        }
                    }
                }
            },
            AiStrategyState::Processing(start_time, end_time, purchase_shares, purchase_price, success_price, stop_loss_price) => {
                if action == Action::Trade && mbo.price >= success_price {
                    let profit = (mbo.price - purchase_price) * purchase_shares as i64;
                    self.profit_loss += profit;
                    info!("========> Success Paid(${}), Sold At(${}) Profit(${}) @ {}", pretty::Px(purchase_price), pretty::Px(mbo.price), pretty::Px(profit), mbo.ts_recv().unwrap());
                    self.current_state = AiStrategyState::Waiting;
                }
                else if action == Action::Trade && mbo.price <= stop_loss_price {
                    let profit = (mbo.price - purchase_price) * purchase_shares as i64;
                    self.profit_loss += profit;
                    info!("========> Failed Stop Loss Paid(${}), Sold At(${}) Profit(${}) @ {}", pretty::Px(purchase_price), pretty::Px(mbo.price), pretty::Px(profit), mbo.ts_recv().unwrap());
                    self.current_state = AiStrategyState::Waiting;
                }
                else if mbo.ts_recv >= end_time {
                    let sold_at_price = self.last_trade_price.unwrap();
                    let profit = (sold_at_price - purchase_price) * purchase_shares as i64;
                    self.profit_loss += profit;
                    info!("========> Timeout - Paid(${}), Sold At(${}) Profit(${}) @ {}", pretty::Px(purchase_price), pretty::Px(sold_at_price), pretty::Px(profit), mbo.ts_recv().unwrap());
                    self.current_state = AiStrategyState::Waiting;
                }

            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct AiStrategyBuilder {
    minimum_ask_shares_in_book: u32,
    purchase_shares: u32,
    bid_ask_volume_ratio: f32,
    maximum_holding_time: u32,
    desired_gain_percentage: f32,
    stop_loss_percentage: f32,
}


impl Default for AiStrategyBuilder {
    fn default() -> Self {
        Self {
            minimum_ask_shares_in_book: 100,                // minimum ask shares in book
            purchase_shares: 100,
            bid_ask_volume_ratio: 1.2,              // ratio of bid to ask in the book
            maximum_holding_time: 1_800,               // seconds
            desired_gain_percentage: 0.1,           // set limit on sell order purchase limit plus this percentage
            stop_loss_percentage: 1.0,              // if stock trades at purchase price less this percent, then sell
        }        
    }
}

impl AiStrategyBuilder {
    pub fn build(&self) -> AiStrategy {
        AiStrategy::new(
            self.purchase_shares,
            self.minimum_ask_shares_in_book,
            self.bid_ask_volume_ratio,
            self.maximum_holding_time,
            self.desired_gain_percentage,
            self.stop_loss_percentage,
        )
    }

    pub fn purchase_shares(&mut self, value: u32) -> &mut Self {
        self.purchase_shares = value;
        self
    }

    pub fn minimum_ask_shares_in_book(&mut self, value: u32) -> &mut Self {
        self.minimum_ask_shares_in_book = value;
        self
    }

    pub fn bid_ask_volume_ratio(&mut self, value: f32) -> &mut Self {
        self.bid_ask_volume_ratio = value;
        self
    }

    pub fn maximum_holding_time(&mut self, value: u32) -> &mut Self {
        self.maximum_holding_time = value;
        self
    }

    pub fn desired_gain_percentage(&mut self, value: f32) -> &mut Self {
        self.desired_gain_percentage = value;
        self
    }

    pub fn stop_loss_percentage(&mut self, value: f32) -> &mut Self {
        self.stop_loss_percentage = value;
        self
    }
}

