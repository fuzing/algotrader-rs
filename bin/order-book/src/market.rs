
use std::collections::HashMap;
use crate::book::Book;
use crate::price_level::PriceLevel;
use databento::dbn::Publisher;

#[derive(Debug, Default)]
pub struct Market {
    books: HashMap<u32, Vec<(Publisher, Book)>>,
}

impl Market {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn books_by_pub(&self, instrument_id: u32) -> Option<&[(Publisher, Book)]> {
        self.books
            .get(&instrument_id)
            .map(|pub_books| pub_books.as_slice())
    }

    pub fn book(&self, instrument_id: u32, publisher: Publisher) -> Option<&Book> {
        let books = self.books.get(&instrument_id)?;
        books.iter().find_map(|(book_pub, book)| {
            if *book_pub == publisher {
                Some(book)
            } else {
                None
            }
        })
    }

    pub fn bbo(
        &self,
        instrument_id: u32,
        publisher: Publisher,
    ) -> (Option<PriceLevel>, Option<PriceLevel>) {
        self.book(instrument_id, publisher)
            .map(|book| book.bbo())
            .unwrap_or_default()
    }

    pub fn aggregated_bbo(&self, instrument_id: u32) -> (Option<PriceLevel>, Option<PriceLevel>) {
        let mut agg_bid = None;
        let mut agg_ask = None;
        let Some(books_by_pub) = self.books_by_pub(instrument_id) else {
            return (None, None);
        };
        for (_, book) in books_by_pub.iter() {
            let (bid, ask) = book.bbo();
            if let Some(bid) = bid {
                match &mut agg_bid {
                    None => agg_bid = Some(bid),
                    Some(ab) if bid.price > ab.price => agg_bid = Some(bid),
                    Some(ab) if bid.price == ab.price => {
                        ab.size += bid.size;
                        ab.count += bid.count;
                    }
                    Some(_) => {}
                }
            }
            if let Some(ask) = ask {
                match &mut agg_ask {
                    None => agg_ask = Some(ask),
                    Some(aa) if ask.price < aa.price => agg_ask = Some(ask),
                    Some(aa) if ask.price == aa.price => {
                        aa.size += ask.size;
                        aa.count += ask.count;
                    }
                    Some(_) => {}
                }
            }
        }
        (agg_bid, agg_ask)
    }

    pub fn apply(&mut self, mbo: MboMsg) {
        let publisher = mbo.publisher().unwrap();
        let books = self.books.entry(mbo.hd.instrument_id).or_default();
        let book = if let Some((_, book)) = books
            .iter_mut()
            .find(|(book_pub, _)| *book_pub == publisher)
        {
            book
        } else {
            books.push((publisher, Book::default()));
            &mut books.last_mut().unwrap().1
        };
        book.apply(mbo);
    }
}

