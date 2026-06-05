
use std::error::Error;
use serde::{
    Serialize,
    Deserialize
};

// generic M denotes the format of the messages that are emitted from the extractor
pub trait DataWriter<T> {
    fn write(&mut self, data: &Vec<T>) -> Result<(), Box<dyn Error>> { Ok(()) }
}


pub trait DataReader<T> {
    fn read(&self, index: usize) -> Result<Vec<T>, Box<dyn Error>> { Ok(vec![]) }
}



