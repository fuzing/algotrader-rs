


#[derive(Debug)]
pub enum PriceGainPatchType {
    Price,
    Volume,
}
impl PriceGainPatchType {
    pub fn value(&self) -> f64 {
        match self {
            Self::Price => -1.0,
            Self::Volume => 1.0,
        }
    }
}


#[derive(Debug)]
pub enum PriceGainPatchSide {
    Bid,
    Ask,
}
impl PriceGainPatchSide {
    pub fn value(&self) -> f64 {
        match self {
            Self::Bid => -1.0,
            Self::Ask => 1.0,
        }
    }
}





