


#[derive(Debug)]
pub enum LobTransPatchType {
    Price,
    Volume,
}
impl LobTransPatchType {
    pub fn value(&self) -> f64 {
        match self {
            Self::Price => -1.0,
            Self::Volume => 1.0,
        }
    }
}


#[derive(Debug)]
pub enum LobTransPatchSide {
    Bid,
    Ask,
}
impl LobTransPatchSide {
    pub fn value(&self) -> f64 {
        match self {
            Self::Bid => -1.0,
            Self::Ask => 1.0,
        }
    }
}





