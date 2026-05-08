//
// // use errors;
// use anyhow;
// // use quickfix::QuickFixError;
// use thiserror;
//
// pub type Result<T, E = Error> = core::result::Result<T, E>;
//
// #[derive(Debug, thiserror::Error)]
// pub enum Error {
//     #[error("quickfix: {0}")]
//     QuickFix(#[from] QuickFixError),
//     #[error("Missing field: id={0}")]
//     MissingField(i32),
//     #[error("invalid message type")]
//     InvalidMessageType,
//     #[error("invalid order type")]
//     InvalidOrderType,
//     #[error("kernel: {0}")]
//     Internal(String),
//     #[error("IO({0}) error")]
//     IO(String),
//     #[error("General({0}) error")]
//     General(String),
//     #[error(transparent)]
//     Error(#[from] anyhow::Error),
// }
//
// impl From<Error> for errors::Error {
//     fn from(value: Error) -> Self {
//         match value {
//             err @ Error::Internal(_) => errors::Error::Internal(err.to_string()),
//             err @ Error::QuickFix(_) => errors::Error::NotFound(err.to_string()),
//             // err @ Error::MissingField(i32) => errors::Error::NotFound(err.to_string()),
//             err @ Error::MissingField(_) => errors::Error::NotFound(err.to_string()),
//             err @ Error::InvalidMessageType => errors::Error::NotFound(err.to_string()),
//             err @ Error::InvalidOrderType => errors::Error::NotFound(err.to_string()),
//             err @ Error::IO(_) => errors::Error::IO(err.to_string()),
//             err @ Error::General(_) => errors::Error::General(err.to_string()),
//             err @ Error::Error(_) => errors::Error::General(err.to_string()),
//         }
//     }
// }