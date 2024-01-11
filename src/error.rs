use std::ops::RangeInclusive;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, TransformerError>;

#[derive(Debug, Error)]
pub enum TransformerError {
    #[error("candle error")]
    CandleError(#[from] candle_core::Error),

    #[error("{msg}, range: {range:?}, got: {value}")]
    InclusiveRangeError {
        value: f32,
        range: RangeInclusive<f32>,
        msg: &'static str,
    },

    #[error("{msg}, dimension must be multiple of {multiple}, was {value}")]
    MultipleError {
        multiple: usize,
        value: usize,
        msg: &'static str,
    },
}
