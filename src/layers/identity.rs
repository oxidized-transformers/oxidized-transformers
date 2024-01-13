use candle_core::{Module, Tensor};

pub struct Identity;

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(xs.clone())
    }
}
