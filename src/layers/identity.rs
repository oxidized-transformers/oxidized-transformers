use candle_core::{Module, ModuleT, Tensor};
use candle_nn::VarBuilder;

use crate::error::BoxedError;
use crate::layers::build_module::BuildModule;

/// Identity module.
///
/// This module passes through input as-is. It is especially useful in
/// cases where a configurable module (such as dropout or normalization)
/// needs to be stubbed with a module that does not do anything.
#[derive(Clone, Debug)]
pub struct Identity;

impl BuildModule for Identity {
    fn build(&self, _vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(Identity))
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(xs.clone())
    }
}
