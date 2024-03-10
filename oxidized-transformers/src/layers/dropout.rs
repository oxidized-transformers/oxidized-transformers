use candle_core::ModuleT;
use candle_nn::{Dropout, VarBuilder};

use crate::error::BoxedError;
use crate::layers::build_module::BuildModule;

/// Dropout configuration.
#[derive(Clone, Debug)]
pub struct DropoutConfig {
    p: f32,
}

impl DropoutConfig {
    /// Dropout probability.
    ///
    /// Default: `0.0`
    pub fn p(mut self, p: f32) -> Self {
        self.p = p;
        self
    }
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self { p: 0.0 }
    }
}

impl BuildModule for DropoutConfig {
    fn build(&self, _vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(Dropout::new(self.p)))
    }
}
