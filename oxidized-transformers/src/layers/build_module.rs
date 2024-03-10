use std::fmt::Debug;

use candle_core::ModuleT;
use candle_nn::VarBuilder;

use crate::error::BoxedError;

/// Traits for types that can build modules.
pub trait BuildModule: Debug {
    /// Build a module.
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError>;
}
