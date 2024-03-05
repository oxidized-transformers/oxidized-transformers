use candle_core::ModuleT;
use candle_nn::{Activation as CandleActivation, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::build_module::BuildModule;

/// Activation functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Gausian Error Linear Unit.
    ///
    /// See [Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415).
    Gelu,

    /// Gausian Error Linear Unit approximation.
    ///
    /// See [Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415).
    GeluNew,

    /// Rectified Linear Unit.
    ///
    /// See [Fukushima, 1969](https://ieeexplore.ieee.org/document/4082265).
    Relu,

    /// Sigmoid Linear Unit.
    ///
    /// See [Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415).
    Silu,
}

impl BuildModule for Activation {
    fn build(&self, _vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        use Activation::*;
        Ok(match self {
            Gelu => Box::new(CandleActivation::Gelu),
            GeluNew => Box::new(CandleActivation::NewGelu),
            Relu => Box::new(CandleActivation::Relu),
            Silu => Box::new(CandleActivation::Silu),
        })
    }
}
