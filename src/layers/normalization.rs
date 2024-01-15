use candle_core::{DType, Module, Tensor};
use candle_nn::VarBuilder;

use crate::error::Result;

/// Root Mean Square (RMS) normalization.
///
/// See [Zhang et al., 2019](https://arxiv.org/abs/1910.07467).
#[derive(Debug)]
pub struct RMSNorm {
    epsilon: f32,
    weight: Tensor,
}

impl RMSNorm {
    /// Construct a RMS normalization module.
    ///
    /// * `vb` - Variable store.
    /// * `width` - The (hidden) width of the representations that
    ///   RMS normalization will be applied to.
    /// * `epsilon` - Epsilon to avoid division by zero.
    pub fn new(vb: VarBuilder, width: usize, epsilon: f32) -> Result<Self> {
        let vb = vb.push_prefix("rms_norm");
        let weight = vb.get_with_hints((width,), "weight", candle_nn::init::ONE)?;
        Ok(Self { epsilon, weight })
    }
}

impl Module for RMSNorm {
    /// Apply RMS normalization to a tensor.
    ///
    /// * xs - The tensor to apply RMS normalization to.
    ///
    /// Returns - Normalized tensor.
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Zhang & Sennrich, Equation 4. If we are in lower precision than
        // float32, then squaring and averaging can get way off. So for
        // normalization, we want to use higher precision.
        let rms = xs
            .to_dtype(DType::F32)?
            .sqr()?
            .mean_keepdim(())?
            .broadcast_add(&Tensor::new(self.epsilon, xs.device())?)?
            .sqrt()?
            .recip()?;

        xs.broadcast_mul(&rms)?
            .to_dtype(xs.dtype())?
            .broadcast_mul(&self.weight)
    }
}
