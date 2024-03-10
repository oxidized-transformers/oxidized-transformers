use candle_core::ModuleT;
use candle_nn::{layer_norm, rms_norm, LayerNormConfig as CandleLayerNormConfig, VarBuilder};

use crate::error::BoxedError;
use crate::layers::build_module::BuildModule;

/// Layer norm configuration.
#[derive(Clone, Debug)]
pub struct LayerNormConfig {
    pub affine: bool,
    pub eps: f64,
    pub remove_mean: bool,
    pub size: usize,
}

impl LayerNormConfig {
    /// Whether to use an affine transformation.
    ///
    /// Default: `true`
    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Epsilon value.
    ///
    /// Default: `1e-12`
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Whether to remove the mean.
    ///
    /// If the mean is not removed, this layer is equivalent to `RMSNorm`.
    ///
    /// Default: `true`
    pub fn remove_mean(mut self, remove_mean: bool) -> Self {
        self.remove_mean = remove_mean;
        self
    }

    /// Dimensionality of the layer.
    ///
    /// Default: `768`
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            affine: true,
            eps: 1e-12,
            remove_mean: true,
            size: 768,
        }
    }
}

impl BuildModule for LayerNormConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(layer_norm(
            self.size,
            CandleLayerNormConfig {
                affine: self.affine,
                eps: self.eps,
                remove_mean: self.remove_mean,
            },
            vb,
        )?))
    }
}

/// RMS norm configuration.
#[derive(Clone, Debug)]
pub struct RMSNormConfig {
    pub eps: f64,
    pub size: usize,
}

impl RMSNormConfig {
    /// Epsilon value.
    ///
    /// Default: `1e-12`
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Dimensionality of the layer.
    ///
    /// Default: `768`
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
}

impl Default for RMSNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-12,
            size: 768,
        }
    }
}

impl BuildModule for RMSNormConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        Ok(Box::new(rms_norm(self.size, self.eps, vb)?))
    }
}
