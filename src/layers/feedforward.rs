/// Feed-forward layers.
use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;

/// Configuration for pointwise feed-forward layers.
#[derive(Debug)]
pub struct PointwiseFeedForwardConfig {
    activation: Box<dyn BuildModule>,
    dropout: Box<dyn BuildModule>,
    hidden_width: usize,
    intermediate_width: usize,
    layer_norm: Box<dyn BuildModule>,
    use_bias: bool,
    use_gate: bool,
}

impl PointwiseFeedForwardConfig {
    /// Activation function.
    ///
    /// Default: `GELU`
    pub fn activation(mut self, activation: Box<dyn BuildModule>) -> Self {
        self.activation = activation;
        self
    }

    /// Dropout to apply after the feed-forward layer.
    ///
    /// Default: `Identity`
    pub fn dropout(mut self, dropout: Box<dyn BuildModule>) -> Self {
        self.dropout = dropout;
        self
    }

    /// Hidden width of the transformer.
    ///
    /// Default: `768`
    pub fn hidden_width(mut self, hidden_width: usize) -> Self {
        self.hidden_width = hidden_width;
        self
    }

    /// Intermediate width in the feed-forward layer.
    ///
    /// Default: `3072`
    pub fn intermediate_width(mut self, intermediate_width: usize) -> Self {
        self.intermediate_width = intermediate_width;
        self
    }

    /// Layer normalization module.
    ///
    /// Default: `Identity`
    pub fn layer_norm(mut self, layer_norm: Box<dyn BuildModule>) -> Self {
        self.layer_norm = layer_norm;
        self
    }

    /// Use bias in linear layers.
    ///
    /// Default: `true`
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Use Gated Linear Units.
    ///
    /// Default: `false`
    pub fn use_gate(mut self, use_gate: bool) -> Self {
        self.use_gate = use_gate;
        self
    }
}

impl Default for PointwiseFeedForwardConfig {
    fn default() -> Self {
        Self {
            activation: Box::new(Activation::Gelu),
            dropout: Box::new(Identity),
            hidden_width: 768,
            intermediate_width: 3072,
            layer_norm: Box::new(Identity),
            use_bias: true,
            use_gate: false,
        }
    }
}

impl BuildModule for PointwiseFeedForwardConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn ModuleT>, BoxedError> {
        let linear_ctor = if self.use_bias {
            linear
        } else {
            linear_no_bias
        };

        let intermediate = linear_ctor(
            self.hidden_width,
            self.intermediate_width,
            vb.push_prefix("intermediate"),
        )
        .context(CreateLinearSnafu)?;

        let gate = if self.use_gate {
            Some(
                linear_ctor(
                    self.hidden_width,
                    self.intermediate_width,
                    vb.push_prefix("gate"),
                )
                .context(CreateLinearSnafu)?,
            )
        } else {
            None
        };

        let output = linear_ctor(
            self.intermediate_width,
            self.hidden_width,
            vb.push_prefix("output"),
        )
        .context(CreateLinearSnafu)?;

        Ok(Box::new(PointwiseFeedForward {
            activation: self
                .activation
                .build(vb.clone())
                .context(BuildActivationSnafu)?,
            dropout: self
                .dropout
                .build(vb.push_prefix("dropout"))
                .context(BuildDropoutSnafu)?,
            gate,
            intermediate,
            layer_norm: self
                .layer_norm
                .build(vb.push_prefix("layer_norm"))
                .context(CreateLayerNormSnafu)?,
            output,
        }))
    }
}

#[derive(Debug, Snafu)]
pub enum PointwiseFeedForwardError {
    #[snafu(display("Cannot build activation"))]
    BuildActivation { source: BoxedError },

    #[snafu(display("Cannot build dropout"))]
    BuildDropout { source: BoxedError },

    #[snafu(display("Cannot create layer norm"))]
    CreateLayerNorm { source: BoxedError },

    #[snafu(display("Cannot create linear layer"))]
    CreateLinear { source: candle_core::Error },
}

/// Point-wise feed-forward layer (_Vaswani et al., 2017_).
///
/// This layer is applied pointwise, meaning that the same
/// transformation is applied to each sequence element. This
/// transformation is:
///
/// `g(xW_1 + b_1)W_2 + b_2`
///
/// `W_1` and `b_1` transform the input to an
/// intermediate width, `g` is a non-linear activation
/// function and `W_2` and `b_2` transform the
/// output of the activation back to the input width.
///
/// Gated Linear Units (_Dauphin et al., 2016_; _Shazeer, 2020_) are also
/// supported. Gating applies the following transformation:
///
/// `(g(xW_g + b_g) * (xW_1 + b_1))W_2 + b_2`
///
/// `W_g` and _b_g_ are the affine transformation for the gate.
///
/// * _Vaswani et al., 2017_: https://arxiv.org/abs/1706.03762
/// * _Dauphin et al., 2016_: https://arxiv.org/abs/1612.08083
/// * _Shazeer, 2020_: https://arxiv.org/abs/2002.05202
pub struct PointwiseFeedForward {
    activation: Box<dyn ModuleT>,
    dropout: Box<dyn ModuleT>,
    gate: Option<Linear>,
    intermediate: Linear,
    layer_norm: Box<dyn ModuleT>,
    output: Linear,
}

impl ModuleT for PointwiseFeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor, candle_core::Error> {
        let xs = self.layer_norm.forward_t(xs, train)?;

        let output = match &self.gate {
            Some(gate) => self.output.forward(
                &self
                    .activation
                    .forward_t(&gate.forward(&xs)?, train)?
                    .mul(&self.intermediate.forward(&xs)?)?,
            ),
            None => self.output.forward(
                &self
                    .activation
                    .forward_t(&self.intermediate.forward(&xs)?, train)?,
            ),
        }?;

        self.dropout.forward_t(&output, train)
    }
}
