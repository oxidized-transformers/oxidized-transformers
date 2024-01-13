use candle_core::{Module, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::error::Result;

/// Feed-forward layers.

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
    pub activation: Box<dyn Module>,
    pub gate: Option<Linear>,
    pub intermediate: Linear,
    pub output: Linear,
}

impl PointwiseFeedForward {
    /// Construct a point-wise feed-forward layer.
    ///
    /// * `vb` - Variable store.
    /// * `activation` - Non-linearity.
    /// * `hidden_width` - Hidden width, dimensionality of the layer input and output.
    /// * `intermediate_width` - Intermediate width inside the feed-forward layer.
    /// * `use_bias` - Use bias in linear transformations.
    /// * `use_gate` - Use Gated Linear Units.
    pub fn new(
        vb: VarBuilder,
        activation: Box<dyn Module>,
        hidden_width: usize,
        intermediate_width: usize,
        use_bias: bool,
        use_gate: bool,
    ) -> Result<Self> {
        let linear_ctor = if use_bias { linear } else { linear_no_bias };

        let intermediate = linear_ctor(
            hidden_width,
            intermediate_width,
            vb.push_prefix("intermediate"),
        )?;

        let gate = if use_gate {
            Some(linear_ctor(
                hidden_width,
                intermediate_width,
                vb.push_prefix("gate"),
            )?)
        } else {
            None
        };

        let output = linear_ctor(intermediate_width, hidden_width, vb.push_prefix("output"))?;

        Ok(Self {
            activation,
            gate,
            intermediate,
            output,
        })
    }
}

impl Module for PointwiseFeedForward {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match &self.gate {
            Some(gate) => self.output.forward(
                &self
                    .activation
                    .forward(&gate.forward(xs)?)?
                    .mul(&self.intermediate.forward(xs)?)?,
            ),
            None => self
                .output
                .forward(&self.activation.forward(&self.intermediate.forward(xs)?)?),
        }
    }
}
