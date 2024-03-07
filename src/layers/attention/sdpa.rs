use candle_core::{ModuleT, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::attention::self_attention::{SelfAttentionMask, SelfAttentionMaskError};
use crate::layers::attention::{
    AttentionLinearBiases, AttentionLinearBiasesConfig, AttentionLinearBiasesError,
    AttentionScorer, BuildAttentionScorer,
};
use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;

/// Configuration for scaled dot-product attention.
#[derive(Debug)]
pub struct ScaledDotProductAttentionConfig {
    dropout: Box<dyn BuildModule>,
    linear_biases: Option<AttentionLinearBiasesConfig>,
}

impl ScaledDotProductAttentionConfig {
    /// Dropout to apply after attention.
    ///
    /// Default: `Identity`.
    pub fn dropout(mut self, dropout: Box<dyn BuildModule>) -> Self {
        self.dropout = dropout;
        self
    }

    /// ALiBi for attention scores.
    ///
    /// Default: `None`.
    pub fn linear_biases(mut self, linear_biases: Option<AttentionLinearBiasesConfig>) -> Self {
        self.linear_biases = linear_biases;
        self
    }
}

impl Default for ScaledDotProductAttentionConfig {
    fn default() -> Self {
        Self {
            dropout: Box::new(Identity),
            linear_biases: None,
        }
    }
}

impl BuildAttentionScorer for ScaledDotProductAttentionConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn AttentionScorer>, BoxedError> {
        Ok(Box::new(ScaledDotProductAttention {
            dropout: self.dropout.build(vb.clone()).context(BuildDropoutSnafu)?,
            linear_biases: self
                .linear_biases
                .as_ref()
                .map(|linear_biases| linear_biases.build())
                .transpose()
                .context(BuildAttentionLinearBiasesSnafu)?,
        }))
    }
}

/// Errors for scaled dot-product attention.
#[derive(Debug, Snafu)]
pub enum ScaledDotProductAttentionError {
    #[snafu(display("Cannot calculate attention linear biases"))]
    AttentionLinearBiases { source: AttentionLinearBiasesError },

    #[snafu(display("Cannot calculate attention scores"))]
    AttentionScores { source: candle_core::Error },

    #[snafu(display("Cannot apply attention mask"))]
    AttentionMask { source: SelfAttentionMaskError },

    #[snafu(display("Cannot weigh representations using attention mask"))]
    AttentionWeight { source: candle_core::Error },

    #[snafu(display("Cannot build attention linear biases"))]
    BuildAttentionLinearBiases { source: AttentionLinearBiasesError },

    #[snafu(display("Cannot build dropout module"))]
    BuildDropout { source: BoxedError },

    #[snafu(display("Cannot apply softmax temperature"))]
    Temperature { source: candle_core::Error },
}

/// Scaled dot-product attention.
///
/// See [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
pub struct ScaledDotProductAttention {
    dropout: Box<dyn ModuleT>,
    linear_biases: Option<AttentionLinearBiases>,
}

impl AttentionScorer for ScaledDotProductAttention {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &SelfAttentionMask,
        train: bool,
    ) -> Result<Tensor, BoxedError> {
        // TODO: add code path for flash attention, but verify the attention
        //       layer is working first...

        // Calculate attention scores.
        let mut attn_scores = key
            .contiguous()
            .and_then(|key| key.transpose(3, 2))
            .and_then(|key| query.broadcast_matmul(&key))
            .context(AttentionScoresSnafu)?;

        let model_width = key.dim(3).context(TemperatureSnafu)?;
        let temperature = (model_width as f64).sqrt();
        attn_scores = (attn_scores / temperature).context(TemperatureSnafu)?;

        // Apply ALiBi.
        if let Some(linear_biases) = &self.linear_biases {
            attn_scores = linear_biases
                .forward(&attn_scores)
                .context(AttentionLinearBiasesSnafu)?;
        }

        attn_scores = attention_mask
            .apply_logit_mask(&attn_scores)
            .context(AttentionMaskSnafu)?;

        // Apply attention weights.
        let attn_weights = softmax(&attn_scores, 3).context(AttentionWeightSnafu)?;
        value
            .contiguous()
            .and_then(|value| attn_weights.broadcast_matmul(&value))
            .and_then(|xs| self.dropout.forward_t(&xs, train))
            .context(AttentionWeightSnafu)
            .boxed()
    }
}
