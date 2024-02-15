use candle_core::Tensor;
use candle_nn::ops::softmax;
use candle_nn::Dropout;
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::attention::{
    AttentionLinearBiases, AttentionLinearBiasesError, AttentionMask, AttentionMaskError,
    AttentionScorer,
};

/// Errors for scaled dot-product attention.
#[derive(Debug, Snafu)]
pub enum ScaledDotProductAttentionError {
    #[snafu(display("Cannot calculate attention linear biases"))]
    AttentionLinearBiases { source: AttentionLinearBiasesError },

    #[snafu(display("Cannot calculate attention scores"))]
    AttentionScores { source: candle_core::Error },

    #[snafu(display("Cannot apply attention mask"))]
    AttentionMask { source: AttentionMaskError },

    #[snafu(display("Cannot weigh representations using attention mask"))]
    AttentionWeight { source: candle_core::Error },

    #[snafu(display("Cannot apply softmax temperature"))]
    Temperature { source: candle_core::Error },
}

/// Scaled dot-product attention.
///
/// See [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
pub struct ScaledDotProductAttention {
    dropout: Dropout,
    linear_biases: Option<AttentionLinearBiases>,
}

impl ScaledDotProductAttention {
    /// Construct a scaled dot-product attention module.
    ///
    /// * dropout - Dropout to apply to the final hidden representation.
    /// * linear_biases - ALiBi for attention scores. Not applied if `None`.
    pub fn new(dropout: Dropout, linear_biases: Option<AttentionLinearBiases>) -> Self {
        ScaledDotProductAttention {
            dropout,
            linear_biases,
        }
    }
}

impl AttentionScorer for ScaledDotProductAttention {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        train: bool,
    ) -> Result<Tensor, BoxedError> {
        // TODO: add code path for flash attention, but verify the attention
        //       layer is working first...

        // Calculate attention scores.
        let mut attn_scores = key
            .transpose(3, 2)
            .and_then(|xs| query.matmul(&xs))
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
        attn_weights
            .matmul(value)
            .and_then(|xs| self.dropout.forward(&xs, train))
            .context(AttentionWeightSnafu)
            .boxed()
    }
}
