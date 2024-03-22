use std::cell::Cell;

use candle_core::{ModuleT, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::attention::self_attention::{SelfAttentionMask, SelfAttentionMaskError};
use crate::layers::attention::{
    AttentionLinearBiases, AttentionLinearBiasesConfig, AttentionLinearBiasesError, AttentionMask,
    AttentionScorer, BuildAttentionScorer, CausalMaskError,
};
use crate::layers::attention::{AttentionMaskError, CausalMask};
use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;
use crate::ops::nonzero::NonzeroError;

#[cfg(feature = "flash-attn")]
use candle_core::DType;

/// Scaled dot product attention implementation.
#[derive(Debug, Clone, Copy)]
pub enum SDPAImplementation {
    /// Default implementation.
    Default,

    /// Flash attention.
    ///
    /// Flash attention will only be used if the model is on a CUDA device
    /// and the input is of type `BF16` or `F16`.
    Flash,
}

thread_local! {
    static SDPA_IMPLEMENTATION: Cell<SDPAImplementation> = const { Cell::new(SDPAImplementation::Default) };
}

/// Run a closure with a specific scaled dot-product attention implementation.
pub fn with_sdpa_implementation<T>(implementation: SDPAImplementation, f: impl FnOnce() -> T) -> T {
    SDPA_IMPLEMENTATION.with(|sdpa_implementation| {
        let prev = sdpa_implementation.replace(implementation);
        let result = f();
        sdpa_implementation.replace(prev);
        result
    })
}

/// Configuration for scaled dot-product attention.
#[derive(Debug)]
pub struct SDPAConfig {
    dropout: Box<dyn BuildModule>,
    linear_biases: Option<AttentionLinearBiasesConfig>,
}

impl SDPAConfig {
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

impl Default for SDPAConfig {
    fn default() -> Self {
        Self {
            dropout: Box::new(Identity),
            linear_biases: None,
        }
    }
}

impl BuildAttentionScorer for SDPAConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn AttentionScorer>, BoxedError> {
        Ok(Box::new(SDPA {
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
pub enum SDPAError {
    #[snafu(display("Cannot calculate attention linear biases"))]
    AttentionLinearBiases { source: AttentionLinearBiasesError },

    #[snafu(display("Cannot apply attention mask"))]
    AttentionMask { source: AttentionMaskError },

    #[snafu(display("Cannot calculate attention scores"))]
    AttentionScores { source: candle_core::Error },

    #[snafu(display("Cannot weigh representations using attention mask"))]
    AttentionWeight { source: candle_core::Error },

    #[snafu(display("Cannot build attention linear biases"))]
    BuildAttentionLinearBiases { source: AttentionLinearBiasesError },

    #[snafu(display("Cannot build dropout module"))]
    BuildDropout { source: BoxedError },

    #[snafu(display("Cannot create causal mask"))]
    CausalMask { source: CausalMaskError },

    #[snafu(display("Cannot apply dropout"))]
    Dropout { source: candle_core::Error },

    #[snafu(display("Cannot apply flash attention"))]
    FlashAttention { source: candle_core::Error },

    #[snafu(display("Cannot calculate indices of tokens that are not masked"))]
    NonzeroIndices { source: NonzeroError },

    #[snafu(display("Cannot pad heads"))]
    PadHeads { source: candle_core::Error },

    #[snafu(display("Cannot update self-attention mask"))]
    SelfAttentionMask { source: SelfAttentionMaskError },

    #[snafu(display("Cannot calculate sequence lengths"))]
    SeqLens { source: candle_core::Error },

    #[snafu(display("Cannot apply softmax temperature"))]
    Temperature { source: candle_core::Error },

    #[snafu(display("Cannot unpad heads"))]
    UnpadHeads { source: candle_core::Error },
}

/// Scaled dot-product attention.
///
/// See [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
pub struct SDPA {
    dropout: Box<dyn ModuleT>,
    linear_biases: Option<AttentionLinearBiases>,
}

impl AttentionScorer for SDPA {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<Tensor, BoxedError> {
        let output = match SDPA_IMPLEMENTATION.get() {
            #[cfg(feature = "flash-attn")]
            SDPAImplementation::Flash if Self::is_flash_attention_supported(key) => self
                .forward_flash_attn(query, key, value, attention_mask, use_causal_mask)
                .boxed(),
            _ => self.forward_default(query, key, value, attention_mask, use_causal_mask),
        }?;
        self.dropout
            .forward_t(&output, train)
            .context(DropoutSnafu)
            .boxed()
    }
}

impl SDPA {
    /// Check whether flash attention is supported.
    #[cfg(feature = "flash-attn")]
    fn is_flash_attention_supported(xs: &Tensor) -> bool {
        xs.device().is_cuda() && (xs.dtype() == DType::BF16 || xs.dtype() == DType::F16)
    }

    /// Default/reference implementation.
    ///
    /// For docs see the `AttentionScorer` trait.
    fn forward_default(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        use_causal_mask: bool,
    ) -> Result<Tensor, BoxedError> {
        // Calculate attention scores.
        let query = query.contiguous().context(AttentionScoresSnafu)?;
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

        let mut combined_mask: SelfAttentionMask = attention_mask.into();
        if use_causal_mask {
            let causal_mask =
                SelfAttentionMask::causal_mask(&query, key).context(CausalMaskSnafu)?;
            combined_mask = combined_mask
                .intersect(&causal_mask)
                .context(SelfAttentionMaskSnafu)?;
        }

        attn_scores = combined_mask
            .apply_logit_mask(&attn_scores)
            .context(SelfAttentionMaskSnafu)?;

        // Apply attention weights.
        let attn_weights = softmax(&attn_scores, D::Minus1).context(AttentionWeightSnafu)?;
        value
            .contiguous()
            .and_then(|value| attn_weights.broadcast_matmul(&value))
            .context(AttentionWeightSnafu)
            .boxed()
    }

    /// Flash attention implementation.
    ///
    /// For docs see the `AttentionScorer` trait.
    #[cfg(feature = "flash-attn")]
    fn forward_flash_attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        use_causal_mask: bool,
    ) -> Result<Tensor, SDPAError> {
        use candle_flash_attn::{flash_attn_varlen, flash_attn_varlen_alibi};

        let (_batch_size, _kv_heads, key_value_len, head_width) =
            key.dims4().context(FlashAttentionSnafu)?;
        let (_batch_size, _q_heads, query_len, _query_width) =
            query.dims4().context(FlashAttentionSnafu)?;

        let softmax_scale = 1.0 / (head_width as f32).sqrt();

        let (key_unpad, key_indices) = Self::unpad_heads(key, attention_mask, None)?;
        let (value_unpad, _) = Self::unpad_heads(value, attention_mask, Some(&key_indices))?;
        let query_mask = AttentionMask::new(
            attention_mask
                .bool_mask
                .narrow(1, key_value_len - query_len, query_len)
                .context(FlashAttentionSnafu)?,
        )
        .context(AttentionMaskSnafu)?;
        let query_indices = if query_len == key_value_len {
            Some(&key_indices)
        } else {
            None
        };
        let (query_unpad, query_indices) = Self::unpad_heads(query, &query_mask, query_indices)?;

        let key_value_lens = Self::seq_lens_flash_attn(attention_mask)?;
        let query_lens = if query_len == key_value_len {
            key_value_lens.clone()
        } else {
            Self::seq_lens_flash_attn(&query_mask)?
        };

        let query_max_len = query_lens
            .max(0)
            .and_then(|maxlen| maxlen.to_scalar::<u32>())
            .context(FlashAttentionSnafu)? as usize;
        let key_value_max_len = key_value_lens
            .max(0)
            .and_then(|maxlen| maxlen.to_scalar::<u32>())
            .context(FlashAttentionSnafu)? as usize;

        let output_unpad = match &self.linear_biases {
            Some(linear_biases) => flash_attn_varlen_alibi(
                &query_unpad,
                &key_unpad,
                &value_unpad,
                &linear_biases
                    .slopes()
                    .reshape(((),))
                    .context(FlashAttentionSnafu)?,
                &query_lens,
                &key_value_lens,
                query_max_len,
                key_value_max_len,
                softmax_scale,
                use_causal_mask,
            ),
            None => flash_attn_varlen(
                &query_unpad,
                &key_unpad,
                &value_unpad,
                &query_lens,
                &key_value_lens,
                query_max_len,
                key_value_max_len,
                softmax_scale,
                use_causal_mask,
            ),
        }
        .context(FlashAttentionSnafu)?;

        Self::pad_heads(&output_unpad, &query_mask, &query_indices)
    }

    /// Compute the sequence lengths from the attention masks.
    ///
    /// This ignores any pieces that are masked out.
    #[cfg(feature = "flash-attn")]
    fn seq_lens(attention_mask: &AttentionMask) -> Result<Tensor, SDPAError> {
        attention_mask
            .bool_mask()
            // Mask is U8, which will overflow for longer sequences.
            .to_dtype(DType::U32)
            .and_then(|mask| mask.sum(D::Minus1))
            .context(SeqLensSnafu)
    }

    /// Compute the sequence lengths for flash attention.
    ///
    /// This method is similar to [`Self::seq_lens`], but prepares the
    /// sequence lengths for the flash attention implementation:
    ///
    /// * Compute the cumulative sum.
    /// * Prepend a zero to cumulative sum.
    ///
    /// As a result, the tensor can be used as starting/ending indices
    /// into the tensor that stores the sequences.
    #[cfg(feature = "flash-attn")]
    fn seq_lens_flash_attn(attention_mask: &AttentionMask) -> Result<Tensor, SDPAError> {
        let seq_lens = Self::seq_lens(attention_mask)?
            // Candle cumsum currently only works on floating point types.
            .to_dtype(DType::F32)
            .and_then(|seq_lens| seq_lens.cumsum(0))
            .and_then(|seq_lens| seq_lens.to_dtype(DType::U32))
            .context(SeqLensSnafu)?;

        // Prepend a zero to the sequence lengths.
        Tensor::zeros(&[1], seq_lens.dtype(), seq_lens.device())
            .and_then(|zeros| Tensor::cat(&[&zeros, &seq_lens], 0))
            .context(SeqLensSnafu)
    }

    /// Add padding to the head representations.
    ///
    /// This helper function takes an input of the shape `(n_pieces, n_heads,
    /// hidden_size)` and transforms it into a representation with shape
    /// `(batch_size, n_heads, seq_len, hidden_size)` by padding all sequences
    /// to the same length using the provided indices.
    #[cfg(feature = "flash-attn")]
    fn pad_heads(
        heads: &Tensor,
        attention_mask: &AttentionMask,
        indices: &Tensor,
    ) -> Result<Tensor, SDPAError> {
        let (batch_size, seq_len) = attention_mask.bool_mask().dims2().context(PadHeadsSnafu)?;
        let (n_indices, n_heads, hidden_width) = heads.dims3().context(PadHeadsSnafu)?;

        let expanded_indices = indices
            .expand((n_indices, n_heads, hidden_width))
            .and_then(|indices| indices.contiguous())
            .context(PadHeadsSnafu)?;

        Tensor::zeros(
            &[batch_size * seq_len, n_heads, hidden_width],
            heads.dtype(),
            heads.device(),
        )
        .and_then(|zeros| zeros.scatter_add(&expanded_indices, &heads, 0))
        // Restore original, non-flash-attention shape.
        .and_then(|padded| padded.reshape((batch_size, seq_len, n_heads, hidden_width)))
        .and_then(|padded| padded.transpose(1, 2))
        .context(PadHeadsSnafu)
    }

    /// Remove padding from the head representations.
    ///
    /// This helper function takes an input of the shape `(batch_size, n_heads,
    /// seq_len, hidden_size)` and transforms it into a representation with
    /// shape `(n_pieces, n_heads, hidden_size)`, where all padding pieces are
    /// removed using the attention mask.
    ///
    /// The indices can be provided optionally to avoid recomputing the indices
    /// from the attention mask.
    #[cfg(feature = "flash-attn")]
    fn unpad_heads(
        heads: &Tensor,
        attention_mask: &AttentionMask,
        indices: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor), SDPAError> {
        use crate::ops::nonzero::Nonzero;

        let (batch_size, n_heads, seq_len, hidden_width) =
            heads.dims4().context(UnpadHeadsSnafu)?;

        let indices = match indices {
            Some(indices) => indices.clone(),
            None => attention_mask
                .bool_mask()
                .reshape(((),))
                .context(UnpadHeadsSnafu)?
                .nonzero()
                .context(NonzeroIndicesSnafu)?
                .reshape(((), 1, 1))
                .context(UnpadHeadsSnafu)?,
        };

        let n_indices = indices.dim(0).context(UnpadHeadsSnafu)?;

        let expanded_indices = indices
            .expand((n_indices, n_heads, hidden_width))
            .and_then(|indices| indices.contiguous())
            .context(UnpadHeadsSnafu)?;

        Ok((
            heads
                .transpose(1, 2)
                .and_then(|heads| heads.reshape((batch_size * seq_len, n_heads, hidden_width)))
                .and_then(|heads| heads.gather(&expanded_indices, 0))
                .context(UnpadHeadsSnafu)?,
            indices,
        ))
    }
}
