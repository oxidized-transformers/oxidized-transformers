use std::error::Error;

use candle_core::Tensor;

mod mask;
pub use mask::{AttentionMask, AttentionMaskError};

mod sdpa;
pub use sdpa::{ScaledDotProductAttention, ScaledDotProductAttentionError};

mod alibi;
pub use alibi::{AttentionLinearBiases, AttentionLinearBiasesError};

mod self_attention;
pub use self_attention::{QkvMode, SelfAttention, SelfAttentionError};

/// Trait implemented by modules that perform attention scoring.
pub trait AttentionScorer {
    /// Apply attention scores to the given key, query and value.
    /// Sequence elements that are marked with `false` in the attention mask
    /// are ignored by the attention mechanism (if a mask is provided).
    ///
    /// * `query` - Query tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * `key` - Key tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * `value` - Value tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which
    ///   the corresponding mask element is set to `false` are ignored in attention.
    /// * `train` - Whether the model is trained.
    ///
    /// Returns: Attention values.
    /// *Shape:* `(batch_size, heads, seq_len, width)`
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
        train: bool,
    ) -> Result<Tensor, Box<dyn Error + Send + Sync>>;
}
