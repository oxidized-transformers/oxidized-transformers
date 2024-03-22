use std::fmt::Debug;

use candle_core::Tensor;
use candle_nn::VarBuilder;

mod mask;
pub use mask::{AttentionMask, AttentionMaskError};

mod sdpa;
pub use sdpa::{with_sdpa_implementation, SDPAConfig, SDPAError, SDPAImplementation, SDPA};

mod alibi;
pub use alibi::{AttentionLinearBiases, AttentionLinearBiasesConfig, AttentionLinearBiasesError};

mod self_attention;
pub use self_attention::{
    AttentionHeads, CausalMask, CausalMaskError, QkvMode, QkvSplit, SelfAttention,
    SelfAttentionConfig, SelfAttentionMask, SelfAttentionMaskError,
};

use crate::error::BoxedError;
use crate::kv_cache::LayerKeyValueCache;

/// Trait for attention modules.
pub trait Attention {
    /// Apply attention to the given input.
    ///
    /// * `input` - Input tensor.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which
    ///   the corresponding mask element is set to `false` are ignored in attention.
    /// * `train` - Whether the model is trained.
    /// * `use_causal_mask` - Whether to use a causal mask.
    ///
    /// Returns: Hidden representations after attention.
    fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        cache: &mut LayerKeyValueCache,
        positions: Option<&Tensor>,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<Tensor, BoxedError>;
}

/// Build an attention module.
pub trait BuildAttention {
    /// Build an attention module.
    ///
    /// * `vb` - Variable builder used for attention parameters.
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Attention>, BoxedError>;
}

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
    /// * `use_causal_mask` - Whether to apply a causal mask. With a causal mask,
    ///   a sequence element can only attend to preceding elements and itself.
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
        use_causal_mask: bool,
    ) -> Result<Tensor, BoxedError>;
}

/// Build an attention scorer module.
pub trait BuildAttentionScorer: Debug {
    /// Build an attention module.
    ///
    /// * `vb` - Variable builder used for attention parameters.
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn AttentionScorer>, BoxedError>;
}
