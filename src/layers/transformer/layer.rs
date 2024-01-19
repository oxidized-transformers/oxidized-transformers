/// Transformer building blocks.
use candle_core::{ModuleT, Tensor};
use candle_nn::Dropout;
use snafu::{ResultExt, Snafu};

use crate::kv_cache::KeyValueCache;
use crate::layers::attention::{AttentionMask, SelfAttention, SelfAttentionError};
use crate::layers::feedforward::PointwiseFeedForward;
use crate::layers::identity::Identity;

/// Layer normalizations used in a transformer embedding layer.
///
/// By default, all the normalizations are disabled by setting the layer
/// normalization to the ``Identity`` module. Therefore, only normalizations
/// that are needed have to be set.
pub struct TransformersLayerNorms {
    /// Normalization of the output to the attention layer after the residual
    /// connection.
    pub attn_residual_layer_norm: Box<dyn ModuleT>,

    /// Normalization of the output of the feed-forward layer after the
    /// residual connection.
    pub ffn_residual_layer_norm: Box<dyn ModuleT>,
}

impl Default for TransformersLayerNorms {
    fn default() -> Self {
        TransformersLayerNorms {
            attn_residual_layer_norm: Box::new(Identity),
            ffn_residual_layer_norm: Box::new(Identity),
        }
    }
}

/// Dropouts used in a transformer layer.
///
/// By default, all the dropouts are disabled by setting the dropout
/// to the Torch `Identity` module. Therefore, only dropouts that are
/// needed have to be set.
pub struct TransformerDropouts {
    /// Dropout after summing the attention and feed-forward layers.
    /// Only used when parallel attention is enabled.
    pub parallel_attn_dropout: Box<dyn ModuleT>,
}

impl Default for TransformerDropouts {
    fn default() -> Self {
        TransformerDropouts {
            parallel_attn_dropout: Box::new(Identity),
        }
    }
}

impl TransformerDropouts {
    /// Parallel attention dropout.
    ///
    /// - `p` - Dropout probability.
    pub fn parallel_attn_dropout(&self, p: f32) -> Self {
        TransformerDropouts {
            parallel_attn_dropout: Box::new(Dropout::new(p)),
        }
    }
}

/// Errors for transformer layers.
#[derive(Debug, Snafu)]
pub enum TransformerLayerError {
    #[snafu(display("Cannot apply point-wise feed-forward layer"))]
    FeedForward { source: candle_core::Error },

    #[snafu(display("Cannot apply parallel attention"))]
    ParallelAttention { source: candle_core::Error },

    #[snafu(display("Cannot apply residual connection"))]
    Residual { source: candle_core::Error },

    #[snafu(display("Cannot apply self-attention"))]
    SelfAttention { source: SelfAttentionError },
}

/// Transformer layer.
///
/// This is a generic transformer layer that is used by `DecoderLayer` and
/// `EncoderLayer` to provide specialized layers.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
struct TransformerLayer {
    use_parallel_attention: bool,
    mha: SelfAttention,
    ffn: PointwiseFeedForward,
    dropouts: TransformerDropouts,
    norms: TransformersLayerNorms,
}

impl TransformerLayer {
    /// Construct a transformer layer.
    ///
    /// * `attention_layer` - The attention layer to use in the transformer
    ///   layers.
    /// * `dropouts` - The dropouts to use in the transformer layers.
    /// * `feed_forward_layer` - The feed-forward layer to use in the
    ///   transformer layers.
    /// * `layer_norms` - The layer norms to use in the transformer
    /// * `use_parallel_attention` - Use parallel attention.
    fn new(
        attention_layer: SelfAttention,
        dropouts: TransformerDropouts,
        feed_forward_layer: PointwiseFeedForward,
        layer_norms: TransformersLayerNorms,
        use_parallel_attention: bool,
    ) -> Self {
        Self {
            dropouts,
            ffn: feed_forward_layer,
            mha: attention_layer,
            norms: layer_norms,
            use_parallel_attention,
        }
    }

    /// Apply the transformer layer to the given piece hidden representations.
    ///
    /// * `input` - Hidden representations to apply the layer to.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///    corresponding mask element is set to `false` are ignored
    ///    during attention calculation.
    /// * `cache` - Key/value cache to avoid recomputing key/value representations
    ///    for tokens that were previously seen.
    /// * `positions` - Input positions. Positions are needed to look up rotary
    ///    embeddings. Normally, these positions are calculated automatically.
    ///    But if the positions deviate for some reason, they can be provided
    ///    through this argument.
    ///    *Shape:* `(batch_size, seq_len)`
    /// * `store_cache` - Whether to cache the key/value representations for
    ///   future reuse.
    /// * `train` - Whether to train the layer.
    /// * `use_causal_mask` - Mask out succeeding sequence elements when `true`.
    ///
    /// Returns layer output and the key/value cache.
    /// *Shape:* ``(batch_size, seq_len, width)``
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        _cache: Option<&KeyValueCache>,
        _positions: Option<&Tensor>,
        _store_cache: bool,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<(Tensor, Option<KeyValueCache>), TransformerLayerError> {
        let mut residual = input.clone();

        // Apply attention block.
        let attn_out = self
            .mha
            .forward(input, attention_mask, train, use_causal_mask)
            .context(SelfAttentionSnafu)?;

        // Apply post-attention residual connection.
        let ffn_in = if self.use_parallel_attention {
            input
        } else {
            residual = (residual + &attn_out)
                .and_then(|xs| self.norms.attn_residual_layer_norm.forward_t(&xs, train))
                .context(ResidualSnafu)?;
            &residual
        };

        // Apply feed-forward block.
        let ffn_out = self
            .ffn
            .forward_t(ffn_in, train)
            .context(FeedForwardSnafu)?;

        // Apply parallel attention.
        let output = if self.use_parallel_attention {
            (attn_out + ffn_out)
                .and_then(|xs| self.dropouts.parallel_attn_dropout.forward_t(&xs, train))
                .context(ParallelAttentionSnafu)?
        } else {
            ffn_out
        };

        let output = (residual + output)
            .and_then(|xs| self.norms.ffn_residual_layer_norm.forward_t(&xs, train))
            .context(ResidualSnafu)?;

        Ok((output, None))
    }
}

/// Transformer decoder layer.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
pub struct DecoderLayer {
    inner: TransformerLayer,
}

impl DecoderLayer {
    /// Construct a decoder layer.
    ///
    /// * `attention_layer` - The attention layer to use in the transformer
    ///   layers.
    /// * `dropouts` - The dropouts to use in the transformer layers.
    /// * `feed_forward_layer` - The feed-forward layer to use in the
    ///   transformer layers.
    /// * `layer_norms` - The layer norms to use in the transformer
    /// * `use_parallel_attention` - Use parallel attention.
    pub fn new(
        attention_layer: SelfAttention,
        dropouts: TransformerDropouts,
        feed_forward_layer: PointwiseFeedForward,
        layer_norms: TransformersLayerNorms,
        use_parallel_attention: bool,
    ) -> Self {
        Self {
            inner: TransformerLayer::new(
                attention_layer,
                dropouts,
                feed_forward_layer,
                layer_norms,
                use_parallel_attention,
            ),
        }
    }

    /// Apply the transformer layer to the given piece hidden representations.
    ///
    /// * `input` - Hidden representations to apply the layer to.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///    corresponding mask element is set to `false` are ignored
    ///    during attention calculation.
    /// * `cache` - Key/value cache to avoid recomputing key/value representations
    ///    for tokens that were previously seen.
    /// * `positions` - Input positions. Positions are needed to look up rotary
    ///    embeddings. Normally, these positions are calculated automatically.
    ///    But if the positions deviate for some reason, they can be provided
    ///    through this argument.
    ///    *Shape:* `(batch_size, seq_len)`
    /// * `store_cache` - Whether to cache the key/value representations for
    ///   future reuse.
    /// * `train` - Whether to train the layer.
    ///
    /// Returns layer output and the key/value cache.
    /// *Shape:* ``(batch_size, seq_len, width)``
    pub fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        cache: Option<&KeyValueCache>,
        positions: Option<&Tensor>,
        store_cache: bool,
        train: bool,
    ) -> Result<(Tensor, Option<KeyValueCache>), TransformerLayerError> {
        self.inner.forward(
            input,
            attention_mask,
            cache,
            positions,
            store_cache,
            train,
            true,
        )
    }
}

/// Transformer ecoder layer.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
pub struct EncoderLayer {
    inner: TransformerLayer,
}

impl EncoderLayer {
    /// Construct an encoder layer.
    ///
    /// * `attention_layer` - The attention layer to use in the transformer
    ///   layers.
    /// * `dropouts` - The dropouts to use in the transformer layers.
    /// * `feed_forward_layer` - The feed-forward layer to use in the
    ///   transformer layers.
    /// * `layer_norms` - The layer norms to use in the transformer
    /// * `use_parallel_attention` - Use parallel attention.
    pub fn new(
        attention_layer: SelfAttention,
        dropouts: TransformerDropouts,
        feed_forward_layer: PointwiseFeedForward,
        layer_norms: TransformersLayerNorms,
        use_parallel_attention: bool,
    ) -> Self {
        Self {
            inner: TransformerLayer::new(
                attention_layer,
                dropouts,
                feed_forward_layer,
                layer_norms,
                use_parallel_attention,
            ),
        }
    }

    /// Apply the transformer layer to the given piece hidden representations.
    ///
    /// * `input` - Hidden representations to apply the layer to.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///    corresponding mask element is set to `false` are ignored
    ///    during attention calculation.
    /// * `cache` - Key/value cache to avoid recomputing key/value representations
    ///    for tokens that were previously seen.
    /// * `positions` - Input positions. Positions are needed to look up rotary
    ///    embeddings. Normally, these positions are calculated automatically.
    ///    But if the positions deviate for some reason, they can be provided
    ///    through this argument.
    ///    *Shape:* `(batch_size, seq_len)`
    /// * `store_cache` - Whether to cache the key/value representations for
    ///   future reuse.
    /// * `train` - Whether to train the layer.
    ///
    /// Returns layer output and the key/value cache.
    /// *Shape:* ``(batch_size, seq_len, width)``
    pub fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        cache: Option<&KeyValueCache>,
        positions: Option<&Tensor>,
        store_cache: bool,
        train: bool,
    ) -> Result<(Tensor, Option<KeyValueCache>), TransformerLayerError> {
        self.inner.forward(
            input,
            attention_mask,
            cache,
            positions,
            store_cache,
            train,
            false,
        )
    }
}
