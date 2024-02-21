/// Transformer building blocks.
use candle_core::{ModuleT, Tensor};
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::architectures::{BuildDecoderLayer, DecoderLayer};
use crate::architectures::{BuildEncoderLayer, EncoderLayer};
use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::attention::{Attention, AttentionMask, BuildAttention, SelfAttentionConfig};
use crate::layers::build_module::BuildModule;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::identity::Identity;

/// Transformer layer configuration.
#[derive(Debug)]
pub struct TransformerLayerConfig {
    /// Attention residual connection layer norm.
    attn_residual_layer_norm: Box<dyn BuildModule>,

    /// Attention layer configuration.
    attention: SelfAttentionConfig,

    /// Feed-forward layer configuration.
    feedforward: PointwiseFeedForwardConfig,

    /// Feed-forward residual connection layer norm.
    ffn_residual_layer_norm: Box<dyn BuildModule>,

    /// Parallel attention dropout.
    parallel_attn_dropout: Box<dyn BuildModule>,

    /// Use parallel attention.
    use_parallel_attention: bool,
}

impl TransformerLayerConfig {
    /// Generic layer builder.
    fn build_layer(&self, vb: VarBuilder) -> Result<TransformerLayer, TransformerLayerError> {
        Ok(TransformerLayer {
            attn_residual_layer_norm: self
                .attn_residual_layer_norm
                .build(vb.push_prefix("attn_residual_layer_norm"))
                .context(CreateLayerNormSnafu)?,
            ffn: self
                .feedforward
                .build(vb.push_prefix("ffn"))
                .context(BuildPointwiseFeedForwardSnafu)?,
            ffn_residual_layer_norm: self
                .ffn_residual_layer_norm
                .build(vb.push_prefix("ffn_residual_layer_norm"))
                .context(CreateLayerNormSnafu)?,
            mha: self
                .attention
                .build(vb.push_prefix("attention"))
                .context(BuildAttentionSnafu)?,
            parallel_attention_dropout: self
                .parallel_attn_dropout
                .build(vb.push_prefix("parallel_attention_dropout"))
                .context(BuildParallelAttentionDropoutSnafu)?,
            use_parallel_attention: self.use_parallel_attention,
        })
    }

    /// Attention residual connection layer norm.
    ///
    /// Default: `Identity`
    pub fn attn_residual_layer_norm(
        mut self,
        attn_residual_layer_norm: Box<dyn BuildModule>,
    ) -> Self {
        self.attn_residual_layer_norm = attn_residual_layer_norm;
        self
    }

    /// Attention layer configuration.
    ///
    /// Default: `SelfAttentionConfig::default()`
    pub fn attention(mut self, attention: SelfAttentionConfig) -> Self {
        self.attention = attention;
        self
    }

    /// Feed-forward layer configuration.
    ///
    /// Default: `PointwiseFeedForwardConfig::default()`
    pub fn feedforward(mut self, feedforward: PointwiseFeedForwardConfig) -> Self {
        self.feedforward = feedforward;
        self
    }

    /// Feed-forward residual connection layer norm.
    ///
    /// Default: `Identity`
    pub fn ffn_residual_layer_norm(
        mut self,
        ffn_residual_layer_norm: Box<dyn BuildModule>,
    ) -> Self {
        self.ffn_residual_layer_norm = ffn_residual_layer_norm;
        self
    }

    /// Parallel attention dropout.
    ///
    /// Default: `Identity`
    pub fn parallel_attn_dropout(mut self, parallel_attn_dropout: Box<dyn BuildModule>) -> Self {
        self.parallel_attn_dropout = parallel_attn_dropout;
        self
    }

    /// Whether to use parallel attention.
    ///
    /// Default: `false`
    pub fn use_parallel_attention(mut self, use_parallel_attention: bool) -> Self {
        self.use_parallel_attention = use_parallel_attention;
        self
    }
}

impl Default for TransformerLayerConfig {
    fn default() -> Self {
        Self {
            attn_residual_layer_norm: Box::new(Identity),
            attention: SelfAttentionConfig::default(),
            feedforward: PointwiseFeedForwardConfig::default(),
            ffn_residual_layer_norm: Box::new(Identity),
            parallel_attn_dropout: Box::new(Identity),
            use_parallel_attention: false,
        }
    }
}

impl BuildDecoderLayer for TransformerLayerConfig {
    type Cache = KeyValueCache;

    fn build_decoder_layer(
        &self,
        vb: VarBuilder,
    ) -> Result<Box<dyn DecoderLayer<Cache = Self::Cache>>, BoxedError> {
        Ok(Box::new(TransformerDecoderLayer {
            inner: self.build_layer(vb)?,
        }))
    }
}

impl BuildEncoderLayer for TransformerLayerConfig {
    fn build_encoder_layer(&self, vb: VarBuilder) -> Result<Box<dyn EncoderLayer>, BoxedError> {
        Ok(Box::new(TransformerEncoderLayer {
            inner: self.build_layer(vb)?,
        }))
    }
}

/// Errors for transformer layers.
#[derive(Debug, Snafu)]
pub enum TransformerLayerError {
    #[snafu(display("Cannot build attention layer"))]
    BuildAttention { source: BoxedError },

    #[snafu(display("Cannot build parallel attention dropout"))]
    BuildParallelAttentionDropout { source: BoxedError },

    #[snafu(display("Cannot build pointwise feed-forward layer"))]
    BuildPointwiseFeedForward { source: BoxedError },

    #[snafu(display("Cannot create layer norm"))]
    CreateLayerNorm { source: BoxedError },

    #[snafu(display("Cannot apply point-wise feed-forward layer"))]
    FeedForward { source: candle_core::Error },

    #[snafu(display("Cannot apply parallel attention"))]
    ParallelAttention { source: candle_core::Error },

    #[snafu(display("Cannot apply residual connection"))]
    Residual { source: candle_core::Error },

    #[snafu(display("Cannot apply self-attention"))]
    SelfAttention { source: BoxedError },
}

/// Transformer layer.
///
/// This is a generic transformer layer that is used by `DecoderLayer` and
/// `EncoderLayer` to provide specialized layers.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
struct TransformerLayer {
    attn_residual_layer_norm: Box<dyn ModuleT>,
    ffn_residual_layer_norm: Box<dyn ModuleT>,
    mha: Box<dyn Attention>,
    parallel_attention_dropout: Box<dyn ModuleT>,
    ffn: Box<dyn ModuleT>,
    use_parallel_attention: bool,
}

impl TransformerLayer {
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
        cache: Option<&KeyValueCache>,
        _positions: Option<&Tensor>,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<(Tensor, Option<KeyValueCache>), TransformerLayerError> {
        let mut residual = input.clone();

        // Apply attention block.
        let (attn_out, cache) = self
            .mha
            .forward_t(input, attention_mask, cache, train, use_causal_mask)
            .context(SelfAttentionSnafu)?;

        // Apply post-attention residual connection.
        let ffn_in = if self.use_parallel_attention {
            input
        } else {
            residual = (residual + &attn_out)
                .and_then(|xs| self.attn_residual_layer_norm.forward_t(&xs, train))
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
                .and_then(|xs| self.parallel_attention_dropout.forward_t(&xs, train))
                .context(ParallelAttentionSnafu)?
        } else {
            ffn_out
        };

        let output = (residual + output)
            .and_then(|xs| self.ffn_residual_layer_norm.forward_t(&xs, train))
            .context(ResidualSnafu)?;

        Ok((output, cache))
    }
}

/// Transformer decoder layer.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
pub struct TransformerDecoderLayer {
    inner: TransformerLayer,
}

impl DecoderLayer for TransformerDecoderLayer {
    type Cache = KeyValueCache;

    fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        cache: Option<&Self::Cache>,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Self::Cache>), BoxedError> {
        Ok(self
            .inner
            .forward(input, attention_mask, cache, positions, train, true)?)
    }
}

/// Transformer encoder layer.
///
/// See [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
pub struct TransformerEncoderLayer {
    inner: TransformerLayer,
}

impl EncoderLayer for TransformerEncoderLayer {
    fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, BoxedError> {
        self.inner
            .forward(input, attention_mask, None, positions, train, false)
            .map(|(output, _cache)| output)
            .boxed()
    }
}
