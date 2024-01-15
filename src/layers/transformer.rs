/// Transformer building blocks.
use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{embedding, Dropout, Embedding, VarBuilder};

use crate::error::Result;
use crate::layers::embeddings::KeyValueCache;
use crate::layers::feedforward::PointwiseFeedForward;
use crate::layers::identity::Identity;
use crate::layers::{AttentionMask, SelfAttention};

/// Layer normalizations used in a transformer embedding layer.
///
/// By default, all the normalizations are disabled by setting the layer
/// normalization to the `Identity` module. Therefore, only normalizations
/// that are needed have to be set.
pub struct EmbeddingLayerNorms {
    /// Normalization of the embeddings.
    pub embed_output_layer_norm: Box<dyn ModuleT>,

    /// Normalization of the projection layer.
    pub proj_output_layer_norm: Box<dyn ModuleT>,
}

impl Default for EmbeddingLayerNorms {
    fn default() -> Self {
        EmbeddingLayerNorms {
            embed_output_layer_norm: Box::new(Identity),
            proj_output_layer_norm: Box::new(Identity),
        }
    }
}

/// Dropouts used in a transformer embedding layer.
///
/// By default, all the dropouts are disabled by setting the dropout
/// to the `Identity` module. Therefore, only dropouts that are needed have
/// to be set.
pub struct EmbeddingLayerDropouts {
    /// Dropout of the embeddings.
    embed_output_layer_dropout: Box<dyn ModuleT>,

    /// Dropout of the projection layer.
    proj_output_layer_dropout: Box<dyn ModuleT>,
}

impl Default for EmbeddingLayerDropouts {
    fn default() -> Self {
        EmbeddingLayerDropouts {
            embed_output_layer_dropout: Box::new(Identity),
            proj_output_layer_dropout: Box::new(Identity),
        }
    }
}

/// Transformer embeddings layer.
///
/// This is a generic transformer embedding layer. The layer always has piece
/// embeddings and can optionally have position embeddings, type embeddings,
/// and a projection of embeddings to the model's hidden size.
pub struct TransformerEmbeddings {
    piece_embeddings: Embedding,
    type_embeddings: Option<Embedding>,
    position_embeddings: Option<Embedding>,
    projection: Option<Embedding>,
    dropouts: EmbeddingLayerDropouts,
    layer_norms: EmbeddingLayerNorms,
}

impl TransformerEmbeddings {
    /// Construct an embeddings layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        dropouts: EmbeddingLayerDropouts,
        embedding_width: usize,
        hidden_width: usize,
        layer_norms: EmbeddingLayerNorms,
        n_pieces: usize,
        n_positions: Option<usize>,
        n_types: Option<usize>,
    ) -> Result<Self> {
        let piece_embeddings = embedding(
            n_pieces,
            embedding_width,
            vb.push_prefix("piece_embeddings"),
        )?;

        let type_embeddings = n_types
            .map(|n_types| embedding(n_types, embedding_width, vb.push_prefix("type_embeddings")))
            .transpose()?;

        let position_embeddings = n_positions
            .map(|n_positions| {
                embedding(
                    n_positions,
                    embedding_width,
                    vb.push_prefix("position_embeddings"),
                )
            })
            .transpose()?;

        let projection = if embedding_width != hidden_width {
            Some(embedding(
                embedding_width,
                hidden_width,
                vb.push_prefix("projection"),
            )?)
        } else {
            None
        };

        Ok(TransformerEmbeddings {
            dropouts,
            layer_norms,
            piece_embeddings,
            position_embeddings,
            projection,
            type_embeddings,
        })
    }

    /// Get position identifiers _[0..seq_len)_.
    fn get_positions(x: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = x.shape().dims2()?;
        Ok(Tensor::arange(0, seq_len as i64, x.device())?.reshape((1, seq_len))?)
    }

    /// Get all-zero type identifiers for the given tensor.
    fn get_type_ids(x: &Tensor) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Calculate the piece embeddings.
    pub fn forward(
        &self,
        piece_ids: &Tensor,
        train: bool,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut embeddings = self.piece_embeddings.forward(piece_ids)?;

        if let Some(type_embeddings) = &self.type_embeddings {
            let type_ids = match type_ids {
                Some(type_ids) => type_ids.clone(),
                None => Self::get_type_ids(piece_ids)?,
            };
            embeddings = (embeddings + type_embeddings.forward(&type_ids)?)?;
        }
        if let Some(position_embeddings) = &self.position_embeddings {
            let positions = match positions {
                Some(positions) => positions.clone(),
                None => Self::get_positions(piece_ids)?,
            };
            embeddings = (embeddings + position_embeddings.forward(&positions)?)?;
        }

        embeddings = self
            .layer_norms
            .embed_output_layer_norm
            .forward_t(&embeddings, train)?;
        embeddings = self
            .dropouts
            .embed_output_layer_dropout
            .forward_t(&embeddings, train)?;

        if let Some(projection) = &self.projection {
            embeddings = projection.forward(&embeddings)?;
            embeddings = self
                .layer_norms
                .proj_output_layer_norm
                .forward_t(&embeddings, train)?;
            embeddings = self
                .dropouts
                .proj_output_layer_dropout
                .forward_t(&embeddings, train)?;
        }

        Ok(embeddings)
    }
}

/// Layer normalizations used in a transformer embedding layer.
///
/// By default, all the normalizations are disabled by setting the layer
/// normalization to the ``Identity`` module. Therefore, only normalizations
/// that are needed have to be set.
pub struct TransformersLayerNorms {
    /// Normalization of the input to the attention layer.
    pub attn_input_layer_norm: Box<dyn ModuleT>,

    /// Normalization of the output to the attention layer after the residual
    /// connection.
    pub attn_residual_layer_norm: Box<dyn ModuleT>,

    /// Normalization of the input to the feed-forward layer.
    pub ffn_input_layer_norm: Box<dyn ModuleT>,

    /// Normalization of the output of the feed-forward layer after the
    /// residual connection.
    pub ffn_residual_layer_norm: Box<dyn ModuleT>,
}

impl Default for TransformersLayerNorms {
    fn default() -> Self {
        TransformersLayerNorms {
            attn_input_layer_norm: Box::new(Identity),
            attn_residual_layer_norm: Box::new(Identity),
            ffn_input_layer_norm: Box::new(Identity),
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
    /// Dropout of the output of the attention layer.
    pub attn_output_dropout: Box<dyn ModuleT>,

    /// Dropout of the output of the feed-forward layer.
    pub ffn_output_dropout: Box<dyn ModuleT>,

    /// Dropout after summing the attention and feed-forward layers.
    /// Only used when parallel attention is enabled.
    pub parallel_attn_dropout: Box<dyn ModuleT>,
}

impl Default for TransformerDropouts {
    fn default() -> Self {
        TransformerDropouts {
            attn_output_dropout: Box::new(Identity),
            ffn_output_dropout: Box::new(Identity),
            parallel_attn_dropout: Box::new(Identity),
        }
    }
}

impl TransformerDropouts {
    /// Attention and feed-forward layer dropouts.
    ///
    /// * `p` - Dropout probability.
    pub fn layer_output_dropouts(&self, p: f32) -> Self {
        TransformerDropouts {
            attn_output_dropout: Box::new(Dropout::new(p)),
            ffn_output_dropout: Box::new(Dropout::new(p)),
            parallel_attn_dropout: Box::new(Identity),
        }
    }

    /// Parallel attention dropout.
    ///
    /// - `p` - Dropout probability.
    pub fn parallel_attn_dropout(&self, p: f32) -> Self {
        TransformerDropouts {
            attn_output_dropout: Box::new(Identity),
            ffn_output_dropout: Box::new(Identity),
            parallel_attn_dropout: Box::new(Dropout::new(p)),
        }
    }
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
    ) -> Result<(Tensor, Option<KeyValueCache>)> {
        let mut residual = input.clone();

        let mut attn_out = self.mha.forward(
            &self.norms.attn_input_layer_norm.forward_t(input, train)?,
            attention_mask,
            use_causal_mask,
        )?;
        attn_out = self
            .dropouts
            .attn_output_dropout
            .forward_t(&attn_out, train)?;

        let ffn_in = if self.use_parallel_attention {
            input
        } else {
            residual = self
                .norms
                .attn_residual_layer_norm
                .forward_t(&(residual + &attn_out)?, train)?;
            &residual
        };

        let mut ffn_out = self
            .ffn
            .forward(&self.norms.ffn_input_layer_norm.forward_t(ffn_in, train)?)?;
        ffn_out = self
            .dropouts
            .ffn_output_dropout
            .forward_t(&ffn_out, train)?;

        let output = if self.use_parallel_attention {
            self.dropouts
                .parallel_attn_dropout
                .forward_t(&(attn_out + ffn_out)?, train)?
        } else {
            ffn_out
        };

        Ok((
            self.norms
                .ffn_residual_layer_norm
                .forward_t(&(residual + output)?, train)?,
            // TODO: cache, once wired up in self-attention.
            None,
        ))
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
    ) -> Result<(Tensor, Option<KeyValueCache>)> {
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
    ) -> Result<(Tensor, Option<KeyValueCache>)> {
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
