use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use snafu::{ensure, ResultExt, Snafu};

use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::attention::{
    Attention, AttentionMask, AttentionMaskError, AttentionScorer, BuildAttention,
    BuildAttentionScorer, CausalMask, CausalMaskError, QueryKeyAttentionMask,
    QueryKeyAttentionMaskError, ScaledDotProductAttentionConfig, ScaledDotProductAttentionError,
};
use crate::layers::build_module::BuildModule;
use crate::layers::embeddings::{
    QueryKeyRotaryEmbeddings, QueryKeyRotaryEmbeddingsConfig, QueryKeyRotaryEmbeddingsError,
};
use crate::layers::identity::Identity;

/// Attention heads configuration to use in self-attention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionHeads {
    pub n_query_heads: usize,
    pub n_key_value_heads: usize,
    pub qkv_mode: QkvMode,
}

/// Query, key, value splitting strategies.
///
/// After the input projection of the attention layer, we have an array with
/// shape `(batch_size, seq_len, n_heads * head_width)` where `n_heads` is
/// the sum of the number of query, key, and value heads. We need to split up
/// the array into separate arrays for query, key, and value heads.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QkvSplit {
    Default,
    KVSizedChunks,
}

/// How the query, key and value projections are handled in
/// the self-attention layer.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QkvMode {
    Separate,
    MergedSplitBefore,
    MergedSplitAfter(QkvSplit),
}

/// Representation of the query, key and value tensors.
pub enum QkvTensors {
    MergedSplitBefore(Linear),
    MergedSplitAfter {
        qkv: Linear,
        split: QkvSplit,
    },
    Separate {
        query: Linear,
        key: Linear,
        value: Linear,
    },
}

/// Configuration for self-attention modules.
#[derive(Debug)]
pub struct SelfAttentionConfig {
    /// Attention head configuration.
    attention_heads: AttentionHeads,

    /// Attention scorer.
    attention_scorer: Box<dyn BuildAttentionScorer>,

    /// Dropout probability to apply after attention.
    dropout: Box<dyn BuildModule>,

    /// Hidden width of the transformer.
    hidden_width: usize,

    /// Layer norm.
    layer_norm: Box<dyn BuildModule>,

    /// Rotary embedding configuration.
    rotary_embeddings: Option<QueryKeyRotaryEmbeddingsConfig>,

    /// Use bias in linear layers.
    use_bias: bool,

    /// Use parallel attention.
    use_parallel_attention: bool,
}

impl SelfAttentionConfig {
    /// Attention head configuration.
    pub fn attention_heads(mut self, attention_heads: AttentionHeads) -> Self {
        self.attention_heads = attention_heads;
        self
    }

    /// Attention scorer.
    ///
    /// Default: `ScaledDotProductAttentionConfig::default()`.
    pub fn attention_scorer(mut self, attention_scorer: Box<dyn BuildAttentionScorer>) -> Self {
        self.attention_scorer = attention_scorer;
        self
    }

    /// Dropout to apply after attention.
    ///
    /// Default: `Identity`.
    pub fn dropout(mut self, dropout: Box<dyn BuildModule>) -> Self {
        self.dropout = dropout;
        self
    }

    /// Hidden width of the transformer.
    ///
    /// Default: `768`.
    pub fn hidden_width(mut self, hidden_width: usize) -> Self {
        self.hidden_width = hidden_width;
        self
    }

    /// Layer norm applied to the input.
    ///
    /// Default: `Identity`.
    pub fn layer_norm(mut self, layer_norm: Box<dyn BuildModule>) -> Self {
        self.layer_norm = layer_norm;
        self
    }

    /// Configuration for rotary embeddings.
    ///
    /// Default: `None`.
    pub fn rotary_embeddings(
        mut self,
        rotary_embeddings: Option<QueryKeyRotaryEmbeddingsConfig>,
    ) -> Self {
        self.rotary_embeddings = rotary_embeddings;
        self
    }

    /// Use bias in linear layers.
    ///
    /// Default: `false`.
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Use parallel attention.
    ///
    /// Default: `false`.
    pub fn use_parallel_attention(mut self, use_parallel_attention: bool) -> Self {
        self.use_parallel_attention = use_parallel_attention;
        self
    }
}

impl Default for SelfAttentionConfig {
    fn default() -> Self {
        Self {
            attention_heads: AttentionHeads {
                n_query_heads: 12,
                n_key_value_heads: 12,
                qkv_mode: QkvMode::Separate,
            },
            attention_scorer: Box::<ScaledDotProductAttentionConfig>::default(),
            dropout: Box::new(Identity),
            hidden_width: 768,
            layer_norm: Box::new(Identity),
            rotary_embeddings: None,
            use_bias: false,
            use_parallel_attention: false,
        }
    }
}

impl BuildAttention for SelfAttentionConfig {
    /// Build a self-attention module.
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Attention>, BoxedError> {
        let hidden_width = self.hidden_width;
        let n_key_value_heads = self.attention_heads.n_key_value_heads;
        let n_query_heads = self.attention_heads.n_query_heads;

        ensure!(
            hidden_width % n_query_heads == 0,
            InvalidNQueryHeadsSnafu {
                hidden_width,
                n_query_heads,
            }
        );

        if self.attention_heads.qkv_mode == QkvMode::MergedSplitBefore {
            ensure!(
                n_query_heads == n_key_value_heads,
                InvalidMergedSplitBeforeNHeadsSnafu {
                    n_key_value_heads,
                    n_query_heads
                }
            )
        }

        let linear_ctor = if self.use_bias {
            linear
        } else {
            linear_no_bias
        };

        let head_width = hidden_width / n_query_heads;
        let key_value_width = n_key_value_heads * head_width;
        let output = linear_ctor(hidden_width, hidden_width, vb.push_prefix("output"))
            .context(SelfAttentionConstructionSnafu)?;
        let qkv = match self.attention_heads.qkv_mode {
            QkvMode::MergedSplitBefore => QkvTensors::MergedSplitBefore(
                linear_ctor(
                    hidden_width,
                    hidden_width + 2 * key_value_width,
                    vb.push_prefix("qkv"),
                )
                .context(SelfAttentionConstructionSnafu)?,
            ),
            QkvMode::MergedSplitAfter(split) => QkvTensors::MergedSplitAfter {
                qkv: linear_ctor(
                    hidden_width,
                    hidden_width + 2 * key_value_width,
                    vb.push_prefix("qkv"),
                )
                .context(SelfAttentionConstructionSnafu)?,
                split,
            },
            QkvMode::Separate => QkvTensors::Separate {
                query: linear_ctor(hidden_width, hidden_width, vb.push_prefix("query"))
                    .context(SelfAttentionConstructionSnafu)?,
                key: linear_ctor(hidden_width, key_value_width, vb.push_prefix("key"))
                    .context(SelfAttentionConstructionSnafu)?,
                value: linear_ctor(hidden_width, key_value_width, vb.push_prefix("value"))
                    .context(SelfAttentionConstructionSnafu)?,
            },
        };

        Ok(Box::new(SelfAttention {
            attention_scorer: self
                .attention_scorer
                .build(vb.clone())
                .context(BuildAttentionScorerSnafu)?,
            attention_heads: self.attention_heads.clone(),
            dropout: self
                .dropout
                .build(vb.push_prefix("dropout"))
                .context(BuildDropoutSnafu)?,
            layer_norm: self
                .layer_norm
                .build(vb.push_prefix("layer_norm"))
                .context(BuildLayerNormSnafu)?,
            output,
            qkv,
            rotary_embeds: self
                .rotary_embeddings
                .as_ref()
                .map(|config| config.build(vb))
                .transpose()
                .context(BuildQueryKeyRotaryEmbeddingsSnafu)?,
        }))
    }
}

/// Errors for self-attention.
#[derive(Debug, Snafu)]
pub enum SelfAttentionError {
    #[snafu(display("Cannot create or apply attention mask"))]
    AttentionMask { source: QueryKeyAttentionMaskError },

    #[snafu(display("Cannot apply attention scorer"))]
    AttentionScorer { source: BoxedError },

    #[snafu(display("Cannot create causal mask"))]
    CausalMask { source: CausalMaskError },

    #[snafu(display("Cannot build attention scorer"))]
    BuildAttentionScorer { source: BoxedError },

    #[snafu(display("Cannot build dropout"))]
    BuildDropout { source: BoxedError },

    #[snafu(display("Cannot build layer norm"))]
    BuildLayerNorm { source: BoxedError },

    #[snafu(display("Cannot build query-key rotary embeddings"))]
    BuildQueryKeyRotaryEmbeddings {
        source: QueryKeyRotaryEmbeddingsError,
    },

    #[snafu(display("Cannot combine heads"))]
    CombineHeads { source: candle_core::Error },

    #[snafu(display("Cannot intersect attention and causal mask"))]
    IntersectMasks { source: AttentionMaskError },

    #[snafu(display(
        "The hidden size ({hidden_width}) must be divisble by the number of query heads ({n_query_heads})"
    ))]
    InvalidNQueryHeads {
        hidden_width: usize,
        n_query_heads: usize,
    },

    #[snafu(display(
        "The number of query ({n_query_heads}) and key-value ({n_key_value_heads}) heads \
         must be equal when using merged QKV matrix and splitting before projection"
    ))]
    InvalidMergedSplitBeforeNHeads {
        n_key_value_heads: usize,
        n_query_heads: usize,
    },

    #[snafu(display("Cannot concatenate append key and value to cache"))]
    ConcatKVCache { source: candle_core::Error },

    #[snafu(display("Cannot apply layer norm"))]
    LayerNorm { source: candle_core::Error },

    #[snafu(display("Cannot apply output layer"))]
    Output { source: candle_core::Error },

    #[snafu(display("Cannot calculate query, key, or value"))]
    Qkv { source: candle_core::Error },

    #[snafu(display("Cannot chunk query, key, and value representations"))]
    QkvChunk { source: candle_core::Error },

    #[snafu(display("Cannot apply rotary embeddings"))]
    RotaryEmbeddings {
        source: QueryKeyRotaryEmbeddingsError,
    },

    #[snafu(display("Cannot apply the scaled dot product attention"))]
    ScaledDotProductAttention {
        source: ScaledDotProductAttentionError,
    },

    #[snafu(display("Cannot construct layer"))]
    SelfAttentionConstruction { source: candle_core::Error },

    #[snafu(display("Cannot split heads"))]
    SplitHeads { source: candle_core::Error },
}

/// Transformer self-attention layer.
///
/// See [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
pub struct SelfAttention {
    attention_scorer: Box<dyn AttentionScorer>,
    attention_heads: AttentionHeads,
    dropout: Box<dyn ModuleT>,
    layer_norm: Box<dyn ModuleT>,
    output: Linear,
    qkv: QkvTensors,
    rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
}

impl Attention for SelfAttention {
    fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        cache: Option<&KeyValueCache>,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<(Tensor, Option<KeyValueCache>), BoxedError> {
        let input = self
            .layer_norm
            .forward_t(input, train)
            .context(LayerNormSnafu)?;

        let (mut query, mut key, value) = match &self.qkv {
            QkvTensors::Separate { query, key, value } => {
                self.query_key_value_separate(value, query, key, &input)?
            }
            QkvTensors::MergedSplitAfter { .. } => todo!(),
            QkvTensors::MergedSplitBefore(qkv) => self.query_key_value_split_before(qkv, &input)?,
        };

        if let Some(rotary_embeds) = &self.rotary_embeds {
            let (query_rot, key_rot) = rotary_embeds
                .forward(&query, &key, cache, None)
                .context(RotaryEmbeddingsSnafu)?;
            query = query_rot;
            key = key_rot;
        }

        let (key, value, attention_mask) = match cache {
            Some(cache) => (
                // Shape is: [batch_size, n_heads, key_len, head_width], so
                // concat along dimension 2.
                Tensor::cat(&[&cache.key, &key], 2).context(ConcatKVCacheSnafu)?,
                Tensor::cat(&[&cache.value, &value], 2).context(ConcatKVCacheSnafu)?,
                cache
                    .attention_mask
                    .extend(attention_mask)
                    .context(IntersectMasksSnafu)?,
            ),
            None => (key, value, attention_mask.clone()),
        };

        // TODO: rotary embeddings positions

        // TODO: causal mask

        // TODO: ALiBi

        let mut combined_mask: QueryKeyAttentionMask = (&attention_mask).into();
        if use_causal_mask {
            let causal_mask =
                QueryKeyAttentionMask::causal_mask(&query, &key).context(CausalMaskSnafu)?;
            combined_mask = combined_mask
                .intersect(&causal_mask)
                .context(AttentionMaskSnafu)?;
        }

        let attn = self
            .attention_scorer
            .forward(&query, &key, &value, &combined_mask, train)
            .context(AttentionScorerSnafu)?
            .combine_heads()?;

        let output = self
            .output
            .forward(&attn)
            .and_then(|xs| self.dropout.forward_t(&xs, train))
            .context(OutputSnafu)?;
        let cache = cache.map(|_| KeyValueCache {
            key,
            value,
            attention_mask,
        });

        Ok((output, cache))
    }
}

impl SelfAttention {
    /// Compute query, key, and value with separate projections.
    fn query_key_value_separate(
        &self,
        value: &Linear,
        query: &Linear,
        key: &Linear,
        input: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), BoxedError> {
        let query = query
            .forward(input)
            .context(QkvSnafu)?
            .split_heads(self.attention_heads.n_query_heads)?;
        let key = key
            .forward(input)
            .context(QkvSnafu)?
            .split_heads(self.attention_heads.n_key_value_heads)?;
        let value = value
            .forward(input)
            .context(QkvSnafu)?
            .split_heads(self.attention_heads.n_key_value_heads)?;
        Ok((query, key, value))
    }

    /// Compute query, key, and value with a single projection.
    ///
    /// Split heads before query, key, and value.
    fn query_key_value_split_before(
        &self,
        qkv: &Linear,
        input: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), BoxedError> {
        let proj = qkv
            .forward(input)
            .context(QkvSnafu)?
            .split_heads(self.attention_heads.n_query_heads)?;

        let (_, _, _, all_heads_size) = proj.shape().dims4().context(QkvSnafu)?;
        let head_size = all_heads_size / 3;

        // Similar to chunk, but avoid intermediate Vec. Needs to be contiguous or
        // matrix multiplication fails later.
        let query = proj.narrow(3, 0, head_size).context(QkvChunkSnafu)?;
        let key = proj
            .narrow(3, head_size, head_size)
            .context(QkvChunkSnafu)?;
        let value = proj
            .narrow(3, 2 * head_size, head_size)
            // Needs to be contiguous to avoid matmul stride error.
            .and_then(|xs| xs.contiguous())
            .context(QkvChunkSnafu)?;

        Ok((query, key, value))
    }
}

trait CombineHeads {
    fn combine_heads(&self) -> Result<Tensor, SelfAttentionError>;
}

impl CombineHeads for Tensor {
    fn combine_heads(&self) -> Result<Tensor, SelfAttentionError> {
        let (batch_size, n_heads, seq_len, model_width) =
            self.dims4().context(CombineHeadsSnafu)?;
        self.transpose(1, 2)
            .and_then(|heads| heads.reshape((batch_size, seq_len, n_heads * model_width)))
            .context(CombineHeadsSnafu)
    }
}

trait SplitHeads {
    fn split_heads(&self, n_heads: usize) -> Result<Tensor, SelfAttentionError>;
}

impl SplitHeads for Tensor {
    fn split_heads(&self, n_heads: usize) -> Result<Tensor, SelfAttentionError> {
        let (batch_size, seq_len, model_width) = self.dims3().context(SplitHeadsSnafu)?;
        let head_width = model_width / n_heads;
        self.reshape((batch_size, seq_len, n_heads, head_width))
            .and_then(|heads| heads.transpose(1, 2))
            .context(SplitHeadsSnafu)
    }
}
