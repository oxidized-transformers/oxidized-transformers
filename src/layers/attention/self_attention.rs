use std::borrow::Cow;

use candle_core::{DType, IndexOp, Module, ModuleT, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::attention::{
    Attention, AttentionMask, AttentionMaskError, AttentionScorer, BuildAttention,
    BuildAttentionScorer, ScaledDotProductAttentionConfig, ScaledDotProductAttentionError,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QkvSplit {
    Default,
    KVSizedChunks,
}

/// How the query, key and value projections are handled in
/// the self-attention layer.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QkvMode {
    Separate,
    MergedSplitBefore,
    MergedSplitAfter(QkvSplit),
}

/// Representation of the query, key and value tensors.
pub enum QkvTensors {
    Merged(Linear),
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
        let linear_ctor = if self.use_bias {
            linear
        } else {
            linear_no_bias
        };

        let head_width = self.hidden_width / self.attention_heads.n_query_heads;
        let key_value_width = self.attention_heads.n_key_value_heads * head_width;
        let output = linear_ctor(
            self.hidden_width,
            self.hidden_width,
            vb.push_prefix("output"),
        )
        .context(SelfAttentionConstructionSnafu)?;
        let qkv = match self.attention_heads.qkv_mode {
            QkvMode::MergedSplitBefore | QkvMode::MergedSplitAfter(_) => QkvTensors::Merged(
                linear_ctor(
                    self.hidden_width,
                    self.hidden_width + 2 * key_value_width,
                    vb.push_prefix("qkv"),
                )
                .context(SelfAttentionConstructionSnafu)?,
            ),
            QkvMode::Separate => QkvTensors::Separate {
                query: linear_ctor(
                    self.hidden_width,
                    self.hidden_width,
                    vb.push_prefix("query"),
                )
                .context(SelfAttentionConstructionSnafu)?,
                key: linear_ctor(self.hidden_width, key_value_width, vb.push_prefix("key"))
                    .context(SelfAttentionConstructionSnafu)?,
                value: linear_ctor(self.hidden_width, key_value_width, vb.push_prefix("value"))
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
    AttentionMask { source: AttentionMaskError },

    #[snafu(display("Cannot apply attention scorer"))]
    AttentionScorer { source: BoxedError },

    #[snafu(display("Cannot create causal mask"))]
    CausalMask { source: candle_core::Error },

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

    #[snafu(display("Cannot apply layer norm"))]
    LayerNorm { source: candle_core::Error },

    #[snafu(display("Cannot apply output layer"))]
    Output { source: candle_core::Error },

    #[snafu(display("Cannot calculate key, query, or value"))]
    Qkv { source: candle_core::Error },

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
        train: bool,
        use_causal_mask: bool,
    ) -> Result<Tensor, BoxedError> {
        let input = self
            .layer_norm
            .forward_t(input, train)
            .context(LayerNormSnafu)?;

        let (mut query, mut key, value) = match &self.qkv {
            QkvTensors::Separate { query, key, value } => {
                let query = query
                    .forward(&input)
                    .context(QkvSnafu)?
                    .split_heads(self.attention_heads.n_query_heads)?;
                let key = key
                    .forward(&input)
                    .context(QkvSnafu)?
                    .split_heads(self.attention_heads.n_key_value_heads)?;
                let value = value
                    .forward(&input)
                    .context(QkvSnafu)?
                    .split_heads(self.attention_heads.n_key_value_heads)?;
                (query, key, value)
            }
            _ => unimplemented!(),
        };

        if let Some(rotary_embeds) = &self.rotary_embeds {
            let (query_rot, key_rot) = rotary_embeds
                .forward(&query, &key, None, None)
                .context(RotaryEmbeddingsSnafu)?;
            query = query_rot;
            key = key_rot;
        }

        // TODO: kv cache

        // TODO: causal mask

        // TODO: ALiBi

        let combined_mask = if use_causal_mask {
            let causal_mask = create_causal_mask(&query, &key)?;
            Cow::Owned(
                attention_mask
                    .intersect(&causal_mask)
                    .context(IntersectMasksSnafu)?,
            )
        } else {
            Cow::Borrowed(attention_mask)
        };

        let attn = self
            .attention_scorer
            .forward(&query, &key, &value, &combined_mask, train)
            .context(AttentionScorerSnafu)?
            .combine_heads()?;

        Ok(self
            .output
            .forward(&attn)
            .and_then(|xs| self.dropout.forward_t(&xs, train))
            .context(OutputSnafu)?)
    }
}

/// Create a causal mask.
///
/// A causal mask ensures that tokens cannot attend to succeeding tokens.
fn create_causal_mask(query: &Tensor, key: &Tensor) -> Result<AttentionMask, SelfAttentionError> {
    let (_, _, query_len, _) = query.shape().dims4().context(CausalMaskSnafu)?;
    let (_, _, key_len, _) = key.shape().dims4().context(CausalMaskSnafu)?;

    let causal_mask = Tensor::tril2(key_len, DType::U32, key.device())
        .and_then(|mask| mask.reshape((1, 1, key_len, key_len)))
        .context(CausalMaskSnafu)?;
    AttentionMask::new(
        causal_mask
            .i((.., .., key_len - query_len..key_len, ..key_len))
            .context(CausalMaskSnafu)?,
    )
    .context(AttentionMaskSnafu)
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
