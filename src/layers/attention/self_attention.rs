use std::borrow::Cow;

use candle_core::{IndexOp, Module, ModuleT, Tensor};
use candle_nn::{linear, Dropout, Linear, VarBuilder};
use snafu::{ResultExt, Snafu};

use crate::error::BoxedError;
use crate::layers::attention::{
    AttentionMask, AttentionMaskError, AttentionScorer, ScaledDotProductAttentionError,
};
use crate::layers::embeddings::{QueryKeyRotaryEmbeddings, QueryKeyRotaryEmbeddingsError};

/// Attention heads configuration to use in self-attention.
pub struct AttentionHeads {
    n_query_heads: usize,
    n_key_value_heads: usize,
    qkv_mode: QkvMode,
}

/// Query, key, value splitting strategies.
///
/// After the input projection of the attention layer, we have an array with
/// shape `(batch_size, seq_len, n_heads * head_width)` where `n_heads` is
/// the sum of the number of query, key, and value heads. We need to split up
/// the array into separate arrays for query, key, and value heads.
#[non_exhaustive]
pub enum QkvSplit {
    Default,
    KVSizedChunks,
}

/// How the query, key and value projections are handled in
/// the self-attention layer.
#[non_exhaustive]
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

/// Errors for self-attention.
#[derive(Debug, Snafu)]
pub enum SelfAttentionError {
    #[snafu(display("Cannot apply attention scorer"))]
    AttentionScorer { source: BoxedError },

    #[snafu(display("Cannot create causal mask"))]
    CausalMask { source: candle_core::Error },

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
    dropout: Dropout,
    layer_norm: Box<dyn ModuleT>,
    output: Linear,
    qkv: QkvTensors,
    rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
}

impl SelfAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        attention_heads: AttentionHeads,
        attention_scorer: Box<dyn AttentionScorer>,
        dropout: Dropout,
        hidden_width: usize,
        layer_norm: Box<dyn ModuleT>,
        rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
        _use_bias: bool,
    ) -> Result<Self, SelfAttentionError> {
        // TODO: use_bias
        let head_width = hidden_width / attention_heads.n_key_value_heads;
        let key_value_width = attention_heads.n_key_value_heads * head_width;
        let output = linear(hidden_width, hidden_width, vb.push_prefix("output"))
            .context(SelfAttentionConstructionSnafu)?;
        let qkv = match attention_heads.qkv_mode {
            QkvMode::MergedSplitBefore | QkvMode::MergedSplitAfter(_) => QkvTensors::Separate {
                query: linear(hidden_width, hidden_width, vb.push_prefix("query"))
                    .context(SelfAttentionConstructionSnafu)?,
                key: linear(hidden_width, key_value_width, vb.push_prefix("key"))
                    .context(SelfAttentionConstructionSnafu)?,
                value: linear(hidden_width, key_value_width, vb.push_prefix("value"))
                    .context(SelfAttentionConstructionSnafu)?,
            },
            QkvMode::Separate => QkvTensors::Merged(
                linear(
                    hidden_width,
                    hidden_width + 2 * key_value_width,
                    vb.push_prefix("qkv"),
                )
                .context(SelfAttentionConstructionSnafu)?,
            ),
        };

        Ok(Self {
            attention_scorer,
            attention_heads,
            dropout,
            layer_norm,
            output,
            qkv,
            rotary_embeds,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        train: bool,
        use_causal_mask: bool,
    ) -> Result<Tensor, SelfAttentionError> {
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

        self.output
            .forward(&attn)
            .and_then(|xs| self.dropout.forward_t(&xs, train))
            .context(OutputSnafu)
    }
}

/// Create a causal mask.
///
/// A causal mask ensures that tokens cannot attend to succeeding tokens.
fn create_causal_mask(query: &Tensor, key: &Tensor) -> Result<AttentionMask, SelfAttentionError> {
    let (_, _, query_len, _) = query.shape().dims4().context(CausalMaskSnafu)?;
    let (_, _, key_len, _) = key.shape().dims4().context(CausalMaskSnafu)?;

    let causal_mask = Tensor::tril2(key_len, key.dtype(), key.device())
        .and_then(|mask| mask.reshape((1, 1, key_len, key_len)))
        .context(CausalMaskSnafu)?;
    Ok(AttentionMask::new(
        causal_mask
            .i((.., .., key_len - query_len..key_len, ..key_len))
            .context(CausalMaskSnafu)?,
    ))
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
