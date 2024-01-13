use candle_core::Tensor;
use candle_core::{IndexOp, Module};
use candle_nn::ops::softmax;
use candle_nn::{linear, Linear, VarBuilder};
use std::borrow::Cow;

use crate::error::Result;
use crate::layers::QueryKeyRotaryEmbeddings;

pub struct AttentionHeads {
    n_query_heads: usize,
    n_key_value_heads: usize,
    qkv_mode: QkvMode,
}

/// Attention mask.
///
/// Sequence elements for which the corresponding mask element is set to
/// ``False`` are ignored during attention calculation.
#[derive(Clone, Debug)]
pub struct AttentionMask {
    bool_mask: Tensor,
}

impl AttentionMask {
    /// Use the attention mask to mask attention logits.
    pub fn apply_logit_mask(&self, input: &Tensor) -> Result<Tensor> {
        // Underflows to -inf for more narrow floating point types, which
        // is ok for masking.
        let blocked_value = Tensor::try_from(f32::MIN)?;
        Ok(self.bool_mask.where_cond(input, &blocked_value)?)
    }

    /// Merge this attention mask with another attention mask.
    pub fn merge_mask(&self, other: &AttentionMask) -> Result<AttentionMask> {
        Ok(AttentionMask {
            bool_mask: (&self.bool_mask * &other.bool_mask)?,
        })
    }
}

#[non_exhaustive]
pub enum QkvSplit {
    Default,
    KVSizedChunks,
}

#[non_exhaustive]
pub enum QkvMode {
    Separate,
    MergedSplitBefore,
    MergedSplitAfter(QkvSplit),
}

pub struct ScaledDotProductAttention {
    // TODO: dropout, linear biases
}

impl ScaledDotProductAttention {
    pub fn new(_dropout: f64) -> Self {
        ScaledDotProductAttention {}
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
    ) -> Result<Tensor> {
        // TODO: add code path for flash attention, but verify the attention
        //       layer is working first...
        let model_width = key.dim(3)?;
        let mut attn_scores = query.matmul(&key.transpose(3, 2)?)?;
        attn_scores = attention_mask.apply_logit_mask(&attn_scores)?;
        let temperature = (model_width as f64).sqrt();
        let attn_scores = (attn_scores / temperature)?;
        let attn_weights = softmax(&attn_scores, 3)?;
        Ok(attn_weights.matmul(value)?)
    }
}

pub enum QkvTensors {
    Merged(Linear),
    Separate {
        query: Linear,
        key: Linear,
        value: Linear,
    },
}

pub struct SelfAttention {
    // TODO: dropout prob
    attention: ScaledDotProductAttention,
    attention_heads: AttentionHeads,
    output: Linear,
    qkv: QkvTensors,
    rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
}

impl SelfAttention {
    pub fn new(
        vb: VarBuilder,
        attention_heads: AttentionHeads,
        dropout: f64,
        hidden_width: usize,
        rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
        _use_bias: bool,
    ) -> Result<Self> {
        // TODO: use_bias
        let head_width = hidden_width / attention_heads.n_key_value_heads;
        let key_value_width = attention_heads.n_key_value_heads * head_width;
        let output = linear(hidden_width, hidden_width, vb.push_prefix("output"))?;
        let qkv = match attention_heads.qkv_mode {
            QkvMode::MergedSplitBefore | QkvMode::MergedSplitAfter(_) => QkvTensors::Separate {
                query: linear(hidden_width, hidden_width, vb.push_prefix("query"))?,
                key: linear(hidden_width, key_value_width, vb.push_prefix("key"))?,
                value: linear(hidden_width, key_value_width, vb.push_prefix("value"))?,
            },
            QkvMode::Separate => QkvTensors::Merged(linear(
                hidden_width,
                hidden_width + 2 * key_value_width,
                vb.push_prefix("qkv"),
            )?),
        };

        Ok(Self {
            attention: ScaledDotProductAttention::new(dropout),
            attention_heads,
            output,
            qkv,
            rotary_embeds,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        use_causal_mask: bool,
    ) -> Result<Tensor> {
        let (mut query, mut key, value) = match &self.qkv {
            QkvTensors::Separate { query, key, value } => {
                let query = query
                    .forward(input)?
                    .split_heads(self.attention_heads.n_query_heads)?;
                let key = key
                    .forward(input)?
                    .split_heads(self.attention_heads.n_key_value_heads)?;
                let value = value
                    .forward(input)?
                    .split_heads(self.attention_heads.n_key_value_heads)?;
                (query, key, value)
            }
            _ => unimplemented!(),
        };

        if let Some(rotary_embeds) = &self.rotary_embeds {
            let (query_rot, key_rot) = rotary_embeds.forward(&query, &key, None, None)?;
            query = query_rot;
            key = key_rot;
        }

        // TODO: kv cache

        // TODO: causal mask

        // TODO: ALiBi

        let combined_mask = if use_causal_mask {
            let causal_mask = create_causal_mask(&query, &key)?;
            Cow::Owned(attention_mask.merge_mask(&causal_mask)?)
        } else {
            Cow::Borrowed(attention_mask)
        };

        let attn = self
            .attention
            .forward(&query, &key, &value, &combined_mask)?
            .combine_heads()?;

        Ok(self.output.forward(&attn)?)
    }
}

/// Create a causal mask.
///
/// A causal mask ensures that tokens cannot attend to succeeding tokens.
fn create_causal_mask(query: &Tensor, key: &Tensor) -> Result<AttentionMask> {
    let (_, _, query_len, _) = query.shape().dims4()?;
    let (_, _, key_len, _) = key.shape().dims4()?;

    let causal_mask =
        Tensor::tril2(key_len, key.dtype(), key.device())?.reshape((1, 1, key_len, key_len))?;
    Ok(AttentionMask {
        bool_mask: causal_mask.i((.., .., key_len - query_len..key_len, ..key_len))?,
    })
}

trait CombineHeads {
    fn combine_heads(&self) -> Result<Tensor>;
}

impl CombineHeads for Tensor {
    fn combine_heads(&self) -> Result<Tensor> {
        let (batch_size, n_heads, seq_len, model_width) = self.dims4()?;
        Ok(self
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, n_heads * model_width))?)
    }
}

trait SplitHeads {
    fn split_heads(&self, n_heads: usize) -> Result<Tensor>;
}

impl SplitHeads for Tensor {
    fn split_heads(&self, n_heads: usize) -> Result<Tensor> {
        let (batch_size, seq_len, model_width) = self.dims3()?;
        let head_width = model_width / n_heads;
        Ok(self
            .reshape((batch_size, seq_len, n_heads, head_width))?
            .transpose(1, 2)?)
    }
}
