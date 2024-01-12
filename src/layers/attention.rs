use candle_core::Module;
use candle_core::Tensor;
use candle_nn::ops::softmax;
use candle_nn::{linear, Linear, VarBuilder};

use crate::error::Result;
use crate::layers::QueryKeyRotaryEmbeddings;

pub struct AttentionHeads {
    n_query_heads: usize,
    n_key_value_heads: usize,
    qkv_mode: QkvMode,
}

pub struct AttentionMask {
    bool_mask: Tensor,
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

pub struct ScaledDotProducAttention {
    // TODO: dropout, linear biases
}

impl ScaledDotProducAttention {
    pub fn new(_dropout: f64) -> Self {
        ScaledDotProducAttention {}
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
    ) -> Result<Tensor> {
        let model_width = key.dim(3)?;
        let attn_scores = query.matmul(&key.transpose(3, 2)?)?;
        let temperature = (model_width as f64).sqrt();
        let attn_scores = (attn_scores / temperature)?;
        let attn_weights = softmax(&attn_scores, 3)?;
        Ok(attn_weights.matmul(&value)?)
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
    attention: ScaledDotProducAttention,
    attention_heads: AttentionHeads,
    head_width: usize,
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
            attention: ScaledDotProducAttention::new(dropout),
            attention_heads,
            head_width,
            output,
            qkv,
            rotary_embeds,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        _use_causal_mask: bool,
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

        if let Some(rotaty_embeds) = &self.rotary_embeds {
            let (query_rot, key_rot) = rotaty_embeds.forward(&query, &key, None, None)?;
            query = query_rot;
            key = key_rot;
        }

        // TODO: rotary embeds

        // TODO: kv cache

        // TODO: causal mask

        // TODO: ALiBi

        let attn = self
            .attention
            .forward(&query, &key, &value, attention_mask)?
            .combine_heads()?;

        Ok(self.output.forward(&attn)?)
    }
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
