use candle_core::{Device, Tensor};
use candle_core::{IndexOp, Module};
use candle_nn::ops::softmax;
use candle_nn::{linear, Dropout, Linear, VarBuilder};
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

/// Linear biases for attention (ALiBi).
///
/// See [Press et al., 2022](https://arxiv.org/abs/2108.12409).
pub struct AttentionLinearBiases {
    slopes: Tensor,
    is_causal: bool,
    is_inverted: bool,
}

impl AttentionLinearBiases {
    /// Construct a new linear bias layer for attention.
    ///
    /// * n_attention_heads - Number of attention heads.
    /// * is_causal - Use causal attention.
    /// * is_inverted - Invert the linear bias.
    pub fn new(n_attention_heads: usize, is_causal: bool, is_inverted: bool) -> Result<Self> {
        let slopes = Self::calculate_slopes(n_attention_heads)?;
        Ok(Self {
            slopes,
            is_causal,
            is_inverted,
        })
    }

    /// Calculate the linear bias slopes for a given number
    /// of attention heads.
    ///
    /// * n_attention_heads - Number of attention heads.
    ///
    /// Returns: Head slope tensor.
    /// *Shape:* `(1, heads, 1, 1)`
    #[allow(clippy::unnecessary_cast)]
    fn calculate_slopes(n_attention_heads: usize) -> Result<Tensor> {
        fn slopes_with_step(n_attention_heads: usize, step: usize) -> Result<Tensor> {
            let ratio = 2.0f32.powf(-8.0f32 / n_attention_heads as f32);
            let slope = (1..n_attention_heads + 1)
                .step_by(step)
                .map(|x| ratio.powi(x as i32))
                .collect::<Vec<_>>();
            Ok(Tensor::new(slope, &Device::Cpu)?)
        }

        // The slope as proposed in the ALiBi paper would be:
        //
        // return slopes_with_step(n_attention_heads)
        //
        // However the authors note in their implementation that using powers
        // of 2 for n in the ratio 2**(-8/n) of the geometric sequence for
        // slopes has better properties.
        //
        // Most implementations use powers of two in the following
        // manner: if the number of heads is not a power of 2, then we find
        // k=the largest power of 2 in 1..n. The slopes are then computed
        // as the concatenation of:
        //
        // - The slopes for 1..k.
        // - The slopes for 1..2*k with step 2, taking the first n-k elements.

        // k is the largest power of 2 in 1..n.
        // Coerced to the same type here to make the bit arithmetic explicit.
        let k = 1 << ((usize::BITS - (n_attention_heads as usize).leading_zeros()) - 1);
        let mut slopes = slopes_with_step(k, 1)?;

        if n_attention_heads != k {
            let remaining = n_attention_heads - k;
            let slopes_rest = slopes_with_step(2 * k, 2)?.i(..remaining)?;
            slopes = Tensor::cat(&[slopes, slopes_rest], 0)?;
        }

        Ok(slopes.reshape((1, (), 1, 1))?)
    }

    /// Calculate the linear bias tensor upto a given (key) sequence length.
    ///
    /// * seq_len - Maximum number of timesteps to calculate.
    ///
    /// Returns: Multi-headed linear bias tensor.
    /// *Shape:* `(1, heads, seq_len, seq_len)` (non-causal) or
    /// `(1, heads, 1, seq_len)` (causal)
    fn calculate_biases(&self, seq_len: usize) -> Result<Tensor> {
        let mut distances = if self.is_causal {
            Tensor::arange(1 - seq_len as i64, 1, &Device::Cpu)?
        } else {
            let mut distance = Tensor::arange(0, seq_len as i64, &Device::Cpu)?;
            distance = distance.broadcast_sub(&distance.reshape(((), 1))?)?;
            distance
                .abs()?
                .broadcast_mul(&Tensor::new(-1i64, &Device::Cpu)?)?
                .reshape((1, 1, seq_len, seq_len))?
        };

        if self.is_inverted {
            distances =
                distances.broadcast_add(&Tensor::new(seq_len as i64 - 1i64, &Device::Cpu)?)?;
        }

        distances = distances.to_dtype(self.slopes.dtype())?;
        Ok(distances.broadcast_mul(&self.slopes)?)
    }

    /// Apply linear biases to (unmasked) attention scores.
    ///
    /// * attention_scores - Attention scores.
    ///   *Shape:* `(batch_size, heads, query_len, key_len)`
    ///
    /// Returns: Attention scores with linear biases.
    /// *Shape:* `(batch_size, heads, query_len, key_len)`
    pub fn forward(&self, attention_scores: &Tensor) -> Result<Tensor> {
        let (_, _, _, key_len) = attention_scores.shape().dims4()?;
        let biases = self
            .calculate_biases(attention_scores.dim(key_len)?)?
            .to_dtype(attention_scores.dtype())?
            .to_device(attention_scores.device())?;

        Ok(attention_scores.add(&biases)?)
    }
}

/// Trait implemented by modules that perform attention scoring.
pub trait AttentionScorer {
    /// Apply attention scores to the given key, query and value.
    /// Sequence elements that are marked with `false` in the attention mask
    /// are ignored by the attention mechanism (if a mask is provided).
    ///
    /// * query - Query tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * key - Key tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * value - Value tensor.
    ///   *Shape:* `(batch_size, heads, seq_len, width)`
    /// * attention_mask - Attention mask. Sequence elements for which
    /// the corresponding mask element is set to `false` are ignored in attention.
    ///
    /// Returns: Attention values.
    /// *Shape:* `(batch_size, heads, seq_len, width)`
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &AttentionMask,
    ) -> Result<Tensor>;
}

pub struct ScaledDotProductAttention {
    dropout: Dropout,
    linear_biases: Option<AttentionLinearBiases>,
}

impl ScaledDotProductAttention {
    pub fn new(dropout_prob: f32, linear_biases: Option<AttentionLinearBiases>) -> Result<Self> {
        Ok(ScaledDotProductAttention {
            dropout: Dropout::new(dropout_prob),
            linear_biases,
        })
    }
}

impl AttentionScorer for ScaledDotProductAttention {
    fn forward(
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
        let temperature = (model_width as f64).sqrt();
        attn_scores = (attn_scores / temperature)?;

        if let Some(linear_biases) = &self.linear_biases {
            attn_scores = linear_biases.forward(&attn_scores)?;
        }

        attn_scores = attention_mask.apply_logit_mask(&attn_scores)?;

        let attn_weights = softmax(&attn_scores, 3)?;
        // TODO: handle training
        Ok(self.dropout.forward(&attn_weights.matmul(value)?, false)?)
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
    attention_scorer: Box<dyn AttentionScorer>,
    attention_heads: AttentionHeads,
    output: Linear,
    qkv: QkvTensors,
    rotary_embeds: Option<QueryKeyRotaryEmbeddings>,
}

impl SelfAttention {
    pub fn new(
        vb: VarBuilder,
        attention_heads: AttentionHeads,
        attention_scorer: Box<dyn AttentionScorer>,
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
            attention_scorer,
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

        if let Some(rotaty_embeds) = &self.rotary_embeds {
            let (query_rot, key_rot) = rotaty_embeds.forward(&query, &key, None, None)?;
            query = query_rot;
            key = key_rot;
        }

        // TODO: rotary embeds

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
            .attention_scorer
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

#[cfg(test)]
mod tests {
    use crate::util::tests::assert_close;
    use candle_core::{DType, Device, Tensor};

    use super::AttentionLinearBiases;

    #[test]
    fn test_attention_linear_biases_slopes() {
        let device = Device::Cpu;

        let pow2_biases = AttentionLinearBiases::new(8, false, false).unwrap();
        assert_close(
            &pow2_biases.slopes,
            &Tensor::new(
                &[
                    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(pow2_biases.slopes.dtype())
            .unwrap()
            .reshape((1, 8, 1, 1))
            .unwrap(),
            1e-4,
        );

        let non_pow2_biases = AttentionLinearBiases::new(12, false, false).unwrap();
        assert_close(
            &non_pow2_biases.slopes,
            &Tensor::new(
                &[
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                    0.03125,
                    0.015625,
                    0.0078125,
                    0.00390625,
                    0.7071067811865476,
                    0.35355339059327384,
                    0.17677669529663692,
                    0.08838834764831849,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(non_pow2_biases.slopes.dtype())
            .unwrap()
            .reshape((1, 12, 1, 1))
            .unwrap(),
            1e-4,
        );
    }

    #[test]
    fn test_attention_linear_biases_causal() {
        let device = Device::Cpu;

        let causal = AttentionLinearBiases::new(4, true, false).unwrap();
        assert_close(
            &causal
                .forward(&Tensor::zeros((1, 4, 1, 3), DType::F32, &device).unwrap())
                .unwrap(),
            &Tensor::new(
                &[
                    -0.5000,
                    -0.2500,
                    0.0000,
                    -0.1250,
                    -0.0625,
                    0.0000,
                    -0.03125,
                    -0.015625,
                    0.0000,
                    -0.0078125,
                    -0.00390625,
                    0.0000,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 4, 1, 3))
            .unwrap(),
            1e-4,
        );

        let inverted = AttentionLinearBiases::new(4, true, true).unwrap();
        assert_close(
            &inverted
                .forward(&Tensor::zeros((1, 4, 1, 3), DType::F32, &device).unwrap())
                .unwrap(),
            &Tensor::new(
                &[
                    0.0000, 0.2500, 0.5000, 0.0000, 0.0625, 0.1250, 0.0000, 0.015625, 0.03125,
                    0.0000, 0.00390625, 0.0078125,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 4, 1, 3))
            .unwrap(),
            1e-4,
        );
    }

    #[test]
    fn test_attention_linear_biases_non_causal() {
        let device = Device::Cpu;

        let non_causal = AttentionLinearBiases::new(4, false, false).unwrap();
        assert_close(
            &non_causal
                .forward(&Tensor::zeros((1, 4, 3, 3), DType::F32, &device).unwrap())
                .unwrap(),
            &Tensor::new(
                &[
                    0.0000,
                    -0.2500,
                    -0.5000,
                    -0.2500,
                    0.0000,
                    -0.2500,
                    -0.5000,
                    -0.2500,
                    0.0000,
                    0.0000,
                    -0.0625,
                    -0.1250,
                    -0.0625,
                    0.0000,
                    -0.0625,
                    -0.1250,
                    -0.0625,
                    0.0000,
                    0.0000,
                    -0.015625,
                    -0.03125,
                    -0.015625,
                    0.0000,
                    -0.015625,
                    -0.03125,
                    -0.015625,
                    0.0000,
                    0.0000,
                    -0.00390625,
                    -0.0078125,
                    -0.00390625,
                    0.0000,
                    -0.00390625,
                    -0.0078125,
                    -0.00390625,
                    0.0000,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 4, 3, 3))
            .unwrap(),
            1e-4,
        );

        let inverted = AttentionLinearBiases::new(4, false, true).unwrap();
        assert_close(
            &inverted
                .forward(&Tensor::zeros((1, 4, 3, 3), DType::F32, &device).unwrap())
                .unwrap(),
            &Tensor::new(
                &[
                    0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.2500, 0.0000, 0.2500, 0.5000, 0.1250,
                    0.0625, 0.0000, 0.0625, 0.1250, 0.0625, 0.0000, 0.0625, 0.1250, 0.03125,
                    0.015625, 0.0000, 0.015625, 0.03125, 0.015625, 0.0000, 0.015625, 0.03125,
                    0.0078125, 0.00390625, 0.0000, 0.00390625, 0.0078125, 0.00390625, 0.0000,
                    0.00390625, 0.0078125,
                ],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 4, 3, 3))
            .unwrap(),
            1e-4,
        );
    }
}
