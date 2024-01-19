use candle_core::{Device, IndexOp, Tensor};
use snafu::{ResultExt, Snafu};

/// Errors for calculation of attention linear biases.
#[derive(Debug, Snafu)]
pub enum AttentionLinearBiasesError {
    #[snafu(display("Cannot apply attention linear biases"))]
    ApplyBiases { source: candle_core::Error },

    #[snafu(display("Cannot calculate biases"))]
    CalculateBiases { source: candle_core::Error },

    #[snafu(display("Cannot calculate slopes"))]
    CalculateSlopes { source: candle_core::Error },
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
    pub fn new(
        n_attention_heads: usize,
        is_causal: bool,
        is_inverted: bool,
    ) -> Result<Self, AttentionLinearBiasesError> {
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
    fn calculate_slopes(n_attention_heads: usize) -> Result<Tensor, AttentionLinearBiasesError> {
        fn slopes_with_step(
            n_attention_heads: usize,
            step: usize,
        ) -> Result<Tensor, AttentionLinearBiasesError> {
            let ratio = 2.0f32.powf(-8.0f32 / n_attention_heads as f32);
            let slope = (1..n_attention_heads + 1)
                .step_by(step)
                .map(|x| ratio.powi(x as i32))
                .collect::<Vec<_>>();
            Tensor::new(slope, &Device::Cpu).context(CalculateSlopesSnafu)
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
            let slopes_rest = slopes_with_step(2 * k, 2)?
                .i(..remaining)
                .context(CalculateSlopesSnafu)?;
            slopes = Tensor::cat(&[slopes, slopes_rest], 0).context(CalculateSlopesSnafu)?;
        }

        slopes.reshape((1, (), 1, 1)).context(CalculateSlopesSnafu)
    }

    /// Calculate the linear bias tensor upto a given (key) sequence length.
    ///
    /// * seq_len - Maximum number of timesteps to calculate.
    ///
    /// Returns: Multi-headed linear bias tensor.
    /// *Shape:* `(1, heads, seq_len, seq_len)` (non-causal) or
    /// `(1, heads, 1, seq_len)` (causal)
    fn calculate_biases(&self, seq_len: usize) -> Result<Tensor, AttentionLinearBiasesError> {
        let mut distances = if self.is_causal {
            Tensor::arange(1 - seq_len as i64, 1, &Device::Cpu).context(CalculateBiasesSnafu)?
        } else {
            Tensor::arange(0, seq_len as i64, &Device::Cpu)
                .and_then(|xs| xs.broadcast_sub(&xs.reshape(((), 1))?))
                .and_then(|xs| xs.abs())
                .and_then(|xs| xs.broadcast_mul(&Tensor::new(-1i64, &Device::Cpu)?))
                .and_then(|xs| xs.reshape((1, 1, seq_len, seq_len)))
                .context(CalculateBiasesSnafu)?
        };

        if self.is_inverted {
            distances = Tensor::new(seq_len as i64 - 1i64, &Device::Cpu)
                .and_then(|xs| distances.broadcast_add(&xs))
                .context(CalculateBiasesSnafu)?;
        }

        distances
            .to_dtype(self.slopes.dtype())
            .and_then(|xs| xs.broadcast_mul(&self.slopes))
            .context(CalculateBiasesSnafu)
    }

    /// Apply linear biases to (unmasked) attention scores.
    ///
    /// * attention_scores - Attention scores.
    ///   *Shape:* `(batch_size, heads, query_len, key_len)`
    ///
    /// Returns: Attention scores with linear biases.
    /// *Shape:* `(batch_size, heads, query_len, key_len)`
    pub fn forward(&self, attention_scores: &Tensor) -> Result<Tensor, AttentionLinearBiasesError> {
        let (_, _, _, key_len) = attention_scores.shape().dims4().context(ApplyBiasesSnafu)?;
        let biases = self
            .calculate_biases(key_len)?
            .to_dtype(attention_scores.dtype())
            .and_then(|xs| xs.to_device(attention_scores.device()))
            .context(ApplyBiasesSnafu)?;

        attention_scores.add(&biases).context(ApplyBiasesSnafu)
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
