use candle_core::{Device, Error, IndexOp, Result, Tensor};

/// Rotary embeddings (Su et al., 2021).
///
/// Paper: https://arxiv.org/abs/2104.09864
pub struct RotaryEmbeddings {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbeddings {
    /// Construct a rotary embedding module.
    ///
    /// The rotary embedding will be precomputed for up to ``seq_len`` positions.
    /// The embedding will be recomputed when a longer sequence is found in the
    /// input.
    ///
    /// * `width` - Rotary embedding width. Must be even.
    /// * `seq_len` - Number of positions to initially precompute.
    /// * `base` - The base used for theta_i (normally 10_000). Determines the cycle
    ///    length of the embeddings.
    /// * `device` - Device on which the module is to be initialized.
    pub fn new(width: usize, seq_len: usize, base: usize, device: &Device) -> Result<Self> {
        // TODO: replace by error.
        assert!(
            width % 2 == 0,
            "Width of rotary embeddings must be even, was: {}",
            width
        );

        let (sin, cos) = Self::create_rotary_embed(width, seq_len, base, device)?;

        Ok(RotaryEmbeddings { cos, sin })
    }

    fn create_rotary_embed(
        width: usize,
        length: usize,
        base: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Θ_i = 10000^(-2(i-1)/d)
        let theta: Vec<_> = (0..width)
            .step_by(2)
            .map(|i| (base as f32).powf(-(i as f32) / width as f32))
            .collect();
        let theta = Tensor::from_vec(theta, (width / 2,), device)?;

        // mΘ
        let position = Tensor::arange(0.0, length as f32, device)?.unsqueeze(1)?;
        let m_theta = (position * theta.unsqueeze(0)?)?;

        // We apply both sin and cos twice (see Eq 15, 34), but the ordering
        // is changed for compatibility with most common implementations.
        let m_theta = Tensor::cat(&[&m_theta, &m_theta], 1)?;

        let re_cos = m_theta.cos()?.reshape(&[length, width])?;
        let re_sin = m_theta.sin()?.reshape(&[length, width])?;

        Ok((re_sin, re_cos))
    }

    /// Rotate the input tensor by half of its innermost width.
    fn rotate(input: &Tensor) -> Result<Tensor> {
        let half_idx = input
            .shape()
            .dims()
            .last()
            .ok_or_else(|| Error::DimOutOfRange {
                shape: input.shape().clone(),
                dim: -1,
                op: "RotaryEmbeddings::rotate",
            })?
            / 2;
        let input_1 = input.i((.., .., .., half_idx..))?;
        let input_2 = input.i((.., .., .., ..half_idx))?;
        Tensor::cat(&[&input_1, &input_2], 3)
    }

    /// Apply rotary embeddings to the input.
    ///
    /// Returns the input with the rotary embeddings applied. *Shape:*
    /// `(batch_size, n_heads, seq_len, width_per_head)`
    ///
    /// * `input` - Input to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `positions` - Positions of the inputs. If no positions are
    ///    provided, they are assumed to be `[0, seq_len)`.
    ///    *Shape:* `(batch_size, seq_len)`
    pub fn forward(&self, input: Tensor, positions: Option<Tensor>) -> Result<Tensor> {
        let (batch_size, _, seq_len, width) = input.shape().dims4()?;
        let (rot_sin, rot_cos) = match positions {
            None => {
                // Fastpath: positions from [0..seq_len), avoid indexing.
                // TODO: resize when too short.
                let rot_cos = self
                    .cos
                    .i((..seq_len, ..))?
                    .reshape((1, 1, seq_len, width))?;
                let rot_sin = self
                    .cos
                    .i((..seq_len, ..))?
                    .reshape((1, 1, seq_len, width))?;
                (rot_sin, rot_cos)
            }
            Some(positions) => {
                let positions_flat = positions.reshape(())?;
                let max_len = positions_flat.max(0)?;
                // TODO: resize when too short.
                // Flatten positions to index cos/sin arrays, then unflatten.
                //
                // Example shapes:
                //
                //   positions_flat - (batch_size * seq_len)
                //   self.cos - (max_len, width)
                //   rot_cos - (batch_size, seq_len, width)
                let rot_cos = self
                    .cos
                    .index_select(&positions_flat, 0)?
                    .reshape((batch_size, 1, seq_len, width))?;
                let rot_sin = self
                    .sin
                    .index_select(&positions_flat, 0)?
                    .reshape((batch_size, 1, seq_len, width))?;
                (rot_sin, rot_cos)
            }
        };

        (rot_cos * &input)? + (rot_sin * Self::rotate(&input)?)?
    }
}
