use std::sync::RwLock;

use candle_core::{Device, Error, IndexOp, Shape, Tensor};

use crate::error::Result;

#[derive(Debug)]
struct RotaryEmbeddingsCache {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbeddingsCache {
    fn seq_len(&self) -> Result<usize> {
        let (seq_len, _) = self.cos.shape().dims2()?;
        Ok(seq_len)
    }
}

/// Rotary embeddings (Su et al., 2021).
///
/// Paper: https://arxiv.org/abs/2104.09864
#[derive(Debug)]
pub struct RotaryEmbeddings {
    cache: RwLock<RotaryEmbeddingsCache>,
    theta: Tensor,
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
        if width % 2 != 0 {
            return Err(crate::error::TransformerError::MultipleError {
                multiple: 2,
                value: width,
                msg: "cannot initialize rotary embeddings",
            });
        }

        // Θ_i = 10000^(-2(i-1)/d)
        let theta: Vec<_> = (0..width)
            .step_by(2)
            .map(|i| (base as f32).powf(-(i as f32 / width as f32)))
            .collect();
        let theta = Tensor::from_vec(theta, (width / 2,), device)?;

        let (cos, sin) = Self::create_rotary_embed(&theta, width, seq_len)?;

        Ok(RotaryEmbeddings {
            cache: RwLock::new(RotaryEmbeddingsCache { cos, sin }),
            theta,
        })
    }

    fn create_rotary_embed(
        theta: &Tensor,
        width: usize,
        length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = theta.device();

        // mΘ
        let position = Tensor::arange(0.0, length as f32, device)?.unsqueeze(1)?;
        let m_theta = position.broadcast_mul(&theta.unsqueeze(0)?)?;

        // We apply both sin and cos twice (see Eq 15, 34), but the ordering
        // is changed for compatibility with most common implementations.
        let m_theta = Tensor::cat(&[&m_theta, &m_theta], 1)?;

        let re_cos = m_theta.cos()?.reshape(&[length, width])?;
        let re_sin = m_theta.sin()?.reshape(&[length, width])?;

        Ok((re_cos, re_sin))
    }

    fn resize_rotary_embed(&self, width: usize, len: usize) -> Result<()> {
        let (re_cos, re_sin) = Self::create_rotary_embed(&self.theta, width, len)?;
        let mut cache = self.cache.write().unwrap();
        cache.cos = re_cos;
        cache.sin = re_sin;
        Ok(())
    }

    /// Rotate the input tensor by half of its innermost width.
    fn rotate(input: &Tensor) -> Result<Tensor> {
        let (_batch_size, _n_heads, _seq_len, n_dims) = input.shape().dims4()?;
        let half_idx = n_dims / 2;
        let input_1 = input.i((.., .., .., half_idx..))?.neg()?;
        let input_2 = input.i((.., .., .., ..half_idx))?;
        Ok(Tensor::cat(&[&input_1, &input_2], 3)?)
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
    pub fn forward(&self, input: &Tensor, positions: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, _, seq_len, width) = input.shape().dims4()?;
        let (rot_cos, rot_sin) = match positions {
            None => {
                // Fastpath: positions from [0..seq_len), avoid indexing.
                if self.cache.read().unwrap().seq_len()? < seq_len {
                    self.resize_rotary_embed(width, seq_len)?;
                }
                let cache = self.cache.read().unwrap();
                let rot_cos = cache
                    .cos
                    .i((..seq_len, ..))?
                    .reshape((1, 1, seq_len, width))?;
                let rot_sin = cache
                    .sin
                    .i((..seq_len, ..))?
                    .reshape((1, 1, seq_len, width))?;
                (rot_cos, rot_sin)
            }
            Some(positions) => {
                let positions_flat = positions.reshape(())?;
                let max_len = positions_flat.max(0)?.to_scalar::<i64>()? as usize;
                if self.cache.read().unwrap().seq_len()? < max_len {
                    self.resize_rotary_embed(width, max_len)?;
                }
                let cache = self.cache.read().unwrap();
                // Flatten positions to index cos/sin arrays, then unflatten.
                //
                // Example shapes:
                //
                //   positions_flat - (batch_size * seq_len)
                //   self.cos - (max_len, width)
                //   rot_cos - (batch_size, seq_len, width)
                let rot_cos = cache
                    .cos
                    .index_select(&positions_flat, 0)?
                    .reshape((batch_size, 1, seq_len, width))?;
                let rot_sin = cache
                    .sin
                    .index_select(&positions_flat, 0)?
                    .reshape((batch_size, 1, seq_len, width))?;
                (rot_cos, rot_sin)
            }
        };

        Ok(((rot_cos * input)? + (rot_sin * Self::rotate(input)?)?)?)
    }
}

#[derive(Debug)]
pub struct QueryKeyRotaryEmbeddings {
    rotary_width: usize,
    rotary_embeds: RotaryEmbeddings,
    pub head_width: usize,
}

impl QueryKeyRotaryEmbeddings {
    /// Construct query-key rotary embeddings module.
    ///
    /// The rotary embedding will be precomputed for up to ``seq_len`` positions.
    /// The embedding will be recomputed when a longer sequence is found in the
    /// input.
    ///
    /// * `fraction` - Fraction of hidden width to apply rotary embeddings to.
    ///   Must be in ``[0,1]``.
    /// * `head_width` - Query and key attention head width.
    /// * `seq_len` - Number of positions to initially precompute.
    /// * `base` - The base used for theta_i (normally 10_000). Determines the cycle
    ///    length of the embeddings.
    /// * `device` - Device on which the module is to be initialized.
    pub fn new(
        fraction: f32,
        head_width: usize,
        seq_len: usize,
        base: usize,
        device: &Device,
    ) -> Result<Self> {
        if !(0f32..=1f32).contains(&fraction) {
            return Err(crate::error::TransformerError::InclusiveRangeError {
                value: fraction,
                range: 0.0..=1.0,
                msg: "fraction out of range",
            });
        }

        // Truncate to get the width in case it is fractional.
        let rotary_width = (fraction * head_width as f32) as usize;
        let rotary_embeds = RotaryEmbeddings::new(rotary_width, seq_len, base, device)?;

        Ok(QueryKeyRotaryEmbeddings {
            head_width,
            rotary_embeds,
            rotary_width,
        })
    }

    /// Apply rotary embeddings to the query and key.
    ///
    /// Returns the input with the rotary embeddings applied. *Shape:*
    /// `(batch_size, n_heads, seq_len, width_per_head)`
    ///
    /// * `query` - Query to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `key` - Key to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `positions` - Positions of the inputs. If no positions are
    ///    provided, they are assumed to be `[0, seq_len)`.
    ///    *Shape:* `(batch_size, seq_len)`
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        cache: Option<KeyValueCache>,
        mut positions: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, n_heads, seq_len, head_width) = query.shape().dims4()?;

        if head_width != self.head_width {
            Err(Error::UnexpectedShape {
                msg: "incorrect head size".to_string(),
                expected: Shape::from_dims(&[batch_size, n_heads, seq_len, self.head_width]),
                got: Shape::from_dims(&[batch_size, n_heads, seq_len, head_width]),
            })?;
        }

        // If a cache was provided, but no positions, assume that the
        // positions of the current batch continue from the cache.
        if let (Some(cache), None) = (cache, &positions) {
            let (_, cache_len, _, _) = cache.key.shape().dims4()?;
            positions = Some(
                Tensor::arange(
                    cache_len as i64,
                    cache_len as i64 + seq_len as i64,
                    query.device(),
                )?
                .repeat((batch_size, 1))?,
            );
        }

        if self.rotary_width == head_width {
            // Fast path: we apply rotary embeddings the full key/query vectors.
            let query_rot = self.rotary_embeds.forward(query, positions.as_ref())?;
            let key_rot = self.rotary_embeds.forward(key, positions.as_ref())?;
            Ok((query_rot, key_rot))
        } else {
            let rotary_index = (.., .., .., ..self.rotary_width);
            let rotary_range = &[..batch_size, ..n_heads, ..seq_len, ..self.rotary_width];
            let query_rot = query.slice_assign(
                rotary_range,
                &self
                    .rotary_embeds
                    .forward(&query.i(rotary_index)?, positions.as_ref())?,
            )?;

            let key_rot = key.slice_assign(
                rotary_range,
                &self
                    .rotary_embeds
                    .forward(&key.i(rotary_index)?, positions.as_ref())?,
            )?;

            Ok((query_rot, key_rot))
        }
    }
}

pub struct KeyValueCache {
    pub key: Tensor,
    pub value: Tensor,
}

#[cfg(test)]
mod tests {
    use crate::util::tests::assert_close;
    use candle_core::{Device, Tensor};

    use super::QueryKeyRotaryEmbeddings;

    #[test]
    fn query_key_rotary_has_correct_output() {
        for initial_len in [10, 2] {
            let device = Device::Cpu;
            let rotary =
                QueryKeyRotaryEmbeddings::new(1.0, 4, initial_len, 10000, &device).unwrap();
            let query = Tensor::arange(0f32, 16f32, &device)
                .unwrap()
                .reshape((1, 1, 4, 4))
                .unwrap();
            let key = Tensor::arange(16f32, 32f32, &device)
                .unwrap()
                .reshape((1, 1, 4, 4))
                .unwrap();
            let (query_rot, key_rot) = rotary.forward(&query, &key, None, None).unwrap();

            assert_close(
                &query_rot,
                &Tensor::from_slice(
                    &[
                        0.0000f32, 1.0000, 2.0000, 3.0000, -2.8876, 4.9298, 6.6077, 7.0496,
                        -12.4221, 8.7782, 3.1129, 11.1778, -13.8556, 12.5442, -12.1665, 15.3832,
                    ],
                    (1, 1, 4, 4),
                    &device,
                )
                .unwrap(),
                1e-4,
            );
            assert_close(
                &key_rot,
                &Tensor::from_slice(
                    &[
                        16.0000f32, 17.0000, 18.0000, 19.0000, -7.7063, 20.7690, 28.7161, 23.2088,
                        -33.6293, 24.4550, 11.0033, 27.4946, -31.9534, 28.0571, -25.7484, 31.8559,
                    ],
                    (1, 1, 4, 4),
                    &device,
                )
                .unwrap(),
                1e-4,
            );
        }
    }

    #[test]
    fn query_key_rotary_fractional_has_correct_output() {
        for initial_len in [10, 2] {
            let device = Device::Cpu;
            let rotary =
                QueryKeyRotaryEmbeddings::new(0.5, 8, initial_len, 10000, &device).unwrap();
            let query = Tensor::arange(0f32, 32f32, &device)
                .unwrap()
                .reshape((1, 1, 4, 8))
                .unwrap();
            let key = Tensor::arange(32f32, 64f32, &device)
                .unwrap()
                .reshape((1, 1, 4, 8))
                .unwrap();
            let (query_rot, key_rot) = rotary.forward(&query, &key, None, None).unwrap();

            assert_close(
                &query_rot,
                &Tensor::from_slice(
                    &[
                        0.0000f32, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, -4.0923,
                        8.8896, 12.1348, 11.0894, 12.0000, 13.0000, 14.0000, 15.0000, -23.0257,
                        16.6166, 7.0581, 19.3362, 20.0000, 21.0000, 22.0000, 23.0000, -27.4289,
                        24.1789, -22.3529, 27.7377, 28.0000, 29.0000, 30.0000, 31.0000,
                    ],
                    (1, 1, 4, 8),
                    &device,
                )
                .unwrap(),
                1e-4,
            );

            assert_close(
                &key_rot,
                &Tensor::from_slice(
                    &[
                        32.0000f32, 33.0000, 34.0000, 35.0000, 36.0000, 37.0000, 38.0000, 39.0000,
                        -13.7297, 40.5680, 56.3515, 43.4078, 44.0000, 45.0000, 46.0000, 47.0000,
                        -65.4399, 47.9703, 22.8389, 51.9697, 52.0000, 53.0000, 54.0000, 55.0000,
                        -63.6245, 55.2046, -49.5168, 60.6832, 60.0000, 61.0000, 62.0000, 63.0000,
                    ],
                    (1, 1, 4, 8),
                    &device,
                )
                .unwrap(),
                1e-4,
            );
        }
    }
}
