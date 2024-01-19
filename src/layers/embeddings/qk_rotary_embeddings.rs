use candle_core::{Device, IndexOp, Tensor};
use snafu::{ensure, ResultExt, Snafu};

use crate::kv_cache::KeyValueCache;
use crate::layers::embeddings::{RotaryEmbeddings, RotaryEmbeddingsError};

/// Errors for query-key rotary embeddings.
#[derive(Debug, Snafu)]
pub enum QueryKeyRotaryEmbeddingsError {
    #[snafu(display("Cannot apply rotary embeddings to key/value fraction"))]
    ApplyToFraction { source: candle_core::Error },

    #[snafu(display("Fraction must be in [0,1), was {fraction}"))]
    InvalidRange { fraction: f32 },

    #[snafu(display("Invalid input rank, expected {expected}, got {got}"))]
    InvalidRank {
        expected: usize,
        got: usize,
        source: candle_core::Error,
    },

    #[snafu(display("Invalid head width, expected {expected_head_width}, got {head_width}"))]
    InvalidHeadWidth {
        expected_head_width: usize,
        head_width: usize,
    },

    #[snafu(display("Cannot calculate positions from cache"))]
    PositionsFromCache { source: candle_core::Error },

    #[snafu(display("Cannot apply rotary embeddings"))]
    RotaryEmbeddings { source: RotaryEmbeddingsError },
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
    /// The rotary embedding will be precomputed for up to `seq_len` positions.
    /// The embedding will be recomputed when a longer sequence is found in the
    /// input.
    ///
    /// * `fraction` - Fraction of hidden width to apply rotary embeddings to.
    ///   Must be in `[0,1]`.
    /// * `head_width` - Query and key attention head width.
    /// * `seq_len` - Number of positions to initially precompute.
    /// * `base` - The base used for `theta` (normally 10_000). Determines the cycle
    ///    length of the embeddings.
    /// * `device` - Device on which the module is to be initialized.
    pub fn new(
        fraction: f32,
        head_width: usize,
        seq_len: usize,
        base: usize,
        device: &Device,
    ) -> Result<Self, QueryKeyRotaryEmbeddingsError> {
        ensure!(
            (0f32..=1f32).contains(&fraction),
            InvalidRangeSnafu { fraction }
        );

        // Truncate to get the width in case it is fractional.
        let rotary_width = (fraction * head_width as f32) as usize;
        let rotary_embeds = RotaryEmbeddings::new(rotary_width, seq_len, base, device)
            .context(RotaryEmbeddingsSnafu)?;

        Ok(QueryKeyRotaryEmbeddings {
            head_width,
            rotary_embeds,
            rotary_width,
        })
    }

    /// Apply rotary embeddings to the query and key.
    ///
    /// * `query` - Query to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `key` - Key to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `positions` - Positions of the inputs. If no positions are
    ///    provided, they are assumed to be `[0, seq_len)`.
    ///    *Shape:* `(batch_size, seq_len)`
    ///
    /// Returns: Input with the rotary embeddings applied.
    /// *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        cache: Option<KeyValueCache>,
        mut positions: Option<Tensor>,
    ) -> Result<(Tensor, Tensor), QueryKeyRotaryEmbeddingsError> {
        let (batch_size, n_heads, seq_len, head_width) =
            query.shape().dims4().context(InvalidRankSnafu {
                expected: 4usize,
                got: query.rank(),
            })?;

        ensure!(
            head_width == self.head_width,
            InvalidHeadWidthSnafu {
                expected_head_width: self.head_width,
                head_width,
            }
        );

        // If a cache was provided, but no positions, assume that the
        // positions of the current batch continue from the cache.
        if let (Some(cache), None) = (cache, &positions) {
            let (_, cache_len, _, _) =
                cache.key.shape().dims4().context(PositionsFromCacheSnafu)?;

            positions = Some(
                Tensor::arange(
                    cache_len as i64,
                    cache_len as i64 + seq_len as i64,
                    query.device(),
                )
                .and_then(|xs| xs.repeat((batch_size, 1)))
                .context(PositionsFromCacheSnafu)?,
            );
        }

        if self.rotary_width == head_width {
            // Fast path: we apply rotary embeddings the full key/query vectors.
            let query_rot = self
                .rotary_embeds
                .forward(query, positions.as_ref())
                .context(RotaryEmbeddingsSnafu)?;
            let key_rot = self
                .rotary_embeds
                .forward(key, positions.as_ref())
                .context(RotaryEmbeddingsSnafu)?;
            Ok((query_rot, key_rot))
        } else {
            let rotary_index = (.., .., .., ..self.rotary_width);
            let rotary_range = &[..batch_size, ..n_heads, ..seq_len, ..self.rotary_width];

            let mut query_rot_frac = query.i(rotary_index).context(ApplyToFractionSnafu)?;
            query_rot_frac = self
                .rotary_embeds
                .forward(&query_rot_frac, positions.as_ref())
                .context(RotaryEmbeddingsSnafu)?;
            let query_rot = query
                .slice_assign(rotary_range, &query_rot_frac)
                .context(ApplyToFractionSnafu)?;

            let mut key_rot_frac = key.i(rotary_index).context(ApplyToFractionSnafu)?;
            key_rot_frac = self
                .rotary_embeds
                .forward(&key_rot_frac, positions.as_ref())
                .context(RotaryEmbeddingsSnafu)?;
            let key_rot = key
                .slice_assign(rotary_range, &key_rot_frac)
                .context(ApplyToFractionSnafu)?;

            Ok((query_rot, key_rot))
        }
    }
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
