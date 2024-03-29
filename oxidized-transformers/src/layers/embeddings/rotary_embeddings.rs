use std::sync::RwLock;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use snafu::{ensure, ResultExt, Snafu};

/// Configuration for rotary embeddings.
pub struct RotaryEmbeddingsConfig {
    width: usize,
    seq_len: usize,
    base: usize,
}

impl RotaryEmbeddingsConfig {
    /// Build a rotary embeddings module.
    pub fn build(&self, vb: VarBuilder) -> Result<RotaryEmbeddings, RotaryEmbeddingsError> {
        ensure!(self.width % 2 == 0, WidthNotEvenSnafu { width: self.width });

        // Θ_i = 10000^(-2(i-1)/d)
        let theta: Vec<_> = (0..self.width)
            .step_by(2)
            .map(|i| (self.base as f32).powf(-(i as f32 / self.width as f32)))
            .collect();
        let theta =
            Tensor::from_vec(theta, (self.width / 2,), vb.device()).context(ThetaTensorSnafu)?;

        let (cos, sin) =
            RotaryEmbeddings::create_rotary_embed(&theta, self.width, self.seq_len, vb.dtype())
                .context(CacheSnafu)?;

        Ok(RotaryEmbeddings {
            cache: RwLock::new(RotaryEmbeddingsCache { cos, sin }),
            theta,
        })
    }

    /// Width of the rotary embeddings.
    ///
    /// The width must be even.
    ///
    /// Default: `96`
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Number of positions to precompute rotary embeddings for.
    ///
    /// The internal data structures will be resized if a longer sequence is
    /// encountered.
    ///
    /// Default: `2048`
    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Base used for `theta`.
    ///
    /// This determines the cycle length of the embeddings.
    ///
    /// Default: `10_000`
    pub fn base(mut self, base: usize) -> Self {
        self.base = base;
        self
    }
}

impl Default for RotaryEmbeddingsConfig {
    fn default() -> Self {
        Self {
            width: 96,
            seq_len: 2048,
            base: 10_000,
        }
    }
}

/// Errors for rotary embeddings.
#[derive(Debug, Snafu)]
pub enum RotaryEmbeddingsError {
    #[snafu(display("Cannot apply rotary embeddings to input"))]
    ApplyEmbeddings { source: candle_core::Error },

    #[snafu(display("Cannot get cache length"))]
    CacheLength { source: candle_core::Error },

    #[snafu(display("Cannot create rotary embeddings cache"))]
    Cache { source: candle_core::Error },

    #[snafu(display("Invalid input rank, expected {expected}, got {got}"))]
    InvalidRank {
        expected: usize,
        got: usize,
        source: candle_core::Error,
    },

    #[snafu(display("Cannot rotate input tensor"))]
    Rotate { source: candle_core::Error },

    #[snafu(display("Cannot slice rotary embeddings cache"))]
    SliceCache { source: candle_core::Error },

    #[snafu(display("Cannot convert theta to candle tensor"))]
    ThetaTensor { source: candle_core::Error },

    #[snafu(display("Rotary width must be even, was {width}"))]
    WidthNotEven { width: usize },
}

#[derive(Debug)]
struct RotaryEmbeddingsCache {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbeddingsCache {
    fn seq_len(&self) -> Result<usize, RotaryEmbeddingsError> {
        let (seq_len, _) = self.cos.shape().dims2().context(CacheLengthSnafu)?;
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
    fn create_rotary_embed(
        theta: &Tensor,
        width: usize,
        length: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor), candle_core::Error> {
        let device = theta.device();

        // mΘ
        let position = Tensor::arange(0.0, length as f32, device)?.unsqueeze(1)?;
        let m_theta = position.broadcast_mul(&theta.unsqueeze(0)?)?;

        // We apply both sin and cos twice (see Eq 15, 34), but the ordering
        // is changed for compatibility with most common implementations.
        let m_theta = Tensor::cat(&[&m_theta, &m_theta], 1)?;

        let re_cos = m_theta.cos()?.reshape(&[length, width])?.to_dtype(dtype)?;
        let re_sin = m_theta.sin()?.reshape(&[length, width])?.to_dtype(dtype)?;

        Ok((re_cos, re_sin))
    }

    fn resize_rotary_embed(
        &self,
        width: usize,
        len: usize,
        dtype: DType,
    ) -> Result<(), RotaryEmbeddingsError> {
        let (re_cos, re_sin) =
            Self::create_rotary_embed(&self.theta, width, len, dtype).context(CacheSnafu)?;
        let mut cache = self.cache.write().unwrap();
        cache.cos = re_cos;
        cache.sin = re_sin;
        Ok(())
    }

    /// Rotate the input tensor by half of its innermost width.
    fn rotate(input: &Tensor) -> Result<Tensor, RotaryEmbeddingsError> {
        let (_batch_size, _n_heads, _seq_len, n_dims) =
            input.shape().dims4().context(RotateSnafu)?;
        let half_idx = n_dims / 2;
        let input_1 = input
            .i((.., .., .., half_idx..))
            .and_then(|xs| xs.neg())
            .context(RotateSnafu)?;
        let input_2 = input.i((.., .., .., ..half_idx)).context(RotateSnafu)?;
        Tensor::cat(&[&input_1, &input_2], 3).context(RotateSnafu)
    }

    /// Apply rotary embeddings to the input.
    ///
    /// * `input` - Input to apply the rotary embeddings to.
    ///   *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`
    /// * `positions` - Positions of the inputs. If no positions are
    ///    provided, they are assumed to be `[0, seq_len)`.
    ///    *Shape:* `(batch_size, seq_len)`
    ///
    /// Returns: Input with the rotary embeddings applied.
    /// *Shape:* `(batch_size, n_heads, seq_len, width_per_head)`    
    pub fn forward(
        &self,
        input: &Tensor,
        positions: Option<&Tensor>,
    ) -> Result<Tensor, RotaryEmbeddingsError> {
        let (batch_size, _, seq_len, width) = input.shape().dims4().context(InvalidRankSnafu {
            expected: 4usize,
            got: input.rank(),
        })?;
        let dtype = self.cache.read().unwrap().cos.dtype();
        let (rot_cos, rot_sin) = match positions {
            None => {
                // Fastpath: positions from [0..seq_len), avoid indexing.
                if self.cache.read().unwrap().seq_len()? < seq_len {
                    self.resize_rotary_embed(width, seq_len, dtype)?;
                }
                let cache = self.cache.read().unwrap();
                let rot_cos = cache
                    .cos
                    .i((..seq_len, ..))
                    .and_then(|xs| xs.reshape((1, 1, seq_len, width)))
                    .context(SliceCacheSnafu)?;
                let rot_sin = cache
                    .sin
                    .i((..seq_len, ..))
                    .and_then(|xs| xs.reshape((1, 1, seq_len, width)))
                    .context(SliceCacheSnafu)?;
                (rot_cos, rot_sin)
            }
            Some(positions) => {
                // Maximum length is highest position (starting from 0) + 1.
                let positions_flat = positions.flatten_all().context(SliceCacheSnafu)?;
                let max_len = positions_flat
                    .max(0)
                    .and_then(|xs| xs.to_scalar::<u32>())
                    .context(SliceCacheSnafu)? as usize
                    + 1;

                if self.cache.read().unwrap().seq_len()? < max_len {
                    self.resize_rotary_embed(width, max_len, dtype)?;
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
                    .index_select(&positions_flat, 0)
                    .and_then(|xs| xs.reshape((batch_size, 1, seq_len, width)))
                    .context(SliceCacheSnafu)?;
                let rot_sin = cache
                    .sin
                    .index_select(&positions_flat, 0)
                    .and_then(|xs| xs.reshape((batch_size, 1, seq_len, width)))
                    .context(SliceCacheSnafu)?;
                (rot_cos, rot_sin)
            }
        };

        let input_rot_cos = input
            .broadcast_mul(&rot_cos)
            .context(ApplyEmbeddingsSnafu)?;
        let input_rot_sin = Self::rotate(input)?
            .broadcast_mul(&rot_sin)
            .context(ApplyEmbeddingsSnafu)?;

        (input_rot_cos + input_rot_sin).context(ApplyEmbeddingsSnafu)
    }
}
