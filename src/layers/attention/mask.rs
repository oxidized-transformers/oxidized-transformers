use candle_core::{DType, IndexOp, Tensor};
use snafu::{ensure, ResultExt, Snafu};

#[derive(Debug, Snafu)]
pub enum QueryKeyAttentionMaskError {
    #[snafu(display("Cannot apply logits mask"))]
    ApplyLogitsMask { source: candle_core::Error },

    #[snafu(display("Cannot intersect masks"))]
    IntersectMasks { source: candle_core::Error },
}

/// Attention mask.
///
/// A 4D attention mask with shape *(batch_size, heads, query_len, key_len)*.
/// Elements for which the corresponding mask element is set to `False` are
/// ignored during attention calculation.
#[derive(Clone, Debug)]
pub struct QueryKeyAttentionMask {
    bool_mask: Tensor,
}

impl From<AttentionMask> for QueryKeyAttentionMask {
    fn from(attention_mask: AttentionMask) -> Self {
        QueryKeyAttentionMask::from(&attention_mask)
    }
}

impl From<&AttentionMask> for QueryKeyAttentionMask {
    fn from(attention_mask: &AttentionMask) -> Self {
        let (batch_len, key_len) = attention_mask
            .bool_mask
            .shape()
            .dims2()
            .expect("input mask must have two dimensions");
        QueryKeyAttentionMask {
            bool_mask: attention_mask
                .bool_mask
                .reshape((batch_len, 1, 1, key_len))
                .expect("Cannot reshape input mask"),
        }
    }
}

impl QueryKeyAttentionMask {
    /// Use the attention mask to mask logits.
    ///
    /// * input - Tensor to which the mask is applied.
    ///   *Shape:* `(batch_size, heads, query_len, key_len)`
    ///
    /// Returns: Logits with the attention mask applied.
    /// *Shape:* `(batch_size, heads, query_len, key_len)`
    pub fn apply_logit_mask(&self, input: &Tensor) -> Result<Tensor, QueryKeyAttentionMaskError> {
        // Underflows to -inf for more narrow floating point types, which
        // is ok for masking.
        let blocked_value = Tensor::try_from(f32::MIN)
            .and_then(|xs| xs.broadcast_as(input.shape()))
            .context(ApplyLogitsMaskSnafu)?;
        self.bool_mask
            .broadcast_as(input.shape())
            .and_then(|xs| xs.where_cond(input, &blocked_value))
            .context(ApplyLogitsMaskSnafu)
    }

    /// Merge this attention mask with another attention mask.
    pub fn intersect(
        &self,
        other: &QueryKeyAttentionMask,
    ) -> Result<QueryKeyAttentionMask, QueryKeyAttentionMaskError> {
        Ok(QueryKeyAttentionMask {
            bool_mask: self
                .bool_mask
                .broadcast_mul(&other.bool_mask)
                .context(IntersectMasksSnafu)?,
        })
    }
}

/// Errors for attention masks.
#[derive(Debug, Snafu)]
pub enum AttentionMaskError {
    #[snafu(display("Cannot concatenate masks"))]
    ConcatMasks { source: candle_core::Error },

    #[snafu(display("Attention mask must be 2D, was {}D", n_dims))]
    InvalidDims { n_dims: usize },
}

/// Attention mask.
///
/// Sequence elements for which the corresponding mask element is set to
/// `False` are ignored during attention calculation. Guaranteed to be
/// a 2D array.
#[derive(Clone, Debug)]
pub struct AttentionMask {
    bool_mask: Tensor,
}

impl AttentionMask {
    /// Create an input attention mask.
    ///
    /// * `bool_mask` - Boolean mask tensor.
    ///   *Shape:* `(batch_size, seq_len)`
    pub fn new(bool_mask: Tensor) -> Result<Self, AttentionMaskError> {
        let n_dims = bool_mask.dims().len();
        ensure!(n_dims == 2, InvalidDimsSnafu { n_dims });
        Ok(AttentionMask { bool_mask })
    }

    /// Extend the mask using another mask.
    pub fn extend(&self, other: &Self) -> Result<Self, AttentionMaskError> {
        Ok(AttentionMask {
            bool_mask: Tensor::cat(&[&self.bool_mask, &other.bool_mask], 1)
                .context(ConcatMasksSnafu)?,
        })
    }
}

#[derive(Debug, Snafu)]
pub enum CausalMaskError {
    #[snafu(display("Cannot create causal mask"))]
    CreateMask { source: candle_core::Error },

    #[snafu(display("Key has invalid number of dimensions"))]
    KeyDim { source: candle_core::Error },

    #[snafu(display("Query has invalid number of dimensions"))]
    QueryDim { source: candle_core::Error },

    #[snafu(display("Query length {query_len} must not be larger than key length {key_len}"))]
    QueryLen { key_len: usize, query_len: usize },

    #[snafu(display("Cannot slice causal mask to key/query size"))]
    SliceMask { source: candle_core::Error },
}

/// Trait for creating causal masks.
pub trait CausalMask: Sized {
    type Error;

    /// Create a causal mask for the given query and key.
    ///
    /// A causal mask ensures that tokens cannot attend to succeeding tokens.
    ///
    /// * `query` - Query tensor.
    ///   *Shape:* `(batch_size, heads, query_len, width)`
    /// * `key` - Key tensor.
    ///   *Shape:* `(batch_size, heads, key_len, width)`
    fn causal_mask(query: &Tensor, key: &Tensor) -> Result<Self, Self::Error>;
}

impl CausalMask for QueryKeyAttentionMask {
    type Error = CausalMaskError;

    fn causal_mask(query: &Tensor, key: &Tensor) -> Result<Self, Self::Error> {
        let (_, _, query_len, _) = query.shape().dims4().context(QueryDimSnafu)?;
        let (_, _, key_len, _) = key.shape().dims4().context(KeyDimSnafu)?;

        // Slicing will fail down the line if the query length is greater than
        // the key length.
        ensure!(query_len <= key_len, QueryLenSnafu { key_len, query_len });

        let causal_mask = Tensor::tril2(key_len, DType::U32, key.device())
            .and_then(|mask| mask.reshape((1, 1, key_len, key_len)))
            .context(CreateMaskSnafu)?;
        Ok(Self {
            bool_mask: causal_mask
                .i((.., .., key_len - query_len..key_len, ..key_len))
                .context(SliceMaskSnafu)?,
        })
    }
}
