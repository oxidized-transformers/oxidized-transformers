use candle_core::Tensor;
use snafu::{ResultExt, Snafu};

/// Errors for attention masks.
#[derive(Debug, Snafu)]
pub enum AttentionMaskError {
    #[snafu(display("Cannot apply logits mask"))]
    ApplyLogitsMask { source: candle_core::Error },

    #[snafu(display("Cannot intersect masks"))]
    IntersectMasks { source: candle_core::Error },

    #[snafu(display("Cannot reshape 2D mask to 4D"))]
    ReshapeMask { source: candle_core::Error },
}

/// Attention mask.
///
/// Sequence elements for which the corresponding mask element is set to
/// `False` are ignored during attention calculation.
#[derive(Clone, Debug)]
pub struct AttentionMask {
    bool_mask: Tensor,
}

impl AttentionMask {
    /// Create an attention mask.
    ///
    /// * `bool_mask` - Boolean mask tensor.
    ///   *Shape:* `(batch_size, seq_len)`
    pub fn new(bool_mask: Tensor) -> Result<Self, AttentionMaskError> {
        let bool_mask = match bool_mask.shape().dims2() {
            Ok((batch_len, key_len)) => bool_mask
                .reshape((batch_len, 1, 1, key_len))
                .context(ReshapeMaskSnafu)?,
            _ => bool_mask,
        };

        Ok(AttentionMask { bool_mask })
    }

    /// Use the attention mask to mask attention logits.
    /// * input - Tensor to which the mask is applied.
    ///   *Shape:* `(batch_size, heads, query_len, key_len)`
    ///
    /// Returns: Logits with the attention mask applied.
    /// *Shape:* `(batch_size, heads, query_len, key_len)`
    pub fn apply_logit_mask(&self, input: &Tensor) -> Result<Tensor, AttentionMaskError> {
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
    pub fn intersect(&self, other: &AttentionMask) -> Result<AttentionMask, AttentionMaskError> {
        Ok(AttentionMask {
            bool_mask: self
                .bool_mask
                .broadcast_mul(&other.bool_mask)
                .context(IntersectMasksSnafu)?,
        })
    }
}
