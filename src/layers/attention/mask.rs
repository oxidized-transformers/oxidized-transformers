use candle_core::Tensor;
use snafu::{ensure, ResultExt, Snafu};

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
    pub(crate) bool_mask: Tensor,
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
