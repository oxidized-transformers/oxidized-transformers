use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::architectures::{BuildEmbeddings, Embeddings};
use crate::error::BoxedError;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerEmbeddingsError};

#[derive(Debug)]
/// RoBERTa embeddings configuration.
pub struct RobertaEmbeddingsConfig {
    padding_id: u32,
    transformer_embeddings: Box<dyn BuildEmbeddings>,
}

impl Default for RobertaEmbeddingsConfig {
    fn default() -> Self {
        Self {
            padding_id: 1,
            transformer_embeddings: Box::<TransformerEmbeddingsConfig>::default(),
        }
    }
}

impl RobertaEmbeddingsConfig {
    /// Padding token id.
    ///
    /// Default: 1
    pub fn padding_id(mut self, padding_id: u32) -> Self {
        self.padding_id = padding_id;
        self
    }

    /// Transformer embeddings configuration.
    ///
    /// Default: `TransformerEmbeddingsConfig::default()`
    pub fn transformer_embeddings(
        mut self,
        transformer_embeddings: Box<dyn BuildEmbeddings>,
    ) -> Self {
        self.transformer_embeddings = transformer_embeddings;
        self
    }
}

impl BuildEmbeddings for RobertaEmbeddingsConfig {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Embeddings>, BoxedError> {
        Ok(Box::new(RobertaEmbeddings {
            inner: self.transformer_embeddings.build(vb)?,
            padding_id: self.padding_id,
        }))
    }
}

/// RoBERTa embeddings errors.
#[derive(Debug, Snafu)]
pub enum RobertaEmbeddingsError {
    #[snafu(display("Error computing transformer embeddings"))]
    TransformerEmbeddings { source: TransformerEmbeddingsError },

    #[snafu(display("Cannot compute token mask"))]
    ComputeMask { source: candle_core::Error },

    #[snafu(display("Cannot compute token positions"))]
    ComputePositions { source: candle_core::Error },

    #[snafu(display("Cannot convert padding id to tensor"))]
    PaddingIdToTensor { source: candle_core::Error },
}

/// RoBERTa (Li et al., 2019) embeddings.
pub struct RobertaEmbeddings {
    inner: Box<dyn Embeddings>,
    padding_id: u32,
}

impl Embeddings for RobertaEmbeddings {
    fn forward(
        &self,
        piece_ids: &Tensor,
        train: bool,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor, BoxedError> {
        let positions = match positions {
            Some(positions) => positions.clone(),
            None => {
                let mask = piece_ids
                    .ne(self.padding_id)
                    // This is a bit weird, but cumsum below uses matmul internally,
                    // which only works with floating point tensors.
                    .and_then(|xs| xs.to_dtype(DType::F32))
                    .context(ComputeMaskSnafu)?;
                let padding_id = Tensor::full(
                    self.padding_id as f32,
                    piece_ids.shape(),
                    piece_ids.device(),
                )
                .context(PaddingIdToTensorSnafu)?;
                mask.cumsum(1)
                    .and_then(|xs| xs.mul(&mask))
                    .and_then(|xs| xs.add(&padding_id))
                    // And back to indices that we can use for embedding lookups.
                    .and_then(|xs| xs.to_dtype(DType::U32))
                    .context(ComputePositionsSnafu)?
            }
        };

        self.inner
            .forward(piece_ids, train, Some(&positions), type_ids)
    }
}
