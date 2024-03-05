use candle_core::Tensor;
use candle_nn::VarBuilder;
use std::fmt::Debug;

use crate::error::BoxedError;

/// Trait for embedding layers.
pub trait Embeddings {
    /// Look up the embeddings for the given piece identifiers.
    ///
    /// * `piece_ids` - Piece identifiers.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `train` - Whether to train the layer.
    /// * `positions` - Input positions.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `type_ids` - Input type identifiers.
    ///   *Shape:* `(batch_size, seq_len)`
    fn forward(
        &self,
        piece_ids: &Tensor,
        train: bool,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor, BoxedError>;
}

/// Trait for building embedding layers.
pub trait BuildEmbeddings: Debug {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Embeddings>, BoxedError>;
}
