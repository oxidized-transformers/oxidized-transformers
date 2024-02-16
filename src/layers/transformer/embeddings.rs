use crate::error::BoxedError;
use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use snafu::{ResultExt, Snafu};

use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;

/// Configuration for transformer embedding layer.
#[derive(Debug)]
pub struct TransformerEmbeddingsConfig {
    /// Dropout to apply to the embeddings.
    embedding_dropout: Box<dyn BuildModule>,

    /// Layer normalization to apply to the embeddings.
    embedding_layer_norm: Box<dyn BuildModule>,

    /// Width of the embeddings.
    embedding_width: usize,

    /// Width of the transformer.
    ///
    /// The embedding layer will use a projection if the hidden width is
    /// not equal to the embedding width.
    hidden_width: usize,

    /// Number of position embeddings.
    n_positions: Option<usize>,

    /// Vocabulary size (number of embeddings).
    n_pieces: usize,

    /// Token type vocabulary size (number of token type embeddings).
    n_types: Option<usize>,

    /// Dropout to apply after embedding projection.
    projection_dropout: Box<dyn BuildModule>,

    /// Layer normalization to apply to the projection layer.
    projection_layer_norm: Box<dyn BuildModule>,
}

impl TransformerEmbeddingsConfig {
    pub fn build(
        &self,
        vb: VarBuilder,
    ) -> Result<TransformerEmbeddings, TransformerEmbeddingsError> {
        let piece_embeddings = embedding(
            self.n_pieces,
            self.embedding_width,
            vb.push_prefix("piece_embeddings"),
        )
        .context(ConstructionSnafu)?;

        let type_embeddings = self
            .n_types
            .map(|n_types| {
                embedding(
                    n_types,
                    self.embedding_width,
                    vb.push_prefix("type_embeddings"),
                )
            })
            .transpose()
            .context(ConstructionSnafu)?;

        let position_embeddings = self
            .n_positions
            .map(|n_positions| {
                embedding(
                    n_positions,
                    self.embedding_width,
                    vb.push_prefix("position_embeddings"),
                )
            })
            .transpose()
            .context(ConstructionSnafu)?;

        let projection = if self.embedding_width != self.hidden_width {
            Some(
                embedding(
                    self.embedding_width,
                    self.hidden_width,
                    vb.push_prefix("projection"),
                )
                .context(ConstructionSnafu)?,
            )
        } else {
            None
        };

        Ok(TransformerEmbeddings {
            embedding_dropout: self
                .embedding_dropout
                .build(vb.push_prefix("embedding_dropout"))
                .context(BuildDropoutSnafu)?,
            embedding_layer_norm: self
                .embedding_layer_norm
                .build(vb.push_prefix("embedding_layer_norm"))
                .context(BuildLayerNormSnafu)?,
            piece_embeddings,
            position_embeddings,
            projection,
            projection_dropout: self
                .projection_dropout
                .build(vb.push_prefix("projection_dropout"))
                .context(BuildDropoutSnafu)?,
            projection_layer_norm: self
                .projection_layer_norm
                .build(vb.push_prefix("projection_layer_norm"))
                .context(BuildLayerNormSnafu)?,

            type_embeddings,
        })
    }

    /// Dropout to apply to the embeddings.
    ///
    /// Default: `Identity`.
    pub fn embedding_dropout(mut self, embedding_dropout: Box<dyn BuildModule>) -> Self {
        self.embedding_dropout = embedding_dropout;
        self
    }

    /// Layer normalization to apply to the embeddings.
    ///
    /// Default: `Identity`.
    pub fn embedding_layer_norm(mut self, embedding_layer_norm: Box<dyn BuildModule>) -> Self {
        self.embedding_layer_norm = embedding_layer_norm;
        self
    }

    /// Width of the embeddings.
    ///
    /// Default: `768`.
    pub fn embedding_width(mut self, embedding_width: usize) -> Self {
        self.embedding_width = embedding_width;
        self
    }

    /// Width of the transformer.
    ///
    /// The embedding layer will use a projection if the hidden width is
    /// not equal to the embedding width.
    ///
    /// Default: `768`.
    pub fn hidden_width(mut self, hidden_width: usize) -> Self {
        self.hidden_width = hidden_width;
        self
    }

    /// Number of position embeddings.
    ///
    /// Default: `None`.
    pub fn n_positions(mut self, n_positions: Option<usize>) -> Self {
        self.n_positions = n_positions;
        self
    }

    /// Vocabulary size (number of embeddings).
    ///
    /// Default: `30000`.
    pub fn n_pieces(mut self, n_pieces: usize) -> Self {
        self.n_pieces = n_pieces;
        self
    }

    /// Token type vocabulary size (number of token type embeddings).
    ///
    /// Default: `None`.
    pub fn n_types(mut self, n_types: Option<usize>) -> Self {
        self.n_types = n_types;
        self
    }

    /// Dropout to apply after embedding projection.
    ///
    /// Default: `Identity`.
    pub fn projection_dropout(mut self, projection_dropout: Box<dyn BuildModule>) -> Self {
        self.projection_dropout = projection_dropout;
        self
    }

    /// Layer normalization to apply to the projection layer.
    ///
    /// Default: `Identity`.
    pub fn projection_layer_norm(mut self, projection_layer_norm: Box<dyn BuildModule>) -> Self {
        self.projection_layer_norm = projection_layer_norm;
        self
    }
}

impl Default for TransformerEmbeddingsConfig {
    fn default() -> Self {
        Self {
            embedding_dropout: Box::new(Identity),
            embedding_layer_norm: Box::new(Identity),
            embedding_width: 768,
            hidden_width: 768,
            n_positions: None,
            n_pieces: 30000,
            n_types: None,
            projection_dropout: Box::new(Identity),
            projection_layer_norm: Box::new(Identity),
        }
    }
}

/// Errors for transformer embeddings.
#[derive(Debug, Snafu)]
pub enum TransformerEmbeddingsError {
    #[snafu(display("Cannot build dropout"))]
    BuildDropout { source: BoxedError },

    #[snafu(display("Cannot build layer norm"))]
    BuildLayerNorm { source: BoxedError },

    #[snafu(display("Cannot construct embeddings layer"))]
    Construction { source: candle_core::Error },

    #[snafu(display("Cannot normalize embeddings or apply dropout"))]
    NormalizeDropout { source: candle_core::Error },

    #[snafu(display("Cannot lookup piece embeddings"))]
    PieceEmbeddings { source: candle_core::Error },

    #[snafu(display("Cannot lookup position embeddings"))]
    PositionEmbeddings { source: candle_core::Error },

    #[snafu(display("Cannot project embeddings to hidden size"))]
    Projection { source: candle_core::Error },

    #[snafu(display("Cannot lookup type embeddings"))]
    TypeEmbeddings { source: candle_core::Error },
}

/// Transformer embeddings layer.
///
/// This is a generic transformer embedding layer. The layer always has piece
/// embeddings and can optionally have position embeddings, type embeddings,
/// and a projection of embeddings to the model's hidden size.
pub struct TransformerEmbeddings {
    embedding_dropout: Box<dyn ModuleT>,
    embedding_layer_norm: Box<dyn ModuleT>,
    piece_embeddings: Embedding,
    type_embeddings: Option<Embedding>,
    position_embeddings: Option<Embedding>,
    projection: Option<Embedding>,
    projection_dropout: Box<dyn ModuleT>,
    projection_layer_norm: Box<dyn ModuleT>,
}

impl TransformerEmbeddings {
    /// Get position identifiers _[0..seq_len)_.
    fn get_positions(x: &Tensor) -> Result<Tensor, TransformerEmbeddingsError> {
        let (_, seq_len) = x.shape().dims2().context(PositionEmbeddingsSnafu)?;
        Tensor::arange(0, seq_len as i64, x.device())
            .and_then(|xs| xs.reshape((1, seq_len)))
            .context(PositionEmbeddingsSnafu)
    }

    /// Get all-zero type identifiers for the given tensor.
    fn get_type_ids(x: &Tensor) -> Result<Tensor, TransformerEmbeddingsError> {
        x.zeros_like().context(TypeEmbeddingsSnafu)
    }

    /// Calculate the piece embeddings.
    pub fn forward(
        &self,
        piece_ids: &Tensor,
        train: bool,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor, TransformerEmbeddingsError> {
        let mut embeddings = self
            .piece_embeddings
            .forward(piece_ids)
            .context(PieceEmbeddingsSnafu)?;

        if let Some(type_embeddings) = &self.type_embeddings {
            let type_ids = match type_ids {
                Some(type_ids) => type_ids.clone(),
                None => Self::get_type_ids(piece_ids)?,
            };
            embeddings = type_embeddings
                .forward(&type_ids)
                .and_then(|xs| embeddings + xs)
                .context(TypeEmbeddingsSnafu)?;
        }
        if let Some(position_embeddings) = &self.position_embeddings {
            let positions = match positions {
                Some(positions) => positions.clone(),
                None => Self::get_positions(piece_ids)?,
            };
            embeddings = position_embeddings
                .forward(&positions)
                .and_then(|xs| embeddings + xs)
                .context(PositionEmbeddingsSnafu)?;
        }

        embeddings = self
            .embedding_layer_norm
            .forward_t(&embeddings, train)
            .and_then(|xs| self.embedding_dropout.forward_t(&xs, train))
            .context(NormalizeDropoutSnafu)?;

        if let Some(projection) = &self.projection {
            embeddings = projection
                .forward(&embeddings)
                .and_then(|xs| self.projection_layer_norm.forward_t(&xs, train))
                .and_then(|xs| self.projection_dropout.forward_t(&xs, train))
                .context(ProjectionSnafu)?;
        }

        Ok(embeddings)
    }
}
