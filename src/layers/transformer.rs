/// Transformer building blocks.
use candle_core::{Module, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::error::Result;
use crate::layers::identity::Identity;

/// Layer normalizations used in a transformer embedding layer.
///
/// By default, all the normalizations are disabled by setting the layer
/// normalization to the `Identity` module. Therefore, only normalizations
/// that are needed have to be set.
pub struct EmbeddingLayerNorms {
    /// Normalization of the embeddings.
    pub embed_output_layer_norm: Box<dyn Module>,

    /// Normalization of the projection layer.
    pub proj_output_layer_norm: Box<dyn Module>,
}

impl Default for EmbeddingLayerNorms {
    fn default() -> Self {
        EmbeddingLayerNorms {
            embed_output_layer_norm: Box::new(Identity),
            proj_output_layer_norm: Box::new(Identity),
        }
    }
}

/// Dropouts used in a transformer embedding layer.
///
/// By default, all the dropouts are disabled by setting the dropout
/// to the `Identity` module. Therefore, only dropouts that are needed have
/// to be set.
pub struct EmbeddingLayerDropouts {
    /// Dropout of the embeddings.
    embed_output_layer_dropout: Box<dyn Module>,

    /// Dropout of the projection layer.
    proj_output_layer_dropout: Box<dyn Module>,
}

impl Default for EmbeddingLayerDropouts {
    fn default() -> Self {
        EmbeddingLayerDropouts {
            embed_output_layer_dropout: Box::new(Identity),
            proj_output_layer_dropout: Box::new(Identity),
        }
    }
}

/// Transformer embeddings layer.
///
/// This is a generic transformer embedding layer. The layer always has piece
/// embeddings and can optionally have position embeddings, type embeddings,
/// and a projection of embeddings to the model's hidden size.
pub struct TransformerEmbeddings {
    piece_embeddings: Embedding,
    type_embeddings: Option<Embedding>,
    position_embeddings: Option<Embedding>,
    projection: Option<Embedding>,
    dropouts: EmbeddingLayerDropouts,
    layer_norms: EmbeddingLayerNorms,
}

impl TransformerEmbeddings {
    /// Construct an embeddings layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        dropouts: EmbeddingLayerDropouts,
        embedding_width: usize,
        hidden_width: usize,
        layer_norms: EmbeddingLayerNorms,
        n_pieces: usize,
        n_positions: Option<usize>,
        n_types: Option<usize>,
    ) -> Result<Self> {
        let piece_embeddings = embedding(
            n_pieces,
            embedding_width,
            vb.push_prefix("piece_embeddings"),
        )?;

        let type_embeddings = n_types
            .map(|n_types| embedding(n_types, embedding_width, vb.push_prefix("type_embeddings")))
            .transpose()?;

        let position_embeddings = n_positions
            .map(|n_positions| {
                embedding(
                    n_positions,
                    embedding_width,
                    vb.push_prefix("position_embeddings"),
                )
            })
            .transpose()?;

        let projection = if embedding_width != hidden_width {
            Some(embedding(
                embedding_width,
                hidden_width,
                vb.push_prefix("projection"),
            )?)
        } else {
            None
        };

        Ok(TransformerEmbeddings {
            dropouts,
            layer_norms,
            piece_embeddings,
            position_embeddings,
            projection,
            type_embeddings,
        })
    }

    /// Get position identifiers _[0..seq_len)_.
    fn get_positions(x: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = x.shape().dims2()?;
        Ok(Tensor::arange(0, seq_len as i64, x.device())?.reshape((1, seq_len))?)
    }

    /// Get all-zero type identifiers for the given tensor.
    fn get_type_ids(x: &Tensor) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Calculate the piece embeddings.
    pub fn forward(
        &self,
        piece_ids: &Tensor,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut embeddings = self.piece_embeddings.forward(piece_ids)?;

        if let Some(type_embeddings) = &self.type_embeddings {
            let type_ids = match type_ids {
                Some(type_ids) => type_ids.clone(),
                None => Self::get_type_ids(piece_ids)?,
            };
            embeddings = (embeddings + type_embeddings.forward(&type_ids)?)?;
        }
        if let Some(position_embeddings) = &self.position_embeddings {
            let positions = match positions {
                Some(positions) => positions.clone(),
                None => Self::get_positions(piece_ids)?,
            };
            embeddings = (embeddings + position_embeddings.forward(&positions)?)?;
        }

        embeddings = self
            .layer_norms
            .embed_output_layer_norm
            .forward(&embeddings)?;
        embeddings = self
            .dropouts
            .embed_output_layer_dropout
            .forward(&embeddings)?;

        if let Some(projection) = &self.projection {
            embeddings = projection.forward(&embeddings)?;
            embeddings = self
                .layer_norms
                .proj_output_layer_norm
                .forward(&embeddings)?;
            embeddings = self
                .dropouts
                .proj_output_layer_dropout
                .forward(&embeddings)?;
        }

        Ok(embeddings)
    }
}
