//! Transformer encoder.
use candle_core::{ModuleT, Tensor};
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::architectures::{BuildArchitecture, BuildEmbeddings, Embeddings};
use crate::architectures::{BuildEncoderLayer, Encoder, EncoderLayer, EncoderOutput};
use crate::error::BoxedError;
use crate::layers::attention::AttentionMask;
use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};

/// Transformer encoder configuration.
#[derive(Debug)]
pub struct TransformerEncoderConfig {
    embeddings: Box<dyn BuildEmbeddings>,
    layer: Box<dyn BuildEncoderLayer>,
    n_hidden_layers: usize,
    output_layer_norm: Box<dyn BuildModule>,
}

impl TransformerEncoderConfig {
    /// Encoder embeddings.
    ///
    /// Default: `TransformerEmbeddingsConfig::default()`
    pub fn embeddings(mut self, embeddings: Box<dyn BuildEmbeddings>) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Encoder layer.
    ///
    /// Default: `TransformerLayerConfig::default()`
    pub fn layer(mut self, layer: Box<dyn BuildEncoderLayer>) -> Self {
        self.layer = layer;
        self
    }

    /// Number of hidden layers.
    ///
    /// Default: `12`
    pub fn n_hidden_layers(mut self, n_hidden_layers: usize) -> Self {
        self.n_hidden_layers = n_hidden_layers;
        self
    }

    /// Output layer normalization module.
    ///
    /// Default: `Identity`
    pub fn output_layer_norm(mut self, output_layer_norm: Box<dyn BuildModule>) -> Self {
        self.output_layer_norm = output_layer_norm;
        self
    }
}

impl BuildArchitecture for TransformerEncoderConfig {
    type Architecture = TransformerEncoder;

    fn build(&self, vb: VarBuilder) -> Result<Self::Architecture, BoxedError> {
        let embeddings = self
            .embeddings
            .build(vb.push_prefix("embeddings"))
            .context(BuildTransformerEmbeddingsSnafu)?;

        let layers = (0..self.n_hidden_layers)
            .map(|n| {
                self.layer
                    .build_encoder_layer(vb.push_prefix(format!("layer_{n}")))
                    .context(BuildTransformerLayerSnafu)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TransformerEncoder {
            embeddings,
            layers,
            output_layer_norm: self
                .output_layer_norm
                .build(vb.push_prefix("output_layer_norm"))
                .context(BuildLayerNormSnafu)?,
        })
    }
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            embeddings: Box::<TransformerEmbeddingsConfig>::default(),
            layer: Box::<TransformerLayerConfig>::default(),
            n_hidden_layers: 12,
            output_layer_norm: Box::new(Identity),
        }
    }
}

/// Transformer encoder errors.
#[derive(Debug, Snafu)]
pub enum TransformerEncoderError {
    #[snafu(display("Cannot build layer norm"))]
    BuildLayerNorm { source: BoxedError },

    #[snafu(display("Cannot build embeddings"))]
    BuildTransformerEmbeddings { source: BoxedError },

    #[snafu(display("Cannot build transformer layer"))]
    BuildTransformerLayer { source: BoxedError },

    #[snafu(display("Cannot apply embeddings"))]
    Embedding { source: BoxedError },

    #[snafu(display("Cannot apply layer norm"))]
    LayerNorm { source: candle_core::Error },

    #[snafu(display("Cannot apply transformer layer"))]
    TransformerLayer { source: BoxedError },
}

/// Encoder using the transformer architecture.
pub struct TransformerEncoder {
    embeddings: Box<dyn Embeddings>,
    layers: Vec<Box<dyn EncoderLayer>>,
    output_layer_norm: Box<dyn ModuleT>,
}

impl Encoder for TransformerEncoder {
    fn forward_t(
        &self,
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<EncoderOutput, BoxedError> {
        let embeddings = self
            .embeddings
            .forward(piece_ids, train, positions, type_ids)
            .context(EmbeddingSnafu)?;

        let mut layer_output = embeddings;
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        layer_outputs.push(layer_output.clone());

        for layer in &self.layers {
            let next_layer_output = layer
                .forward_t(&layer_output, attention_mask, positions, train)
                .context(TransformerLayerSnafu)?;

            layer_outputs.push(next_layer_output.clone());
            layer_output = next_layer_output;
        }

        if let Some(last) = layer_outputs.last_mut() {
            *last = self
                .output_layer_norm
                .forward_t(last, train)
                .context(LayerNormSnafu)?;
        }

        Ok(EncoderOutput::new(layer_outputs))
    }
}
