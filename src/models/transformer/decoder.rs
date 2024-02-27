/// Transformer decoder.
use candle_core::{ModuleT, Tensor};
use candle_nn::VarBuilder;
use snafu::{ResultExt, Snafu};

use crate::architectures::BuildArchitecture;
use crate::architectures::{BuildDecoderLayer, Decoder, DecoderLayer, DecoderOutput};
use crate::error::BoxedError;
use crate::kv_cache::{KeyValueCache, LayerKeyValueCache};
use crate::layers::attention::AttentionMask;
use crate::layers::build_module::BuildModule;
use crate::layers::identity::Identity;
use crate::layers::transformer::{
    TransformerEmbeddings, TransformerEmbeddingsConfig, TransformerEmbeddingsError,
    TransformerLayerConfig,
};

/// Transformer decoder configuration.
#[derive(Debug)]
pub struct TransformerDecoderConfig {
    embeddings: TransformerEmbeddingsConfig,
    layer: Box<dyn BuildDecoderLayer<Cache = LayerKeyValueCache>>,
    n_hidden_layers: usize,
    output_layer_norm: Box<dyn BuildModule>,
}

impl TransformerDecoderConfig {
    /// Decoder embeddings.
    ///
    /// Default: `TransformerEmbeddingsConfig::default()`
    pub fn embeddings(mut self, embeddings: TransformerEmbeddingsConfig) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Decoder layer.
    ///
    /// Default: `TransformerLayerConfig::default()`
    pub fn layer(mut self, layer: Box<dyn BuildDecoderLayer<Cache = LayerKeyValueCache>>) -> Self {
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

impl BuildArchitecture for TransformerDecoderConfig {
    type Architecture = TransformerDecoder;

    fn build(&self, vb: VarBuilder) -> Result<Self::Architecture, BoxedError> {
        let embeddings = self
            .embeddings
            .build(vb.push_prefix("embeddings"))
            .context(BuildTransformerEmbeddingsSnafu)?;

        let layers = (0..self.n_hidden_layers)
            .map(|n| {
                self.layer
                    .build_decoder_layer(vb.push_prefix(format!("layer_{n}")))
                    .context(BuildTransformerLayerSnafu)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TransformerDecoder {
            embeddings,
            layers,
            output_layer_norm: self
                .output_layer_norm
                .build(vb.push_prefix("output_layer_norm"))
                .context(BuildLayerNormSnafu)?,
        })
    }
}

impl Default for TransformerDecoderConfig {
    fn default() -> Self {
        Self {
            embeddings: TransformerEmbeddingsConfig::default(),
            layer: Box::<TransformerLayerConfig>::default(),
            n_hidden_layers: 12,
            output_layer_norm: Box::new(Identity),
        }
    }
}

/// Transformer decoder errors.
#[derive(Debug, Snafu)]
pub enum TransformerDecoderError {
    #[snafu(display("Cannot build layer norm"))]
    BuildLayerNorm { source: BoxedError },

    #[snafu(display("Cannot construct or apply embeddings"))]
    BuildTransformerEmbeddings { source: TransformerEmbeddingsError },

    #[snafu(display("Cannot build transformer layer"))]
    BuildTransformerLayer { source: BoxedError },

    #[snafu(display("Cannot construct or apply embeddings"))]
    Embedding { source: TransformerEmbeddingsError },

    #[snafu(display("Cannot construct or apply layer norm"))]
    LayerNorm { source: candle_core::Error },

    #[snafu(display("Cannot apply transformer layer"))]
    TransformerLayer { source: BoxedError },
}

/// Decoder using the transformer architecture.
pub struct TransformerDecoder {
    embeddings: TransformerEmbeddings,
    layers: Vec<Box<dyn DecoderLayer<Cache = LayerKeyValueCache>>>,
    output_layer_norm: Box<dyn ModuleT>,
}

impl Decoder for TransformerDecoder {
    type Cache = KeyValueCache;

    fn forward_t(
        &self,
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<DecoderOutput, BoxedError> {
        let embeddings = self
            .embeddings
            .forward(piece_ids, train, None, None)
            .context(EmbeddingSnafu)?;

        let mut layer_output = embeddings;
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let next_layer_output = layer
                .forward_t(
                    &layer_output,
                    attention_mask,
                    &mut cache[layer_idx],
                    positions,
                    train,
                )
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

        Ok(DecoderOutput::new(layer_outputs))
    }
}
