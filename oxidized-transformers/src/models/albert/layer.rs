use candle_core::Tensor;
use snafu::{ResultExt, Snafu};

use crate::{
    architectures::{BuildEncoderLayer, EncoderLayer},
    error::BoxedError,
    layers::{attention::AttentionMask, transformer::TransformerLayerConfig},
};

/// ALBERT layer group errors.
#[derive(Debug, Snafu)]
enum BuildAlbertLayerGroupError {
    #[snafu(display("Cannot build ALBERT group layer {n}"))]
    BuildTransformerLayer {
        source: crate::error::BoxedError,
        n: usize,
    },
}

/// ALBERT layer group configuration
///
/// A group's layer can in turn consist of multiple layers (though this is
/// rarely used in ALBERT models).
#[derive(Debug)]
pub struct AlbertLayerGroupConfig {
    n_layers_per_group: usize,
    transformer_layer: TransformerLayerConfig,
}

impl AlbertLayerGroupConfig {
    /// Number of layers within a group layer.
    pub fn n_layers_per_group(mut self, n_layers_per_group: usize) -> Self {
        self.n_layers_per_group = n_layers_per_group;
        self
    }

    /// Transformer layer configuration.
    pub fn transformer_layer(mut self, transformer_layer: TransformerLayerConfig) -> Self {
        self.transformer_layer = transformer_layer;
        self
    }
}

impl Default for AlbertLayerGroupConfig {
    fn default() -> Self {
        Self {
            n_layers_per_group: 1,
            transformer_layer: TransformerLayerConfig::default(),
        }
    }
}

impl BuildEncoderLayer for AlbertLayerGroupConfig {
    fn build_encoder_layer(
        &self,
        vb: candle_nn::VarBuilder,
    ) -> Result<Box<dyn crate::architectures::EncoderLayer>, crate::error::BoxedError> {
        let layers = (0..self.n_layers_per_group)
            .map({
                |n| {
                    self.transformer_layer
                        .build_encoder_layer(vb.push_prefix(format!("group_layer_{n}")))
                        .context(BuildTransformerLayerSnafu { n })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Box::new(AlbertLayerGroup { layers }))
    }
}

pub struct AlbertLayerGroup {
    layers: Vec<Box<dyn EncoderLayer>>,
}

impl EncoderLayer for AlbertLayerGroup {
    fn forward_t(
        &self,
        input: &Tensor,
        mask: &AttentionMask,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, BoxedError> {
        let mut layer_output = input.clone();

        for layer in &self.layers {
            layer_output = layer.forward_t(&layer_output, mask, positions, train)?;
        }

        Ok(layer_output)
    }
}
