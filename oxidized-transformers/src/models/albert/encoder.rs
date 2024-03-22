use std::sync::OnceLock;

use candle_core::Tensor;
use regex::Regex;
use serde::{Deserialize, Serialize};
use snafu::{ensure, ResultExt, Snafu};

use crate::architectures::{
    BuildArchitecture, BuildEmbeddings, BuildEncoderLayer, Embeddings, Encoder, EncoderLayer,
    EncoderOutput,
};
use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{
    AttentionHeads, AttentionMask, QkvMode, SDPAConfig, SelfAttentionConfig,
};
use crate::layers::dropout::DropoutConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::albert::AlbertLayerGroupConfig;
use crate::models::hf::FromHF;

/// HF ALBERT model types
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HFModelType {
    Albert,
}

/// HF ALBERT encoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFAlbertEncoderConfig {
    attention_probs_dropout_prob: f32,
    embedding_size: usize,
    hidden_act: Activation,
    hidden_dropout_prob: f32,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    inner_group_num: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: HFModelType,
    num_attention_heads: usize,
    num_hidden_groups: usize,
    num_hidden_layers: usize,
    pad_token_id: u32,
    type_vocab_size: usize,
    vocab_size: usize,
}

impl TryFrom<HFAlbertEncoderConfig> for AlbertEncoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFAlbertEncoderConfig) -> Result<Self, Self::Error> {
        let n_hidden_groups = hf_config.num_hidden_groups;
        let n_hidden_layers = hf_config.num_hidden_layers;

        ensure!(
            n_hidden_layers % n_hidden_groups == 0,
            IncorrectNHiddenGroupsSnafu {
                n_hidden_groups,
                n_hidden_layers
            }
        );

        let attention_probs_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));
        let hidden_dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let embedding_layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps)
                .size(hf_config.embedding_size),
        );
        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_dropout(hidden_dropout.clone())
            .embedding_layer_norm(embedding_layer_norm)
            .embedding_width(hf_config.embedding_size)
            .hidden_width(hf_config.hidden_size)
            .n_pieces(hf_config.vocab_size)
            .n_positions(Some(hf_config.max_position_embeddings))
            .n_types(Some(hf_config.type_vocab_size));

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_attention_heads,
                qkv_mode: QkvMode::Separate,
            })
            .attention_scorer(Box::new(
                SDPAConfig::default().dropout(attention_probs_dropout),
            ))
            .hidden_width(hf_config.hidden_size);

        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .dropout(hidden_dropout)
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size);

        let layer = AlbertLayerGroupConfig::default()
            .n_layers_per_group(hf_config.inner_group_num)
            .transformer_layer(
                TransformerLayerConfig::default()
                    .attention(attention)
                    .attn_residual_layer_norm(layer_norm.clone())
                    .feedforward(feedforward)
                    .ffn_residual_layer_norm(layer_norm),
            );

        Ok(Self::default()
            .embeddings(Box::new(embeddings))
            .layer_group(Box::new(layer))
            .n_hidden_groups(n_hidden_groups)
            .n_hidden_layers(n_hidden_layers))
    }
}

/// ALBERT encoder configuration.
///
/// Configuration for [AlbertEncoder]. ALBERT differs from an encoder like
/// BERT in two ways:
///
/// 1. The embedding size can be different from the hidden size. A projection
///    matrix is used when the sizes differ.
/// 2. A group of layers can share parameters. For instance, if the number of
///    layers (`n_hidden_layers`) is 12 and the number of groups (`n_hidden_groups`)
///    is 4, then layers 0..3, 3..6, 6..9, and 9..12 will share parameters.
#[derive(Debug)]
pub struct AlbertEncoderConfig {
    embeddings: Box<dyn BuildEmbeddings>,
    layer_group: Box<dyn BuildEncoderLayer>,
    n_hidden_groups: usize,
    n_hidden_layers: usize,
}

impl AlbertEncoderConfig {
    /// Encoder embeddings.
    ///
    /// Default: `TransformerEmbeddingsConfig::default()`
    pub fn embeddings(mut self, embeddings: Box<dyn BuildEmbeddings>) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Number of layers within a group.
    ///
    /// Default: `AlbertLayerGroupConfig::default()`
    pub fn layer_group(mut self, layer_group: Box<dyn BuildEncoderLayer>) -> Self {
        self.layer_group = layer_group;
        self
    }

    /// Number of hidden groups.
    ///
    /// Default: `1`
    pub fn n_hidden_groups(mut self, n_hidden_groups: usize) -> Self {
        self.n_hidden_groups = n_hidden_groups;
        self
    }

    /// Number of hidden layers.
    ///
    /// Default: `12`
    pub fn n_hidden_layers(mut self, n_hidden_layers: usize) -> Self {
        self.n_hidden_layers = n_hidden_layers;
        self
    }
}

impl Default for AlbertEncoderConfig {
    fn default() -> Self {
        Self {
            embeddings: Box::<TransformerEmbeddingsConfig>::default(),
            layer_group: Box::<AlbertLayerGroupConfig>::default(),
            n_hidden_groups: 1,
            n_hidden_layers: 12,
        }
    }
}

impl BuildArchitecture for AlbertEncoderConfig {
    type Architecture = AlbertEncoder;

    fn build(&self, vb: candle_nn::VarBuilder) -> Result<Self::Architecture, BoxedError> {
        let embeddings = self
            .embeddings
            .build(vb.push_prefix("embeddings"))
            .context(BuildEmbeddingsSnafu)?;

        let groups = (0..self.n_hidden_groups)
            .map(|n| {
                self.layer_group
                    .build_encoder_layer(vb.push_prefix(format!("group_{n}")))
                    .context(BuildLayerGroupSnafu { n })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(AlbertEncoder {
            embeddings,
            groups,
            n_hidden_layers: self.n_hidden_layers,
        })
    }
}

/// ALBERT encoder errors.
#[derive(Debug, Snafu)]
enum AlbertEncoderError {
    #[snafu(display("Cannot build layer group {n}"))]
    BuildLayerGroup { source: BoxedError, n: usize },

    #[snafu(display("Cannot build embeddings"))]
    BuildEmbeddings { source: BoxedError },

    #[snafu(display("Cannot apply embeddings"))]
    Embeddings { source: BoxedError },

    #[snafu(display("Number of hidden layers ({n_hidden_layers}) not divisable by number of hidden groups ({n_hidden_groups})"))]
    IncorrectNHiddenGroups {
        n_hidden_groups: usize,
        n_hidden_layers: usize,
    },

    #[snafu(display("Cannot apply transformer layer {n}"))]
    TransformerLayer { source: BoxedError, n: usize },
}

/// ALBERT encoder (Lan et al., 2019).
///
/// See [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942).
pub struct AlbertEncoder {
    embeddings: Box<dyn Embeddings>,
    groups: Vec<Box<dyn EncoderLayer>>,
    n_hidden_layers: usize,
}

impl Encoder for AlbertEncoder {
    fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: &AttentionMask,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<EncoderOutput, BoxedError> {
        let embeddings = self
            .embeddings
            .forward(input, train, positions, type_ids)
            .context(EmbeddingsSnafu)?;

        let mut layer_output = embeddings;
        let mut layer_outputs = Vec::with_capacity(self.n_hidden_layers + 1);
        layer_outputs.push(layer_output.clone());

        let layers_per_group = self.n_hidden_layers / self.groups.len();

        for (group_id, group) in self.groups.iter().enumerate() {
            for layer_in_group in 0..layers_per_group {
                layer_output = group
                    .forward_t(&layer_output, attention_mask, positions, train)
                    .context(TransformerLayerSnafu {
                        n: group_id * layers_per_group + layer_in_group,
                    })?;
                layer_outputs.push(layer_output.clone());
            }
        }

        Ok(EncoderOutput::new(layer_outputs))
    }
}

impl FromHF for AlbertEncoder {
    type Config = AlbertEncoderConfig;

    type HFConfig = HFAlbertEncoderConfig;

    type Model = AlbertEncoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("encoder.") {
                name.replace("encoder.", "albert.")
            } else if !name.starts_with("predictions") {
                format!("albert.{name}")
            } else {
                name.to_string()
            };

            // Embeddings
            name = name.replace("piece_embeddings", "word_embeddings");
            name = name.replace("type_embeddings", "token_type_embeddings");
            name = name.replace("embedding_layer_norm", "LayerNorm");
            name = name.replace(
                "embeddings.projection.",
                "encoder.embedding_hidden_mapping_in.",
            );

            // Layers
            static GROUP_RE: OnceLock<Regex> = OnceLock::new();
            let layer_re =
                GROUP_RE.get_or_init(|| Regex::new(r"group_(\d+)").expect("Invalid regex"));
            name = layer_re
                .replace(&name, "encoder.albert_layer_groups.$1")
                .to_string();
            static GROUP_LAYER_RE: OnceLock<Regex> = OnceLock::new();
            let group_layer_re = GROUP_LAYER_RE
                .get_or_init(|| Regex::new(r"group_layer_(\d+)").expect("Invalid regex"));
            name = group_layer_re
                .replace(&name, "albert_layers.$1")
                .to_string();

            // Attention layer.
            name = name.replace("attention.output", "attention.dense");
            name = name.replace("attn_residual_layer_norm", "attention.LayerNorm");

            // Feed-forward layer.
            name = name.replace("ffn.intermediate", "ffn");
            name = name.replace("ffn.output", "ffn_output");
            name = name.replace("ffn_residual_layer_norm", "full_layer_layer_norm");

            name
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, ResultExt, Whatever};

    use crate::models::util::tests::check_encoder;

    use super::AlbertEncoder;

    #[test]
    #[report]
    fn albert_encoder_emits_correct_output() -> Result<(), Whatever> {
        check_encoder::<AlbertEncoder, _>(
            "explosion-testing/albert-test",
            None,
            array![
                [0.4989, -0.3332, 3.2000, -3.6963, 0.0619, 0.1232, 2.3507, -2.1934],
                [-3.3217, 2.9269, 3.4843, -0.7933, -3.8832, -0.7925, 1.8436, -0.9704],
                [0.5875, 0.8119, 6.6794, 0.0263, -2.5903, 0.1582, 4.9209, 3.9640]
            ],
        )
        .whatever_context("Cannot check encoder")
    }
}
