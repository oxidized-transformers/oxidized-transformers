use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{AttentionHeads, QkvMode, SDPAConfig, SelfAttentionConfig};
use crate::layers::dropout::DropoutConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf::FromHF;
use crate::models::transformer::{TransformerEncoder, TransformerEncoderConfig};

/// BERT encoder (Devlin et al., 2018).
///
/// See [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
pub struct BertEncoder;

/// HF BERT model types
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfModelType {
    Bert,
}

/// HF BERT encoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFBertEncoderConfig {
    attention_probs_dropout_prob: f32,
    hidden_act: Activation,
    hidden_dropout_prob: f32,
    hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: HfModelType,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    pad_token_id: u32,
    type_vocab_size: usize,
    vocab_size: usize,
}

impl TryFrom<HFBertEncoderConfig> for TransformerEncoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFBertEncoderConfig) -> Result<Self, Self::Error> {
        let attention_probs_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));
        let hidden_dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_dropout(hidden_dropout.clone())
            .embedding_layer_norm(layer_norm.clone())
            .embedding_width(hf_config.hidden_size)
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

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .attn_residual_layer_norm(layer_norm.clone())
            .feedforward(feedforward)
            .ffn_residual_layer_norm(layer_norm);

        Ok(Self::default()
            .embeddings(Box::new(embeddings))
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers))
    }
}

impl FromHF for BertEncoder {
    type Config = TransformerEncoderConfig;

    type HFConfig = HFBertEncoderConfig;

    type Model = TransformerEncoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("encoder.") {
                name.replace("encoder.", "bert.")
            } else if !name.starts_with("cls") {
                format!("bert.{name}")
            } else {
                name.to_string()
            };

            // Embeddings
            name = name.replace("piece_embeddings", "word_embeddings");
            name = name.replace("type_embeddings", "token_type_embeddings");
            name = name.replace("embedding_layer_norm", "LayerNorm");

            // Layers
            name = name.replace("bert.layer", "bert.encoder.layer");
            static LAYER_RE: OnceLock<Regex> = OnceLock::new();
            let layer_re =
                LAYER_RE.get_or_init(|| Regex::new(r"layer_(\d+)").expect("Invalid regex"));
            name = layer_re.replace(&name, "layer.$1").to_string();

            // Attention layer.
            name = name.replace("attention.output", "attention.output.dense");
            name = name.replace("query", "self.query");
            name = name.replace("key", "self.key");
            name = name.replace("value", "self.value");
            name = name.replace("attn_residual_layer_norm", "attention.output.LayerNorm");

            // Feed-forward layer.
            name = name.replace("ffn.intermediate", "intermediate.dense");
            name = name.replace("ffn.output", "output.dense");
            name = name.replace("ffn_residual_layer_norm", "output.LayerNorm");

            name
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, ResultExt, Whatever};

    use crate::models::bert::BertEncoder;
    use crate::models::util::tests::check_encoder;

    #[test]
    #[report]
    fn bert_encoder_emits_correct_output() -> Result<(), Whatever> {
        check_encoder::<BertEncoder, _>(
            "explosion-testing/bert-test",
            None,
            array![
                [6.6632, 4.4528, 8.7430, 3.5811, 7.8127, 3.0751, 1.0560, 2.6171],
                [5.9245, 4.3979, 5.7063, 5.2123, 5.0856, 2.5916, 0.5269, 5.9339],
                [7.1707, 6.5958, 3.5610, 3.3650, 6.1192, 6.3981, 2.1112, 7.8207]
            ],
        )
        .whatever_context("Cannot check encoder")
    }
}
