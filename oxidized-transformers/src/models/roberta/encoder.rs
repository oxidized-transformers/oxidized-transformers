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
use crate::models::roberta::embeddings::RobertaEmbeddingsConfig;
use crate::models::transformer::{TransformerEncoder, TransformerEncoderConfig};

/// RoBERTa encoder (Liu et al., 2019).
///
/// See [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
pub struct RobertaEncoder;

/// HF BERT model types
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HfModelType {
    Roberta,
    XlmRoberta,
}

// HF RoBERTa encoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFRobertaEncoderConfig {
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

impl TryFrom<HFRobertaEncoderConfig> for TransformerEncoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFRobertaEncoderConfig) -> Result<Self, Self::Error> {
        let attention_probs_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));
        let hidden_dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps)
                .size(hf_config.hidden_size),
        );

        let embeddings = RobertaEmbeddingsConfig::default()
            .padding_id(hf_config.pad_token_id)
            .transformer_embeddings(Box::new(
                TransformerEmbeddingsConfig::default()
                    .embedding_dropout(hidden_dropout.clone())
                    .embedding_layer_norm(layer_norm.clone())
                    .embedding_width(hf_config.hidden_size)
                    .hidden_width(hf_config.hidden_size)
                    .n_pieces(hf_config.vocab_size)
                    .n_positions(Some(
                        hf_config.max_position_embeddings - hf_config.pad_token_id as usize + 1,
                    ))
                    .n_types(Some(hf_config.type_vocab_size)),
            ));

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

impl FromHF for RobertaEncoder {
    type Config = TransformerEncoderConfig;

    type HFConfig = HFRobertaEncoderConfig;

    type Model = TransformerEncoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("encoder.") {
                name.replace("encoder.", "roberta.")
            } else if !name.starts_with("lm_head") {
                format!("roberta.{name}")
            } else {
                name.to_string()
            };

            // Embeddings
            name = name.replace("embeddings.piece_embeddings", "embeddings.word_embeddings");
            name = name.replace(
                "embeddings.type_embeddings",
                "embeddings.token_type_embeddings",
            );
            name = name.replace("embeddings.embedding_layer_norm", "embeddings.LayerNorm");

            // Layers
            name = name.replace("roberta.layer", "roberta.encoder.layer");
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

    use crate::models::roberta::encoder::RobertaEncoder;
    use crate::models::util::tests::check_encoder;

    #[test]
    #[report]
    fn roberta_encoder_emits_correct_output() -> Result<(), Whatever> {
        check_encoder::<RobertaEncoder, _>(
            "explosion-testing/roberta-test",
            None,
            array![
                [0.5322, 6.6674, 7.6708, 2.6427, 0.7608, 6.1489, -0.8932, 7.3365],
                [-0.3528, -0.2615, 5.3301, 1.5377, 0.6676, 1.2909, 7.5252, 3.0232],
                [2.0244, 4.2406, 4.4161, 2.6682, 3.4550, -2.3752, 6.7672, 1.8406]
            ],
        )
        .whatever_context("Cannot check encoder")
    }
}
