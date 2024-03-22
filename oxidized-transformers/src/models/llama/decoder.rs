use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{AttentionHeads, QkvMode, SelfAttentionConfig};
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::RMSNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf::FromHF;
use crate::models::transformer::{TransformerDecoder, TransformerDecoderConfig};

/// Llama decoder (Touvron et al., 2023).
///
/// See:
/// * [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
/// * [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
pub struct LlamaDecoder;

/// HF Llama model types
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfModelType {
    Llama,
}

/// Hf Llama decoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFLlamaDecoderConfig {
    hidden_act: Activation,
    pub(crate) hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: HfModelType,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    pub(crate) vocab_size: usize,
}

impl TryFrom<HFLlamaDecoderConfig> for TransformerDecoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFLlamaDecoderConfig) -> Result<Self, Self::Error> {
        let layer_norm = Box::new(
            RMSNormConfig::default()
                .eps(hf_config.rms_norm_eps as f64)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_width(hf_config.hidden_size)
            .hidden_width(hf_config.hidden_size)
            .n_pieces(hf_config.vocab_size);

        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size)
            .layer_norm(layer_norm.clone())
            .use_bias(false)
            .use_gate(true);

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_key_value_heads,
                qkv_mode: QkvMode::Separate,
            })
            .hidden_width(hf_config.hidden_size)
            .layer_norm(layer_norm.clone())
            .rotary_embeddings(Some(
                QueryKeyRotaryEmbeddingsConfig::default()
                    .head_width(hf_config.hidden_size / hf_config.num_attention_heads)
                    .seq_len(hf_config.max_position_embeddings),
            ));

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .feedforward(feedforward);

        Ok(TransformerDecoderConfig::default()
            .embeddings(embeddings)
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers)
            .output_layer_norm(layer_norm))
    }
}

impl FromHF for LlamaDecoder {
    type Config = TransformerDecoderConfig;

    type HFConfig = HFLlamaDecoderConfig;

    type Model = TransformerDecoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("decoder.") {
                name.replace("decoder.", "model.")
            } else if !name.starts_with("output_embeddings") {
                format!("model.{name}")
            } else {
                name.to_string()
            };
            name = name.replace("embeddings.piece_embeddings", "embed_tokens");

            // Attention layer.
            name = name.replace("attention.query", "attention.q_proj");
            name = name.replace("attention.key", "attention.k_proj");
            name = name.replace("attention.value", "attention.v_proj");
            name = name.replace("attention.output", "attention.o_proj");
            name = name.replace("attention.layer_norm", "input_layernorm");
            name = name.replace("attention.", "self_attn.");

            // Feed-forward layer.
            name = name.replace("ffn.layer_norm", "post_attention_layernorm");
            name = name.replace("ffn.output", "ffn.down_proj");
            name = name.replace("ffn.", "mlp.");
            name = name.replace("intermediate", "up_proj");
            name = name.replace("gate", "gate_proj");

            // Layer norm after all layers.
            name = name.replace("output_layer_norm", "norm");

            // Output vocab.
            name = name.replace("output_embeddings", "lm_head");

            static LAYER_RE: OnceLock<Regex> = OnceLock::new();
            let layer_re =
                LAYER_RE.get_or_init(|| Regex::new(r"layer_(\d+)").expect("Invalid regex"));
            name = layer_re.replace(&name, "layers.$1").to_string();
            name
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, ResultExt, Whatever};

    use crate::models::llama::LlamaDecoder;
    use crate::models::util::tests::{check_decoder, check_decoder_with_cache};

    #[report]
    fn llama_decoder_gives_correct_output() -> Result<(), Whatever> {
        check_decoder::<LlamaDecoder, _>(
            "explosion-testing/llama2-kv-sharing",
            None,
            array![
                [0.0000, -0.7430, -5.4662, -6.5113, -7.6470, -12.3254, -7.7909, -7.3655],
                [-9.9933, -10.4256, -10.3813, -12.0933, -12.3758, -17.6279, -17.4024, -11.2087],
                [-1.7355, 1.8150, 2.2124, 1.4387, 1.2247, 1.7853, -0.4188, -1.9727]
            ],
        )
        .whatever_context("Cannot check decoder")
    }

    #[test]
    #[report]
    fn llama_decoder_give_correct_output_with_cache() -> Result<(), Whatever> {
        check_decoder_with_cache!(
            LlamaDecoder,
            "explosion-testing/llama2-kv-sharing",
            None,
            array![[0.0], [-11.2087], [0.0]],
            epsilon = 1e-4,
        )
    }

    #[test]
    #[report]
    fn llama_decoder_give_correct_output_with_cache_float16() -> Result<(), Whatever> {
        check_decoder_with_cache!(
            LlamaDecoder,
            "explosion-testing/llama2-kv-sharing-float16",
            None,
            array![[0.0], [-11.223126], [0.0]],
            epsilon = 1e-1,
        )
    }
}
