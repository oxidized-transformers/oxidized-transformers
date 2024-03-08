use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::layers::activation::Activation;
use crate::layers::attention::{
    AttentionHeads, QkvMode, ScaledDotProductAttentionConfig, SelfAttentionConfig,
};
use crate::layers::dropout::DropoutConfig;
use crate::layers::embeddings::QueryKeyRotaryEmbeddingsConfig;
use crate::layers::feedforward::PointwiseFeedForwardConfig;
use crate::layers::layer_norm::LayerNormConfig;
use crate::layers::transformer::{TransformerEmbeddingsConfig, TransformerLayerConfig};
use crate::models::hf::FromHF;
use crate::models::transformer::{TransformerDecoder, TransformerDecoderConfig};

/// GPT-NeoX decoder (Black et al., 2022).
///
/// See: [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)
pub struct GPTNeoXDecoder;

/// HF GPT-NeoX model types.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HfModelType {
    GptNeox,
}

/// HF GPT-NeoX decoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFGPTNeoXDecoderConfig {
    attention_probs_dropout_prob: f32,
    hidden_act: Activation,
    hidden_dropout_prob: f32,
    pub(crate) hidden_size: usize,
    initializer_range: f32,
    intermediate_size: usize,
    layer_norm_eps: f32,
    max_position_embeddings: usize,
    model_type: HfModelType,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    rotary_emb_base: usize,
    rotary_pct: f32,
    tie_word_embeddings: bool,
    type_vocab_size: usize,
    use_parallel_residual: bool,
    pub(crate) vocab_size: usize,
}

impl TryFrom<HFGPTNeoXDecoderConfig> for TransformerDecoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFGPTNeoXDecoderConfig) -> Result<Self, Self::Error> {
        let attention_dropout =
            Box::new(DropoutConfig::default().p(hf_config.attention_probs_dropout_prob));

        let layer_norm = Box::new(
            LayerNormConfig::default()
                .eps(hf_config.layer_norm_eps as f64)
                .size(hf_config.hidden_size),
        );

        let embeddings = TransformerEmbeddingsConfig::default()
            .embedding_width(hf_config.hidden_size)
            .hidden_width(hf_config.hidden_size)
            .n_pieces(hf_config.vocab_size);

        let dropout = Box::new(DropoutConfig::default().p(hf_config.hidden_dropout_prob));
        let feedforward = PointwiseFeedForwardConfig::default()
            .activation(Box::new(hf_config.hidden_act))
            .dropout(dropout.clone())
            .hidden_width(hf_config.hidden_size)
            .intermediate_width(hf_config.intermediate_size)
            .layer_norm(layer_norm.clone());

        let attention = SelfAttentionConfig::default()
            .attention_heads(AttentionHeads {
                n_query_heads: hf_config.num_attention_heads,
                n_key_value_heads: hf_config.num_attention_heads,
                qkv_mode: QkvMode::MergedSplitBefore,
            })
            .attention_scorer(Box::new(
                ScaledDotProductAttentionConfig::default().dropout(attention_dropout),
            ))
            .dropout(dropout)
            .hidden_width(hf_config.hidden_size)
            .layer_norm(layer_norm.clone())
            .rotary_embeddings(Some(
                QueryKeyRotaryEmbeddingsConfig::default()
                    .base(hf_config.rotary_emb_base)
                    .fraction(hf_config.rotary_pct)
                    .head_width(hf_config.hidden_size / hf_config.num_attention_heads)
                    .seq_len(hf_config.max_position_embeddings),
            ));

        let layer = TransformerLayerConfig::default()
            .attention(attention)
            .feedforward(feedforward)
            .use_parallel_attention(hf_config.use_parallel_residual);

        Ok(TransformerDecoderConfig::default()
            .embeddings(embeddings)
            .layer(Box::new(layer))
            .n_hidden_layers(hf_config.num_hidden_layers)
            .output_layer_norm(layer_norm))
    }
}

impl FromHF for GPTNeoXDecoder {
    type Config = TransformerDecoderConfig;

    type HFConfig = HFGPTNeoXDecoderConfig;

    type Model = TransformerDecoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        |name| {
            let mut name = if name.starts_with("decoder.") {
                name.replace("decoder.", "gpt_neox.")
            } else if !name.starts_with("output_embeddings") {
                format!("gpt_neox.{name}")
            } else {
                name.to_string()
            };

            // Embedding layer.
            name = name.replace("embeddings.piece_embeddings", "embed_in");

            // Attention layer.
            name = name.replace("attention.layer_norm", "input_layernorm");
            name = name.replace(".attention.output", ".attention.dense");
            name = name.replace("qkv", "query_key_value");

            // Feed-forward layer.
            name = name.replace(".ffn.layer_norm", ".post_attention_layernorm");
            name = name.replace(".intermediate", ".dense_h_to_4h");
            name = name.replace(".ffn.output", ".ffn.dense_4h_to_h");
            name = name.replace(".ffn", ".mlp");

            // Layer norm after all layers.
            name = name.replace("output_layer_norm", "final_layer_norm");

            // Output embeddings.
            name = name.replace("output_embeddings", "embed_out");

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
    use candle_core::Device;
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::{Decoder, LayerOutputs};
    use crate::kv_cache::KeyValueCache;
    use crate::layers::attention::AttentionMask;
    use crate::models::gpt_neox::GPTNeoXDecoder;
    use crate::models::hf::FromHFHub;
    use crate::util::tests::{assert_tensor_eq, sample_transformer_inputs, PseudoRandomReduction};

    #[test]
    #[report]
    fn gpt_neox_decoder_gives_correct_output() -> Result<(), Whatever> {
        let decoder = GPTNeoXDecoder::from_hf_hub(
            "trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors-sharded",
            None,
            Device::Cpu,
        )
        .with_whatever_context(|_| "Cannot load model")?;

        let (input, mask) = sample_transformer_inputs()?;

        let output = decoder
            .forward_t(&input, &mask, &mut KeyValueCache::no_cache(), None, false)
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let last_output = output.layer_outputs().last().unwrap();

        assert_tensor_eq::<f32>(
            last_output
                .pseudo_random_reduction()
                .whatever_context("Cannot apply reduction using random vector")?,
            array![
                [2.8711, 2.2852, 2.6235, 3.7102, 1.3372, 2.9834, 2.7712, 5.1699],
                [1.0860, 5.2414, 1.7125, 1.5052, 0.8727, 3.4021, 5.8198, -0.8003],
                [-3.6789, -1.5767, -4.2494, 0.3412, -4.3807, -3.3196, -3.2535, 0.5096]
            ],
            1e-4,
        );

        Ok(())
    }

    #[test]
    #[report]
    fn gpt_neox_decoder_gives_correct_output_with_cache() -> Result<(), Whatever> {
        let decoder = GPTNeoXDecoder::from_hf_hub(
            "trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors-sharded",
            None,
            Device::Cpu,
        )
        .with_whatever_context(|_| "Cannot load model")?;

        let (input, mask) = sample_transformer_inputs()?;

        let mut cache = KeyValueCache::cache();
        let attention_mask = AttentionMask::new(
            mask.bool_mask()
                .narrow(1, 0, 7)
                .whatever_context("Cannot slice attention mask")?,
        )
        .whatever_context("Cannot build attention mask")?;

        let _ = decoder
            .forward_t(
                &input
                    .narrow(1, 0, 7)
                    .whatever_context("Cannot slice input")?,
                &attention_mask,
                &mut cache,
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let attention_mask = attention_mask
            .extend(
                &AttentionMask::new(
                    mask.bool_mask()
                        .narrow(1, 7, 1)
                        .whatever_context("Cannot slice attention mask")?,
                )
                .whatever_context("Cannot build attention mask")?,
            )
            .whatever_context("Cannot extend attention mask")?;

        let output = decoder
            .forward_t(
                &input
                    .narrow(1, 7, 1)
                    .whatever_context("Cannot slice input")?,
                &attention_mask,
                &mut cache,
                None,
                false,
            )
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let last_output = output.layer_outputs().last().unwrap();

        assert_tensor_eq::<f32>(
            last_output
                .pseudo_random_reduction()
                .whatever_context("Cannot apply reduction using random vector")?,
            array![[5.1699], [-0.8003], [0.5096]],
            1e-4,
        );

        Ok(())
    }
}
