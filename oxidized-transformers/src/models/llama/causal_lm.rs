use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::models::hf::FromHF;
use crate::models::llama::decoder::HFLlamaDecoderConfig;
use crate::models::llama::LlamaDecoder;
use crate::models::transformer::{
    TransformerCausalLM, TransformerCausalLMConfig, TransformerDecoderConfig,
};

/// Llama causal language model (Touvron et al., 2023).
///
/// See:
/// * [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
/// * [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
pub struct LlamaCausalLM;

/// Llama causal language model configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HfLlamaCausalLMConfig {
    #[serde(flatten)]
    decoder: HFLlamaDecoderConfig,
}

impl TryFrom<HfLlamaCausalLMConfig> for TransformerCausalLMConfig {
    type Error = BoxedError;

    fn try_from(config: HfLlamaCausalLMConfig) -> Result<Self, Self::Error> {
        Ok(Self::default()
            .hidden_size(config.decoder.hidden_size)
            // Input and output vocab sizes are the same.
            .n_pieces(config.decoder.vocab_size)
            .decoder(Box::new(TransformerDecoderConfig::try_from(
                config.decoder,
            )?)))
    }
}

impl FromHF for LlamaCausalLM {
    type Config = TransformerCausalLMConfig;

    type HFConfig = HfLlamaCausalLMConfig;

    type Model = TransformerCausalLM;

    fn rename_parameters() -> impl Fn(&str) -> String {
        LlamaDecoder::rename_parameters()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, Whatever};

    use crate::models::llama::causal_lm::LlamaCausalLM;
    use crate::models::util::tests::check_causal_lm;

    #[test]
    #[report]
    fn llama_causal_lm_emits_correct_output() -> Result<(), Whatever> {
        check_causal_lm::<LlamaCausalLM, _>(
            "explosion-testing/llama2-kv-sharing",
            None,
            array![
                [0.0000, -0.7422, 3.9272, 2.4643, 1.2032, -0.2746, 0.0612, 2.6404],
                [-1.6657, -1.5350, -0.9877, 0.1828, 0.2311, 0.7174, 0.4477, -0.4943],
                [-1.4341, -2.5877, -1.4347, -1.1339, -1.8117, -0.2561, -0.6859, -2.5824]
            ],
        )
    }
}
