use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::models::gpt_neox::decoder::HFGPTNeoXDecoderConfig;
use crate::models::gpt_neox::GPTNeoXDecoder;
use crate::models::hf::FromHF;
use crate::models::transformer::{
    TransformerCausalLM, TransformerCausalLMConfig, TransformerDecoderConfig,
};

/// GPT-NeoX causal language model (Black et al., 2022).
///
/// See: [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)
pub struct GPTNeoXCausalLM;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HfGPTNeoXCausalLMConfig {
    #[serde(flatten)]
    decoder: HFGPTNeoXDecoderConfig,
}

impl TryFrom<HfGPTNeoXCausalLMConfig> for TransformerCausalLMConfig {
    type Error = BoxedError;

    fn try_from(config: HfGPTNeoXCausalLMConfig) -> Result<Self, Self::Error> {
        Ok(Self::default()
            .hidden_size(config.decoder.hidden_size)
            // Input and output vocab sizes are the same.
            .n_pieces(config.decoder.vocab_size)
            .decoder(Box::new(TransformerDecoderConfig::try_from(
                config.decoder,
            )?)))
    }
}

impl FromHF for GPTNeoXCausalLM {
    type Config = TransformerCausalLMConfig;

    type HFConfig = HfGPTNeoXCausalLMConfig;

    type Model = TransformerCausalLM;

    fn rename_parameters() -> impl Fn(&str) -> String {
        GPTNeoXDecoder::rename_parameters()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, Whatever};

    use crate::models::gpt_neox::causal_lm::GPTNeoXCausalLM;
    use crate::models::util::tests::check_causal_lm;

    #[test]
    #[report]
    fn gpt_neox_causal_lm_emits_correct_output() -> Result<(), Whatever> {
        check_causal_lm!(
            GPTNeoXCausalLM,
            "trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors-sharded",
            None,
            array![
                [-1.4418, 0.0000, -1.8183, 0.0000, 0.0000, 1.2548, 0.0000, 0.0000],
                [0.8485, 0.0000, 0.0000, 3.2619, 0.0000, 2.0168, -0.2824, -1.9384],
                [0.0000, -0.7587, -3.1774, 0.0000, 0.0000, 3.4785, 3.6013, 0.0000]
            ],
            epsilon = 1e-4,
        )
    }

    #[test]
    #[report]
    fn gpt_neox_causal_lm_emits_correct_output_float16() -> Result<(), Whatever> {
        check_causal_lm!(
            GPTNeoXCausalLM,
            "explosion-testing/gpt-neox-float16",
            None,
            array![
                [-1.4418, 0.0000, -1.8183, 0.0000, 0.0000, 1.2548, 0.0000, 0.0000],
                [0.8485, 0.0000, 0.0000, 3.2619, 0.0000, 2.0168, -0.2824, -1.9384],
                [0.0000, -0.7587, -3.1774, 0.0000, 0.0000, 3.4785, 3.6013, 0.0000]
            ],
            epsilon = 1e-1
        )
    }
}
