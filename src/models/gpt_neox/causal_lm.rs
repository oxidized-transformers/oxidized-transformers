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
    use candle_core::Device;
    use ndarray::array;
    use snafu::{report, FromString, ResultExt, Whatever};

    use crate::architectures::CausalLM;
    use crate::kv_cache::KeyValueCache;
    use crate::models::gpt_neox::causal_lm::GPTNeoXCausalLM;
    use crate::models::hf::FromHFHub;
    use crate::util::tests::{assert_tensor_eq, sample_transformer_inputs};

    #[test]
    #[report]
    fn gptneox_causal_lm_emits_correct_output() -> Result<(), Whatever> {
        let causal_lm = GPTNeoXCausalLM::from_hf_hub(
            "trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors-sharded",
            None,
            Device::Cpu,
        )
        .whatever_context("Cannot load model")?;

        let (input, mask) = sample_transformer_inputs()?;

        let output = causal_lm
            .forward_t(&input, &mask, &mut KeyValueCache::no_cache(), None, false)
            .map_err(|e| Whatever::with_source(e, "Cannot decode input".to_string()))?;

        let logits = output.logits();
        let logits_max = logits.max(2).whatever_context("Cannot find max logits")?;

        assert_tensor_eq::<f32>(
            logits_max,
            array![
                [0.3865, 0.4071, 0.3704, 0.3072, 0.3520, 0.3358, 0.3111, 0.3711],
                [0.3771, 0.3409, 0.3430, 0.3939, 0.3990, 0.3734, 0.3794, 0.3643],
                [0.3730, 0.3944, 0.3295, 0.3375, 0.3606, 0.4349, 0.3595, 0.3239]
            ],
            1e-4,
        );

        Ok(())
    }
}
