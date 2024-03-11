use serde::{Deserialize, Serialize};

use crate::error::BoxedError;
use crate::models::hf::FromHF;
use crate::models::roberta::{HFRobertaEncoderConfig, RobertaEncoder};
use crate::models::transformer::{TransformerEncoder, TransformerEncoderConfig};

/// XLM-RoBERTa encoder (Conneau et al., 2019).
///
/// This is a simple convenience wrapper around
/// [`RobertaEncoder`](crate::models::roberta::RobertaEncoder),
/// since RoBERTa and XLM-RoBERTa share the same architecture, only their tokenizers differ.
///
/// See [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116).
pub struct XLMREncoder;

/// HF XLM-RoBERTa encoder configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HFXLMREncoderConfig {
    #[serde(flatten)]
    roberta: HFRobertaEncoderConfig,
}

impl TryFrom<HFXLMREncoderConfig> for TransformerEncoderConfig {
    type Error = BoxedError;

    fn try_from(hf_config: HFXLMREncoderConfig) -> Result<Self, Self::Error> {
        TransformerEncoderConfig::try_from(hf_config.roberta)
    }
}

impl FromHF for XLMREncoder {
    type Config = TransformerEncoderConfig;

    type HFConfig = HFXLMREncoderConfig;

    type Model = TransformerEncoder;

    fn rename_parameters() -> impl Fn(&str) -> String {
        RobertaEncoder::rename_parameters()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use snafu::{report, ResultExt, Whatever};

    use crate::models::util::tests::check_encoder;
    use crate::models::xlm_roberta::encoder::XLMREncoder;

    #[test]
    #[report]
    fn xlm_roberta_encoder_emits_correct_output() -> Result<(), Whatever> {
        check_encoder::<XLMREncoder, _>(
            "explosion-testing/xlm-roberta-test",
            None,
            array![
                [-0.1922, 1.1965, -2.1958, 5.0741, 0.8208, 1.2687, -1.3562, -3.4613],
                [-4.4533, -4.1397, 2.2614, 0.0210, 1.7515, 2.7256, -1.7625, 3.2116],
                [-1.7351, 2.3118, 5.6222, -1.0945, -0.5056, 0.6371, 2.2917, 2.0503]
            ],
        )
        .whatever_context("Cannot check encoder")
    }
}
