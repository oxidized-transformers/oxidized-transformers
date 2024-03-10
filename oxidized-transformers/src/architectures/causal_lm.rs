use std::fmt::Debug;

use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::architectures::{BuildArchitecture, DecoderOutput, LayerOutputs};
use crate::error::BoxedError;
use crate::layers::attention::AttentionMask;

/// Causal language model output.
pub struct CausalLMOutput {
    decoder_output: DecoderOutput,
    logits: Tensor,
}

impl CausalLMOutput {
    /// Create a causal language model output.
    pub fn new(decoder_output: DecoderOutput, logits: Tensor) -> Self {
        Self {
            decoder_output,
            logits,
        }
    }

    /// Get the output of the decoder used by the causal language model.
    pub fn decoder_output(&self) -> &DecoderOutput {
        &self.decoder_output
    }

    /// Get the logits of the next predicted token.
    ///
    /// The logits are the unnormalized probabilities. Applying softmax to the
    /// logits will give the probability distribution of the next token over the
    /// vocabulary.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

impl LayerOutputs for CausalLMOutput {
    fn layer_outputs(&self) -> &[Tensor] {
        self.decoder_output.layer_outputs()
    }

    fn embedding_layer_output(&self) -> Option<&Tensor> {
        self.decoder_output.embedding_layer_output()
    }
}

/// Trait for building causal language models.
pub trait BuildCausalLM: Debug {
    type CausalLM: CausalLM;

    /// Build a causal language model.
    fn build(&self, vb: VarBuilder) -> Result<Self::CausalLM, BoxedError>;
}

impl<C> BuildCausalLM for C
where
    C: BuildArchitecture + Debug,
    C::Architecture: CausalLM,
{
    type CausalLM = C::Architecture;

    fn build(&self, vb: VarBuilder) -> Result<Self::CausalLM, BoxedError> {
        self.build(vb)
    }
}

/// Trait for causal language models.
pub trait CausalLM {
    /// Cache type for the causal language model.
    type Cache;

    /// Predict the next token using a causal language model.
    ///
    /// Returns the piece representations, cache, and logits of the next
    /// predicted token.
    ///
    /// * `piece_ids` - Input sequence.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///   corresponding mask element is set to `false` are ignored during
    ///   attention calculation.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `cache` - Cache to avoid recomputing intermediate values.
    /// * `positions` - Input positions.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `train` - Whether to train the layer.
    fn forward_t(
        &self,
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<CausalLMOutput, BoxedError>;
}
