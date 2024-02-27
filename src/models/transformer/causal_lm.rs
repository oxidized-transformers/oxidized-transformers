use candle_core::{ModuleT, Tensor};
use candle_nn::{linear_no_bias, VarBuilder};
use snafu::{ResultExt, Snafu};

use crate::architectures::BuildArchitecture;
use crate::architectures::{BuildDecoder, CausalLM, CausalLMOutput, Decoder, LayerOutputs};
use crate::error::BoxedError;
use crate::kv_cache::KeyValueCache;
use crate::layers::attention::AttentionMask;
use crate::models::transformer::{TransformerDecoder, TransformerDecoderConfig};

/// Transformer causal language model configuration.
#[derive(Debug)]
pub struct TransformerCausalLMConfig {
    decoder: Box<dyn BuildDecoder<Decoder = TransformerDecoder>>,
    hidden_size: usize,
    n_pieces: usize,
}

impl TransformerCausalLMConfig {
    /// Decoder.
    ///
    /// Default: `TransformerDecoderConfig`.
    pub fn decoder(mut self, decoder: Box<dyn BuildDecoder<Decoder = TransformerDecoder>>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Hidden size.
    ///
    /// Default: 4096.
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Number of pieces in the output vocabulary.
    ///
    /// Default: 32000.
    pub fn n_pieces(mut self, n_pieces: usize) -> Self {
        self.n_pieces = n_pieces;
        self
    }
}

impl Default for TransformerCausalLMConfig {
    fn default() -> Self {
        Self {
            decoder: Box::<TransformerDecoderConfig>::default(),
            hidden_size: 4096,
            n_pieces: 32000,
        }
    }
}

impl BuildArchitecture for TransformerCausalLMConfig {
    type Architecture = TransformerCausalLM;

    fn build(&self, vb: VarBuilder) -> Result<Self::Architecture, BoxedError> {
        let decoder = Box::new(
            self.decoder
                .build(vb.push_prefix("decoder"))
                .context(BuildDecoderSnafu)?,
        );

        Ok(TransformerCausalLM {
            decoder,
            output_embeddings: Box::new(
                linear_no_bias(
                    self.hidden_size,
                    self.n_pieces,
                    vb.push_prefix("output_embeddings"),
                )
                .context(BuildEmbeddingsSnafu)?,
            ),
        })
    }
}

/// `TransformerCausalLM` errors.
#[derive(Debug, Snafu)]
pub enum TransformerCausalLMError {
    #[snafu(display("Cannot build decoder"))]
    BuildDecoder { source: BoxedError },

    #[snafu(display("Cannot build output embeddings"))]
    BuildEmbeddings { source: candle_core::Error },

    #[snafu(display("Nothing to decode, the decoder does not have any layer outputs"))]
    NoLayers,

    #[snafu(display("Cannot compute logits for the output vocabulary"))]
    ComputeLogits { source: candle_core::Error },
}

/// Transformer-based causal language model.
pub struct TransformerCausalLM {
    decoder: Box<dyn Decoder<Cache = KeyValueCache>>,
    output_embeddings: Box<dyn ModuleT>,
}

impl CausalLM for TransformerCausalLM {
    type Cache = KeyValueCache;

    fn forward_t(
        &self,
        piece_ids: &Tensor,
        mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<CausalLMOutput, BoxedError> {
        let decoder_output = self
            .decoder
            .forward_t(piece_ids, mask, cache, positions, train)?;
        let last_layer = decoder_output
            .layer_outputs()
            .last()
            .ok_or(TransformerCausalLMError::NoLayers)?;
        let logits = self
            .output_embeddings
            .forward_t(last_layer, train)
            .context(ComputeLogitsSnafu)?;

        Ok(CausalLMOutput::new(decoder_output, logits))
    }
}
