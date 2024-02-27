use std::fmt::Debug;

use crate::architectures::BuildArchitecture;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::architectures::output::LayerOutputs;
use crate::error::BoxedError;
use crate::layers::attention::AttentionMask;

/// Decoder output.
pub struct DecoderOutput {
    all_outputs: Vec<Tensor>,
}

impl DecoderOutput {
    pub fn new(all_outputs: Vec<Tensor>) -> Self {
        Self { all_outputs }
    }
}

impl LayerOutputs for DecoderOutput {
    fn layer_outputs(&self) -> &[Tensor] {
        &self.all_outputs
    }

    fn embedding_layer_output(&self) -> Option<&Tensor> {
        self.all_outputs.first()
    }
}

/// Trait for building decoders.
pub trait BuildDecoder: Debug {
    /// Decoder type.
    type Decoder: Decoder;

    /// Build a decoder.
    fn build(&self, vb: VarBuilder) -> Result<Self::Decoder, BoxedError>;
}

impl<D> BuildDecoder for D
where
    D: BuildArchitecture + Debug,
    D::Architecture: Decoder,
{
    type Decoder = D::Architecture;

    fn build(&self, vb: VarBuilder) -> Result<Self::Decoder, BoxedError> {
        self.build(vb)
    }
}

/// Trait for decoders.
pub trait Decoder {
    /// Cache type for the decoder.
    type Cache;

    /// Decode an input sequence.
    ///
    /// Returns the decoder output and cache.
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
    ) -> Result<DecoderOutput, BoxedError>;
}

/// Trait for decoder layers.
pub trait DecoderLayer {
    /// Cache type for the decoder.
    ///
    /// The cache can store the intermediate values of the decoder layer,
    /// avoiding recomputation when calling the decoder again for generating
    /// another output.
    type Cache;

    /// Apply the decoder layer to the given hidden representations.
    ///
    /// * `piece_ids` - Hidden representations to apply the layer to.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///    corresponding mask element is set to `false` are ignored
    ///    during attention calculation.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `cache` - Cache to avoid recomputing intermediate values.
    /// * `positions` - Input positions.
    ///    *Shape:* `(batch_size, seq_len)`
    /// * `train` - Whether to train the layer.
    ///
    /// Returns layer output and the cache.
    /// *Shape:* ``(batch_size, seq_len, width)``
    fn forward_t(
        &self,
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        cache: &mut Self::Cache,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, BoxedError>;
}

/// Trait for building decoder layers.
pub trait BuildDecoderLayer: Debug {
    /// Cache type for the decoder.
    ///
    /// The cache can store the intermediate values of the decoder layer,
    /// avoiding recomputation when calling the decoder again for generating
    /// another output.
    type Cache;

    /// Build a decoder layer.
    fn build_decoder_layer(
        &self,
        vb: VarBuilder,
    ) -> Result<Box<dyn DecoderLayer<Cache = Self::Cache>>, BoxedError>;
}
