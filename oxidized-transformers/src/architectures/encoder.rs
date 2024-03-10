use std::fmt::Debug;

use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::architectures::output::LayerOutputs;
use crate::architectures::BuildArchitecture;
use crate::error::BoxedError;
use crate::layers::attention::AttentionMask;

/// Encoder output.
pub struct EncoderOutput {
    all_outputs: Vec<Tensor>,
}

impl EncoderOutput {
    /// Create an encoder output.
    pub fn new(all_outputs: Vec<Tensor>) -> Self {
        Self { all_outputs }
    }
}

impl LayerOutputs for EncoderOutput {
    /// All layer outputs.
    fn layer_outputs(&self) -> &[Tensor] {
        &self.all_outputs
    }

    /// Embedding layer output.
    fn embedding_layer_output(&self) -> Option<&Tensor> {
        self.all_outputs.first()
    }
}

/// Trait for encoders.
pub trait Encoder {
    /// Encode an input sequence.
    ///
    /// Returns the encoder output.
    ///
    /// * `piece_ids` - Input sequence.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///   corresponding mask element is set to `false` are ignored during
    ///   attention calculation.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `positions` - Input positions.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `type_ids` - Input type ids.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `train` - Whether to train the layer.
    fn forward_t(
        &self,
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        positions: Option<&Tensor>,
        type_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<EncoderOutput, BoxedError>;
}

/// Trait for building encoders.
pub trait BuildEncoder: Debug {
    /// Encoder type.
    type Encoder: Encoder;

    /// Build an encoder.
    fn build(&self, vb: VarBuilder) -> Result<Self::Encoder, BoxedError>;
}

impl<D> BuildEncoder for D
where
    D: BuildArchitecture + Debug,
    D::Architecture: Encoder,
{
    type Encoder = D::Architecture;

    fn build(&self, vb: VarBuilder) -> Result<Self::Encoder, BoxedError> {
        self.build(vb)
    }
}

/// Trait for encoder layers.
pub trait EncoderLayer {
    /// Apply the encoder layer to the given hidden representations.
    ///
    /// * `input` - Hidden representations to apply the layer to.
    ///   *Shape:* `(batch_size, seq_len, width)`
    /// * `attention_mask` - Attention mask. Sequence elements for which the
    ///    corresponding mask element is set to `false` are ignored
    ///    during attention calculation.
    ///   *Shape:* `(batch_size, seq_len)`
    /// * `positions` - Input positions.
    ///    *Shape:* `(batch_size, seq_len)`
    /// * `train` - Whether to train the layer.
    ///
    /// Returns layer output and the cache.
    /// *Shape:* ``(batch_size, seq_len, width)``
    fn forward_t(
        &self,
        input: &Tensor,
        mask: &AttentionMask,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, BoxedError>;
}

/// Trait for building encoder layers.
pub trait BuildEncoderLayer: Debug {
    /// Build a encoder layer.
    fn build_encoder_layer(&self, vb: VarBuilder) -> Result<Box<dyn EncoderLayer>, BoxedError>;
}
