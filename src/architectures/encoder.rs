use std::fmt::Debug;

use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::architectures::output::LayerOutputs;
use crate::error::BoxedError;
use crate::layers::attention::AttentionMask;

/// Encoder output.
pub struct EncoderOutput {
    all_outputs: Vec<Tensor>,
}

impl EncoderOutput {
    pub fn new(all_outputs: Vec<Tensor>) -> Self {
        Self { all_outputs }
    }
}

impl LayerOutputs for EncoderOutput {
    fn layer_outputs(&self) -> &[Tensor] {
        &self.all_outputs
    }

    fn embedding_layer_output(&self) -> Option<&Tensor> {
        self.all_outputs.first()
    }
}

/// Trait for encoder layers.
pub trait EncoderLayer {
    /// Apply the encoder layer to the given hidden representations.
    ///
    /// * `piece_ids` - Hidden representations to apply the layer to.
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
        piece_ids: &Tensor,
        attention_mask: &AttentionMask,
        positions: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, BoxedError>;
}

/// Trait for building encoder layers.
pub trait BuildEncoderLayer: Debug {
    /// Build a encoder layer.
    fn build_encoder_layer(&self, vb: VarBuilder) -> Result<Box<dyn EncoderLayer>, BoxedError>;
}
