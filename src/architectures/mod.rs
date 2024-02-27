/// Traits for model architectures.
mod causal_lm;

use candle_nn::VarBuilder;
pub use causal_lm::{BuildCausalLM, CausalLM, CausalLMOutput};

mod decoder;
pub use decoder::{BuildDecoder, BuildDecoderLayer, Decoder, DecoderLayer, DecoderOutput};

mod encoder;
pub use encoder::{BuildEncoderLayer, EncoderLayer, EncoderOutput};

mod output;
use crate::error::BoxedError;
pub use output::LayerOutputs;

/// Trait for building model architectures.
pub trait BuildArchitecture {
    /// The architecture to build.
    type Architecture;

    /// Build the architecture.
    fn build(&self, vb: VarBuilder) -> Result<Self::Architecture, BoxedError>;
}
