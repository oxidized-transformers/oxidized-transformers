/// Traits for model architectures.
mod causal_lm;
pub use causal_lm::{BuildCausalLM, CausalLM, CausalLMOutput};

mod decoder;
pub use decoder::{BuildDecoder, BuildDecoderLayer, Decoder, DecoderLayer, DecoderOutput};

mod encoder;
pub use encoder::{BuildEncoderLayer, EncoderLayer, EncoderOutput};

mod output;
pub use output::LayerOutputs;
