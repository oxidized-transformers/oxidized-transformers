/// Traits for model architectures.
mod decoder;
pub use decoder::{BuildDecoderLayer, DecoderLayer, DecoderOutput};

mod encoder;
pub use encoder::{BuildEncoderLayer, EncoderLayer, EncoderOutput};

mod output;
pub use output::LayerOutputs;
