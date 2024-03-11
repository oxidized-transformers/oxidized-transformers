/// RoBERTa architectures
mod embeddings;
pub use embeddings::{RobertaEmbeddings, RobertaEmbeddingsConfig};

mod encoder;
pub(crate) use encoder::HFRobertaEncoderConfig;
pub use encoder::RobertaEncoder;
