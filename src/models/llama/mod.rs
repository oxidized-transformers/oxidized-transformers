//! Llama architectures (Touvron et al., 2023).
mod causal_lm;
pub use causal_lm::LlamaCausalLM;

mod decoder;
pub use decoder::LlamaDecoder;
