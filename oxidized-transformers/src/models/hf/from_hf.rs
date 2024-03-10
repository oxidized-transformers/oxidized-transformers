use candle_core::{DType, Device};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use snafu::{ResultExt, Snafu};

use crate::architectures::BuildArchitecture;
use crate::error::BoxedError;
use crate::util::renaming_backend::RenamingBackend;

#[derive(Debug, Snafu)]
pub enum FromHFError {
    #[snafu(display("Cannot convert Hugging Face model config"))]
    ConvertConfig { source: BoxedError },

    #[snafu(display("Cannot build model"))]
    BuildModel { source: BoxedError },
}

/// Models that can be loaded from Huggingface transformers checkpoints.
pub trait FromHF {
    /// Model configuration.
    type Config: BuildArchitecture<Architecture = Self::Model>
        + TryFrom<Self::HFConfig, Error = BoxedError>;

    /// HF transformers model configuration.
    type HFConfig: Clone;

    /// The type of model that is constructed.
    ///
    /// Note that this is different from `Self`. `Self` is typically a
    /// unit struct that only implements various loading strategies.
    /// `Model` is a concrete model type such as `TransformerDecoder`.
    type Model;

    /// Construct a model from an HF model configuration and parameter backend.
    ///
    /// * `hf_config` - The Hugging Face transformers model configuration.
    /// * `backend` - The parameter store backend.
    /// * `device` - The device to place the model on.
    fn from_hf(
        hf_config: HFConfigWithDType<Self::HFConfig>,
        backend: Box<dyn SimpleBackend>,
        device: Device,
    ) -> Result<Self::Model, FromHFError> {
        // Ideally we would not clone here, but TryFrom<&...> adds a lot of
        // pesky lifetime annotations everywhere.
        let config =
            Self::Config::try_from(hf_config.config().clone()).context(ConvertConfigSnafu)?;
        let rename_backend = RenamingBackend::new(backend, Self::rename_parameters());
        let vb = VarBuilder::from_backend(Box::new(rename_backend), hf_config.dtype(), device);
        config.build(vb).context(BuildModelSnafu)
    }

    /// Create a parameter renaming function.
    ///
    /// This method should return a function that renames Oxidized Transformers
    /// parameter names to Hugging Face transformers parameter names.
    fn rename_parameters() -> impl Fn(&str) -> String + Send + Sync;
}

/// Torch dtype
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum TorchDType {
    BFloat16,
    Float16,
    Float32,
}

/// Simple wrapper for a HF config that exposes the dtype.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
pub struct HFConfigWithDType<T> {
    #[serde(flatten)]
    config: T,
    torch_dtype: TorchDType,
}

impl<T> HFConfigWithDType<T> {
    /// Get the configuration.
    pub fn config(&self) -> &T {
        &self.config
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        match self.torch_dtype {
            TorchDType::BFloat16 => DType::BF16,
            TorchDType::Float16 => DType::F16,
            TorchDType::Float32 => DType::F32,
        }
    }
}
