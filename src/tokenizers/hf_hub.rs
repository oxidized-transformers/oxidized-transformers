use super::tokenizer::FromRepo;
use crate::error::BoxedError;
use crate::repository::hf_hub::HfHubRepo;

/// Trait implemented by tokenziers that can be loaded from the Hugging Face Hub.
pub trait FromHFHub
where
    Self: FromRepo,
{
    /// Load a tokenizer from the Hugging Face Hub.
    ///
    /// * name - Name of the model on the Hugging Face Hub.
    /// * revision - Revision of the model to load. If `None`, the main branch is used.
    ///
    /// Returns: The tokenizer loaded from the Hugging Face Hub.
    fn from_hf_hub(name: &str, revision: Option<&str>) -> Result<Self, BoxedError> {
        let hf_repo = HfHubRepo::new(name, revision)?;
        Self::from_repo(&hf_repo)
    }
}
