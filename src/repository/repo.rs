use std::path::{Path, PathBuf};

use crate::error::BoxedError;

/// Represents a repository that contains a model or tokenizer.
pub trait Repo
where
    Self: Sized,
{
    /// Get a repository file.
    ///
    /// * `path` - The path to the file within the repository.
    ///
    /// Returns: The local file path.
    fn file(&self, path: impl AsRef<Path>) -> Result<Option<PathBuf>, BoxedError>;
}
