use std::path::Path;

use super::file::RepoFile;
use crate::error::BoxedError;

/// Represents a repository that contains a model or tokenizer.
pub trait Repo
where
    Self: Sized,
{
    type File: RepoFile;

    /// Get a repository file.
    ///
    /// * `path` - The path to the file within the repository.
    ///
    /// Returns: A repository file object.
    fn file(&self, path: impl AsRef<Path>) -> Result<Self::File, BoxedError>;
}
