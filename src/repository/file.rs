use std::path::Path;

/// Represents a file in a repository.
///
/// Repository files can be a local path or a remote path exposed as a
/// file-like object. This trait is implemented for such different types
/// of repository files.
pub trait RepoFile
where
    Self: Sized,
{
    /// Get the local path of the file, if it exists.
    fn local_path(&self) -> Option<impl AsRef<Path>>;

    /// Check if the file exists.
    fn exists(&self) -> bool;
}
