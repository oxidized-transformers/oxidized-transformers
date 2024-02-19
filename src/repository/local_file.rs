use std::path::{Path, PathBuf};

use snafu::Snafu;

use super::file::RepoFile;

/// `LocalFile` errors.
#[derive(Debug, Snafu)]
pub enum LocalFileError {
    #[snafu(display("Couldn't open file"))]
    Open { source: std::io::Error },
}

/// Repository file on the local machine.
pub struct LocalFile {
    path: PathBuf,
}

impl LocalFile {
    /// Create a new local file.
    ///
    /// * `path` - The path to the file on the local machine.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl RepoFile for LocalFile {
    fn local_path(&self) -> Option<impl AsRef<Path>> {
        Some(self.path.as_path())
    }

    fn exists(&self) -> bool {
        self.path.exists()
    }
}
