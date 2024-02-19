use std::collections::HashSet;
use std::path::Path;

use hf_hub::api::sync::{Api, ApiError, ApiRepo};
use hf_hub::{Repo as HuggingFaceRepo, RepoType as HuggingFaceRepoType};
use snafu::{ResultExt, Snafu};

use super::file::RepoFile;
use super::local_file::LocalFile;
use super::repo::Repo;
use crate::error::BoxedError;

/// `HfHubRepo` errors.
#[derive(Debug, Snafu)]
pub enum HfHubRepoError {
    #[snafu(display("Couldn't initialize Hugging Face Hub API"))]
    InitializeAPI { source: ApiError },

    #[snafu(display("Couldn't fetch metadata for Hugging Face Hub repo"))]
    FetchRepoMetadata { source: ApiError },

    #[snafu(display("Couldn't download remote file at '{path}'"))]
    GetRemoteFile { path: String, source: ApiError },
}

/// Hugging Face Hub repository.
pub struct HfHubRepo {
    api_repo: ApiRepo,
    remote_files: HashSet<String>,
}

impl HfHubRepo {
    /// Create a new Hugging Face Hub repository.
    ///
    /// * `name` - Name of the model on the Hugging Face Hub.
    /// * `revision` - Revision of the model to load. If `None`, the main branch is used.
    pub fn new(name: &str, revision: Option<&str>) -> Result<Self, BoxedError> {
        let revision = revision.unwrap_or("main").to_owned();

        let hub_api = Api::new().context(InitializeAPISnafu)?;
        let api_repo = hub_api.repo(HuggingFaceRepo::with_revision(
            name.to_owned(),
            HuggingFaceRepoType::Model,
            revision,
        ));
        let repo_info = api_repo.info().context(FetchRepoMetadataSnafu)?;

        Ok(Self {
            api_repo,
            remote_files: repo_info
                .siblings
                .iter()
                .map(|f| f.rfilename.clone())
                .collect(),
        })
    }

    fn remote_path_exists(&self, path: &str) -> bool {
        self.remote_files.contains(path)
    }
}

impl Repo for HfHubRepo {
    type File = HfHubFile;

    fn file(&self, path: impl AsRef<Path>) -> Result<Self::File, BoxedError> {
        let path_str = path.as_ref().to_string_lossy();
        if self.remote_path_exists(&path_str) {
            let local_path = self
                .api_repo
                .get(&path_str)
                .context(GetRemoteFileSnafu { path: path_str })?;

            Ok(HfHubFile::local_file(local_path))
        } else {
            Ok(HfHubFile::non_existent())
        }
    }
}

/// Wraps either a remote file on a Hugging Face Hub repository
/// or a local file in the Hugging Face cache.
pub enum HfHubFile {
    Local(LocalFile),
    NonExistent,
}

impl HfHubFile {
    /// Create a new `HfHubFile` from a local file.
    ///
    /// * `local_path` - The path of the file on the local disk.
    pub fn local_file(local_path: impl AsRef<Path>) -> Self {
        Self::Local(LocalFile::new(local_path))
    }

    /// Create a new `HfHubFile` that represents a non-existent (remote) file.
    pub fn non_existent() -> Self {
        Self::NonExistent
    }
}

impl RepoFile for HfHubFile {
    fn local_path(&self) -> Option<impl AsRef<Path>> {
        match self {
            Self::Local(l) => l.local_path(),
            _ => None,
        }
    }

    fn exists(&self) -> bool {
        match self {
            Self::Local(l) => l.exists(),
            _ => false,
        }
    }
}
