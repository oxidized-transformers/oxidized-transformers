use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use candle_core::safetensors::MmapedSafetensors;
use candle_nn::var_builder::SimpleBackend;
use serde::Deserialize;
use snafu::{ensure, ResultExt, Snafu};

use crate::error::BoxedError;
use crate::repository::repo::Repo;

static SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
static SAFETENSORS_SINGLE: &str = "model.safetensors";

/// Extension trait for HF transformers checkpoint loading.
///
/// This trait has a single implementation that adds support to `Repo` for
/// loading Hugging Face transformers-style checkpoints from a repository.
pub trait LoadHFCheckpoint {
    /// Load a Hugging Face transformers checkpoint.
    fn load_hf_checkpoint(&self) -> Result<Box<dyn SimpleBackend>, BoxedError>;
}

/// HF transformers checkpoint loading errors.
#[derive(Debug, Snafu)]
pub enum HFCheckpointError {
    #[snafu(display("Cannot download checkpoint: {name}"))]
    Download { source: BoxedError, name: String },

    #[snafu(display("Cannot open or load checkpoint"))]
    LoadCheckpoint { source: candle_core::Error },

    #[snafu(display("Checkpoint does not exist: {}", name))]
    NonExistentCheckpoint { name: String },

    #[snafu(display("Shard does not exist: {}", name))]
    NonExistentShard { name: String },

    #[snafu(display("Cannot open SafeTensors index file: {}", path.to_string_lossy()))]
    OpenSafeTensorsIndex { source: io::Error, path: PathBuf },

    #[snafu(display("Cannot parse SafeTensors index file: {}", path.to_string_lossy()))]
    ParseSafeTensorsIndex {
        source: serde_json::Error,
        path: PathBuf,
    },
}

impl<R> LoadHFCheckpoint for R
where
    R: Repo,
{
    fn load_hf_checkpoint(&self) -> Result<Box<dyn SimpleBackend>, BoxedError> {
        self.load_safetensors()
    }
}

/// Private trait for loading safetensors checkpoint.
///
/// This trait is used to load a safetensors checkpoint from the repository.
/// We just use it so that we can implement these methods on `Repo` rather
/// than having them as standalone functions.
trait LoadHFSafeTensors {
    fn load_safetensors(&self) -> Result<Box<dyn SimpleBackend>, BoxedError>;

    fn load_safetensors_multi(&self, index_path: &Path) -> Result<Vec<PathBuf>, BoxedError>;

    fn load_safetensors_single(&self) -> Result<Vec<PathBuf>, BoxedError>;
}

impl<R> LoadHFSafeTensors for R
where
    R: Repo,
{
    /// Load a safetensors checkpoint.
    ///
    /// This method will first probe if there is a shard index. If there is,
    /// a sharded checkpoint will be loaded. Otherwise, a single-file
    /// checkpoint is loaded.
    fn load_safetensors(&self) -> Result<Box<dyn SimpleBackend>, BoxedError> {
        let file = self.file(SAFETENSORS_INDEX).context(DownloadSnafu {
            name: SAFETENSORS_INDEX,
        })?;

        let paths = match file {
            // We have a safetensor index, so load from shards.
            Some(index_path) => self.load_safetensors_multi(&index_path),

            // No index file, so assume that there is a single checkpoint.
            None => self.load_safetensors_single(),
        }?;

        Ok(Box::new(unsafe {
            MmapedSafetensors::multi(&paths).context(LoadCheckpointSnafu)?
        }))
    }

    /// Load sharded safetensors checkpoint.
    fn load_safetensors_multi(&self, index_path: &Path) -> Result<Vec<PathBuf>, BoxedError> {
        // Parse the shard index.
        let index_file = BufReader::new(
            File::open(index_path).context(OpenSafeTensorsIndexSnafu { path: index_path })?,
        );
        let index: SafeTensorsIndex = serde_json::from_reader(index_file)
            .context(ParseSafeTensorsIndexSnafu { path: index_path })?;

        let shard_names = index.shards();
        let mut shards = Vec::with_capacity(shard_names.len());
        for shard_name in shard_names {
            let path = self.file(&shard_name).context(DownloadSnafu {
                name: shard_name.clone(),
            })?;
            ensure!(path.is_some(), NonExistentShardSnafu { name: shard_name });
            shards.push(path.unwrap());
        }

        Ok(shards)
    }

    /// Load non-sharded safetensors checkpoint.
    fn load_safetensors_single(&self) -> Result<Vec<PathBuf>, BoxedError> {
        let path = self.file(SAFETENSORS_SINGLE).context(DownloadSnafu {
            name: SAFETENSORS_SINGLE,
        })?;

        ensure!(
            path.is_some(),
            NonExistentCheckpointSnafu {
                name: SAFETENSORS_SINGLE.to_string(),
            }
        );

        Ok(vec![path.unwrap()])
    }
}

#[derive(Debug, Deserialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    /// Get the names of the shards.
    fn shards(&self) -> HashSet<String> {
        self.weight_map.values().cloned().collect()
    }
}
