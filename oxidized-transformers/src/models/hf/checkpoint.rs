use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use candle_core::pickle::{read_pth_tensor_info, PthTensors};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::Init;
use serde::Deserialize;
use snafu::{ensure, ResultExt, Snafu};

use crate::error::BoxedError;
use crate::repository::repo::Repo;

static PYTORCH_INDEX: &str = "pytorch_model.bin.index.json";
static PYTORCH_SINGLE: &str = "pytorch_model.bin";
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

    #[snafu(display("Checkpoint index file does not exist: {}", name))]
    NonExistentCheckpointIndex { name: String },

    #[snafu(display("Cannot open index file: {}", path.to_string_lossy()))]
    OpenCheckpointIndex { source: io::Error, path: PathBuf },

    #[snafu(display("Cannot parse index file: {}", path.to_string_lossy()))]
    ParseCheckpointIndex {
        source: serde_json::Error,
        path: PathBuf,
    },
}

impl<R> LoadHFCheckpoint for R
where
    R: Repo,
{
    fn load_hf_checkpoint(&self) -> Result<Box<dyn SimpleBackend>, BoxedError> {
        if self.has_safetensors() {
            self.load_safetensors()
        } else {
            self.load_pytorch_state_dict()
        }
    }
}

/// Private trait for loading safetensors checkpoints.
///
/// This trait is used to load a safetensors checkpoint from the repository.
/// We just use it so that we can implement these methods on `Repo` rather
/// than having them as standalone functions.
trait LoadHFSafeTensors {
    fn has_safetensors(&self) -> bool;

    fn load_safetensors(&self) -> Result<Box<dyn SimpleBackend>, BoxedError>;
}

impl<R> LoadHFSafeTensors for R
where
    R: Repo,
{
    /// Check if the repository has a safetensors checkpoint.
    fn has_safetensors(&self) -> bool {
        self.exists(SAFETENSORS_SINGLE) || self.exists(SAFETENSORS_INDEX)
    }

    /// Load a safetensors checkpoint.
    ///
    /// This method will first probe if there is a shard index. If there is,
    /// a sharded checkpoint will be loaded. Otherwise, a single-file
    /// checkpoint is loaded.
    fn load_safetensors(&self) -> Result<Box<dyn SimpleBackend>, BoxedError> {
        let checkpoint = if self.exists(SAFETENSORS_INDEX) {
            // We have a safetensor index, so load from shards.
            HFCheckpoint::Multiple(SAFETENSORS_INDEX.into())
        } else {
            // No index file, so assume that there is a single checkpoint.
            HFCheckpoint::Single(SAFETENSORS_SINGLE.into())
        };

        let paths = checkpoint.load(self)?;

        Ok(Box::new(unsafe {
            MmapedSafetensors::multi(&paths).context(LoadCheckpointSnafu)?
        }))
    }
}

/// Private trait for loading PyTorch state dict checkpoints.
trait LoadHFPyTorchStateDict {
    fn load_pytorch_state_dict(&self) -> Result<Box<dyn SimpleBackend>, BoxedError>;
}

impl<R> LoadHFPyTorchStateDict for R
where
    R: Repo,
{
    fn load_pytorch_state_dict(&self) -> Result<Box<dyn SimpleBackend>, BoxedError> {
        let checkpoint = if self.exists(PYTORCH_INDEX) {
            HFCheckpoint::Multiple(PYTORCH_INDEX.into())
        } else {
            HFCheckpoint::Single(PYTORCH_SINGLE.into())
        };

        let paths = checkpoint.load(self)?;

        Ok(Box::new(
            PyTorchTensors::multi(&paths).context(LoadCheckpointSnafu)?,
        ))
    }
}
struct PyTorchTensors {
    tensors: Vec<PthTensors>,
    routing: HashMap<String, usize>,
}

impl PyTorchTensors {
    fn multi(paths: &[impl AsRef<Path>]) -> Result<Self, candle_core::Error> {
        let mut routing = HashMap::new();
        let mut tensors = vec![];

        for (index, p) in paths.iter().enumerate() {
            // We need to read the tensor metadata separately as PthTensors
            // does not expose this information.
            let tensor_names: Vec<String> = read_pth_tensor_info(p, false, None)?
                .into_iter()
                .map(|ti| ti.name.clone())
                .collect();

            let pth = PthTensors::new(p, None)?;
            for k in tensor_names {
                routing.insert(k, index);
            }
            tensors.push(pth);
        }
        Ok(Self { tensors, routing })
    }

    fn load(&self, name: &str, dev: &Device) -> Result<Tensor, candle_core::Error> {
        self.get(name)?.to_device(dev)
    }

    fn get(&self, name: &str) -> Result<Tensor, candle_core::Error> {
        let index =
            *self
                .routing
                .get(name)
                .ok_or_else(|| candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                })?;

        self.tensors[index]
            .get(name)?
            .ok_or_else(|| candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            })
    }
}

impl SimpleBackend for PyTorchTensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let tensor = self.load(name, dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

enum HFCheckpoint {
    Single(PathBuf),
    Multiple(PathBuf),
}

impl HFCheckpoint {
    fn load(&self, repo: &impl Repo) -> Result<Vec<PathBuf>, BoxedError> {
        match self {
            HFCheckpoint::Single(checkpoint_path) => {
                let path = repo.file(checkpoint_path).context(DownloadSnafu {
                    name: checkpoint_path.to_string_lossy(),
                })?;

                ensure!(
                    path.is_some(),
                    NonExistentCheckpointSnafu {
                        name: checkpoint_path.to_string_lossy(),
                    }
                );

                Ok(vec![path.unwrap()])
            }
            HFCheckpoint::Multiple(index_path) => {
                let path = repo
                    .file(index_path)
                    .context(DownloadSnafu {
                        name: index_path.to_string_lossy(),
                    })?
                    .ok_or_else(|| HFCheckpointError::NonExistentCheckpointIndex {
                        name: index_path.to_string_lossy().to_string(),
                    })?;

                // Parse the shard index.
                let index_file = BufReader::new(
                    File::open(&path).context(OpenCheckpointIndexSnafu { path: path.clone() })?,
                );
                let index: HFCheckpointIndex = serde_json::from_reader(index_file)
                    .context(ParseCheckpointIndexSnafu { path: path.clone() })?;

                let shard_names = index.shards();
                let mut shards = Vec::with_capacity(shard_names.len());
                for shard_name in shard_names {
                    let path = repo.file(&shard_name).context(DownloadSnafu {
                        name: shard_name.clone(),
                    })?;
                    ensure!(path.is_some(), NonExistentShardSnafu { name: shard_name });
                    shards.push(path.unwrap());
                }

                Ok(shards)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct HFCheckpointIndex {
    weight_map: HashMap<String, String>,
}

impl HFCheckpointIndex {
    /// Get the names of the shards.
    fn shards(&self) -> HashSet<String> {
        self.weight_map.values().cloned().collect()
    }
}
