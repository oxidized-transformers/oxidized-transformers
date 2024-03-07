use std::iter::repeat_with;
use std::ops::{Index, IndexMut};

use candle_core::{DType, Device, Tensor};
use snafu::{ResultExt, Snafu};

/// Errors in layer cache operations.
#[derive(Debug, Snafu)]
pub enum LayerKeyValueCacheError {
    #[snafu(display("Failed to create empty key"))]
    CreateEmptyKey { source: candle_core::Error },

    #[snafu(display("Failed to create empty value"))]
    CreateEmptyValue { source: candle_core::Error },

    #[snafu(display("Failed to extend key"))]
    ExtendKey { source: candle_core::Error },

    #[snafu(display("Failed to extend value"))]
    ExtendValue { source: candle_core::Error },
}

/// Internal representation of `LayerKeyValueCache`.
enum LayerKeyValueCacheEnum {
    Cache { key: Tensor, value: Tensor },

    NoCache,
}

/// Key-value cache for a layer.
pub struct LayerKeyValueCache(LayerKeyValueCacheEnum);

impl LayerKeyValueCache {
    /// Create an empty layer cache.
    ///
    /// * `batch_size` - Batch size.
    /// * `hidden_width` - Hidden width.
    /// * `n_key_value_heads` - Number of key-value heads.
    /// * `dtype` - Cache data type.
    /// * `device` - Device to store the cache on.
    pub fn cache(
        batch_size: usize,
        hidden_width: usize,
        n_key_value_heads: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, LayerKeyValueCacheError> {
        Ok(LayerKeyValueCache(LayerKeyValueCacheEnum::Cache {
            key: Tensor::zeros(
                (
                    batch_size,
                    n_key_value_heads,
                    0,
                    hidden_width / n_key_value_heads,
                ),
                dtype,
                device,
            )
            .context(CreateEmptyKeySnafu)?,
            value: Tensor::zeros(
                (
                    batch_size,
                    n_key_value_heads,
                    0,
                    hidden_width / n_key_value_heads,
                ),
                dtype,
                device,
            )
            .context(CreateEmptyValueSnafu)?,
        }))
    }

    /// Create a no-op cache.
    ///
    /// This type of cache does not store anything. Updates to the cache are
    /// discarded.
    pub fn no_cache() -> Self {
        Self(LayerKeyValueCacheEnum::NoCache)
    }

    /// Get the cached key.
    pub fn key(&self) -> Option<&Tensor> {
        match &self.0 {
            LayerKeyValueCacheEnum::Cache { key, .. } => Some(key),
            LayerKeyValueCacheEnum::NoCache => None,
        }
    }

    /// Get the cached value.
    pub fn value(&self) -> Option<&Tensor> {
        match &self.0 {
            LayerKeyValueCacheEnum::Cache { value, .. } => Some(value),
            LayerKeyValueCacheEnum::NoCache => None,
        }
    }

    /// Update the cache.
    ///
    /// This adds the new key/value tensors to the cache.
    ///
    /// * `new_key` - New key tensor.
    /// * `new_value` - New value tensor.
    pub fn update(
        &mut self,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<(), LayerKeyValueCacheError> {
        match &mut self.0 {
            LayerKeyValueCacheEnum::Cache { key, value } => {
                *key = Tensor::cat(&[&*key, new_key], 2).context(ExtendKeySnafu)?;
                *value = Tensor::cat(&[&*value, new_value], 2).context(ExtendKeySnafu)?;
            }
            LayerKeyValueCacheEnum::NoCache => (),
        }

        Ok(())
    }
}

/// Key-value cache errors.
#[derive(Debug, Snafu)]
pub enum KeyValueCacheError {
    #[snafu(display("Failed to create layer cache"))]
    CreateLayerKeyValueCache { source: LayerKeyValueCacheError },
}

/// Internal representation of `KeyValueCache`.
enum KeyValueCacheEnum {
    #[allow(private_interfaces)]
    Cache {
        layer_caches: Vec<LayerKeyValueCache>,
    },

    #[allow(private_interfaces)]
    NoCache {
        stub: LayerKeyValueCache,

        // Note: it's a bit nonsensical to have this, but we need it to
        // return the number of layers in the cache uniformly.
        n_layers: usize,
    },
}

/// Cache type for layers that cache keys and values.
pub struct KeyValueCache(KeyValueCacheEnum);

impl KeyValueCache {
    /// Create a key-value cache.
    ///
    /// * `batch_size` - Batch size.
    /// * `hidden_width` - Hidden width.
    /// * `n_key_value_heads` - Number of key-value heads.
    /// * `n_layers` - Number of hidden layers.
    /// * `dtype` - Cache data type.
    /// * `device` - Device to store the cache on.
    pub fn cache(
        batch_size: usize,
        hidden_width: usize,
        n_key_value_heads: usize,
        n_layers: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, KeyValueCacheError> {
        let layer_caches = repeat_with(|| {
            LayerKeyValueCache::cache(batch_size, hidden_width, n_key_value_heads, dtype, device)
        })
        .take(n_layers)
        .collect::<Result<Vec<_>, _>>()
        .context(CreateLayerKeyValueCacheSnafu)?;

        Ok(Self(KeyValueCacheEnum::Cache { layer_caches }))
    }

    /// Get the number of layers in the cache.
    pub fn n_layers(&self) -> usize {
        match &self.0 {
            KeyValueCacheEnum::Cache { layer_caches } => layer_caches.len(),
            KeyValueCacheEnum::NoCache { n_layers, .. } => *n_layers,
        }
    }

    /// Create a no-op cache.
    ///
    /// This type of cache does not store anything. Updates to the cache are
    /// discarded.
    pub fn no_cache(n_layers: usize) -> Self {
        Self(KeyValueCacheEnum::NoCache {
            stub: LayerKeyValueCache::no_cache(),
            n_layers,
        })
    }
}

impl Index<usize> for KeyValueCache {
    type Output = LayerKeyValueCache;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.0 {
            KeyValueCacheEnum::Cache { layer_caches } => &layer_caches[index],
            KeyValueCacheEnum::NoCache { stub, .. } => stub,
        }
    }
}

impl IndexMut<usize> for KeyValueCache {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.0 {
            KeyValueCacheEnum::Cache { layer_caches } => &mut layer_caches[index],
            KeyValueCacheEnum::NoCache { stub, .. } => stub,
        }
    }
}
