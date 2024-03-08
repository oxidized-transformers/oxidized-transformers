use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use candle_core::Tensor;
use snafu::{ResultExt, Snafu};

/// Errors in layer cache operations.
#[derive(Debug, Snafu)]
pub enum LayerKeyValueCacheError {
    #[snafu(display("Failed to extend key"))]
    ExtendKey { source: candle_core::Error },

    #[snafu(display("Failed to extend value"))]
    ExtendValue { source: candle_core::Error },
}

/// Internal representation of `LayerKeyValueCache`.
enum LayerKeyValueCacheEnum {
    Empty,
    Cache { key: Tensor, value: Tensor },
    NoCache,
}

/// Key-value cache for a layer.
pub struct LayerKeyValueCache(LayerKeyValueCacheEnum);

impl LayerKeyValueCache {
    /// Create an empty layer cache.
    pub fn empty() -> LayerKeyValueCache {
        LayerKeyValueCache(LayerKeyValueCacheEnum::Empty)
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
        use LayerKeyValueCacheEnum::*;
        match &self.0 {
            Cache { key, .. } => Some(key),
            Empty => None,
            NoCache => None,
        }
    }

    /// Get the cached value.
    pub fn value(&self) -> Option<&Tensor> {
        use LayerKeyValueCacheEnum::*;
        match &self.0 {
            Cache { value, .. } => Some(value),
            Empty => None,
            NoCache => None,
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
        use LayerKeyValueCacheEnum::*;
        match &mut self.0 {
            Cache { key, value } => {
                *key = Tensor::cat(&[&*key, new_key], 2).context(ExtendKeySnafu)?;
                *value = Tensor::cat(&[&*value, new_value], 2).context(ExtendKeySnafu)?;
            }
            Empty => {
                self.0 = Cache {
                    key: new_key.clone(),
                    value: new_value.clone(),
                }
            }

            NoCache => (),
        }

        Ok(())
    }
}

/// Internal representation of `KeyValueCache`.
enum KeyValueCacheEnum {
    #[allow(private_interfaces)]
    Cache {
        layer_caches: HashMap<usize, LayerKeyValueCache>,
    },

    #[allow(private_interfaces)]
    NoCache { stub: LayerKeyValueCache },
}

/// Cache type for layers that cache keys and values.
pub struct KeyValueCache(KeyValueCacheEnum);

impl KeyValueCache {
    /// Create a key-value cache.
    pub fn cache() -> KeyValueCache {
        Self(KeyValueCacheEnum::Cache {
            layer_caches: HashMap::new(),
        })
    }

    /// Create a no-op cache.
    ///
    /// This type of cache does not store anything. Updates to the cache are
    /// discarded.
    pub fn no_cache() -> Self {
        Self(KeyValueCacheEnum::NoCache {
            stub: LayerKeyValueCache::no_cache(),
        })
    }
}

impl Index<usize> for KeyValueCache {
    type Output = LayerKeyValueCache;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.0 {
            KeyValueCacheEnum::Cache { layer_caches } => &layer_caches[&index],
            KeyValueCacheEnum::NoCache { stub, .. } => stub,
        }
    }
}

impl IndexMut<usize> for KeyValueCache {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.0 {
            KeyValueCacheEnum::Cache { layer_caches } => layer_caches
                .entry(index)
                .or_insert(LayerKeyValueCache::empty()),
            KeyValueCacheEnum::NoCache { stub, .. } => stub,
        }
    }
}
