use candle_core::Tensor;

use crate::layers::attention::AttentionMask;

/// Cache type for layers that cache keys and values.
pub struct KeyValueCache {
    pub key: Tensor,
    pub value: Tensor,
    pub attention_mask: AttentionMask,
}
