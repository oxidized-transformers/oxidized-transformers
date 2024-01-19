use candle_core::Tensor;

/// Cache type for layers that cache keys and values.
pub struct KeyValueCache {
    pub key: Tensor,
    pub value: Tensor,
}
