use candle_core::Tensor;

/// Trait for querying layer outputs of a model.
pub trait LayerOutputs {
    /// Outputs of all layers.
    fn layer_outputs(&self) -> &[Tensor];

    /// Output of the embedding layer.
    fn embedding_layer_output(&self) -> Option<&Tensor>;
}
