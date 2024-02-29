use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::Init;

/// A backend that renames tensors before passing them to the inner backend.
pub struct RenamingBackend<F>
where
    F: Fn(&str) -> String + Send + Sync,
{
    inner: Box<dyn SimpleBackend>,
    rename_to_inner: F,
}

impl<F> RenamingBackend<F>
where
    F: Fn(&str) -> String + Send + Sync,
{
    /// Create a new renaming backend.
    pub fn new(inner: Box<dyn SimpleBackend>, rename_to_inner: F) -> Self {
        Self {
            inner,
            rename_to_inner,
        }
    }
}

impl<F> SimpleBackend for RenamingBackend<F>
where
    F: Fn(&str) -> String + Send + Sync,
{
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.inner
            .get(s, &(self.rename_to_inner)(name), h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(&(self.rename_to_inner)(name))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use candle_core::{DType, Device};
    use candle_nn::init::ZERO;
    use candle_nn::{VarBuilder, VarMap};

    use crate::util::renaming_backend::RenamingBackend;

    #[test]
    fn renaming_backend_renames() {
        let var_map = VarMap::new();
        let rename = RenamingBackend::new(Box::new(var_map.clone()), |name| {
            let mut name = name.to_string();
            name.insert_str(0, "roberta.");
            name = name.replace("piece_embeddings", "word_embeddings");
            name
        });
        let vb = VarBuilder::from_backend(Box::new(rename), DType::F32, Device::Cpu);
        vb.get_with_hints((100, 32), "piece_embeddings.weight", ZERO)
            .unwrap();
        vb.get_with_hints((100, 32), "type_embeddings.weight", ZERO)
            .unwrap();
        let data = var_map.data().lock().unwrap();
        assert_eq!(
            data.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from([
                "roberta.word_embeddings.weight".to_string(),
                "roberta.type_embeddings.weight".to_string()
            ])
        );
    }
}
