pub mod renaming_backend;

#[cfg(test)]
pub(crate) mod tests {
    use candle_core::Tensor;

    pub(crate) fn assert_close(a: &Tensor, b: &Tensor, atol: f32) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Shape mismatch: {:?}, {:?}",
            a.shape(),
            b.shape()
        );

        let a: Vec<f32> = a
            .flatten_all()
            .unwrap()
            .to_vec1()
            .expect("Cannot convert Tensor to Vec<f32>");
        let b: Vec<f32> = b
            .flatten_all()
            .unwrap()
            .to_vec1()
            .expect("Cannot convert Tensor to Vec<f32>");

        for (x, y) in a.into_iter().zip(b.into_iter()) {
            assert!((x - y).abs() < atol)
        }
    }
}
