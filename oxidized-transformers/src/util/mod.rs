pub mod renaming_backend;

#[cfg(test)]
pub(crate) mod tests {
    use std::error::Error;
    use std::fmt::Debug;

    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use candle_core::{Device, Tensor, WithDType, D};
    use ndarray::{ArrayBase, ArrayD, DataOwned, Dimension};
    use rand_core::RngCore;
    use rand_pcg::Pcg32;
    use snafu::{ResultExt, Snafu, Whatever};

    use crate::layers::attention::AttentionMask;

    // Like TryInto, but we need our own trait so that we can implement it
    // for external types.
    pub trait IntoArrayD<T> {
        fn into_arrayd(self) -> Result<ArrayD<T>, Box<dyn Error>>;
    }

    impl<T> IntoArrayD<T> for Tensor
    where
        T: WithDType,
    {
        fn into_arrayd(self) -> Result<ArrayD<T>, Box<dyn Error>> {
            (&self).into_arrayd()
        }
    }

    impl<T> IntoArrayD<T> for &Tensor
    where
        T: WithDType,
    {
        fn into_arrayd(self) -> Result<ArrayD<T>, Box<dyn Error>> {
            let data = self.reshape(((),))?.to_vec1()?;
            Ok(ArrayD::from_shape_vec(self.shape().dims(), data)?)
        }
    }

    impl<S, D, T> IntoArrayD<T> for ArrayBase<S, D>
    where
        D: Dimension,
        S: DataOwned<Elem = T>,
        T: Clone,
    {
        fn into_arrayd(self) -> Result<ArrayD<T>, Box<dyn Error>> where {
            Ok(self.to_owned().into_dyn())
        }
    }

    pub(crate) fn assert_tensor_eq<T>(a: impl IntoArrayD<T>, b: impl IntoArrayD<T>, epsilon: T)
    where
        T: AbsDiffEq<Epsilon = T> + Clone + Debug,
    {
        let a = a.into_arrayd().expect("Cannot convert array");
        let b = b.into_arrayd().expect("Cannot convert array");

        assert_eq!(
            a.shape(),
            b.shape(),
            "Shape mismatch: {:?}, {:?}",
            a.shape(),
            b.shape()
        );

        assert_abs_diff_eq!(a, b, epsilon = epsilon);
    }

    /// Generate vectors with a PRNG.
    trait PseudoRandom {
        /// Generate a vector with a PRNG.
        ///
        /// This method generates a vector with the given length. The seed of
        /// the PRNG is set to the given length.
        ///
        /// * `len` - The length of the vector to generate.
        /// * `device` - The device to allocate the tensor on.
        fn pseudo_random(len: usize, device: &Device) -> Self;
    }

    impl PseudoRandom for Tensor {
        fn pseudo_random(len: usize, device: &Device) -> Self {
            let mut rng = Pcg32::new(len as u64, 0);
            let iter = (0..len).map(|_| {
                let next = rng.next_u32();

                // Generate a uniform random number in [0, 1). We don't use
                // rand's uniform sampler, because we want full control over
                // the sampling. This allows us to mirror the sampling in
                // Python-land for getting test vectors and test vectors don't
                // get invalidated by changes in the rand crate.
                let mantissa_bits_shift = u32::BITS - f32::MANTISSA_DIGITS;
                let zero_one =
                    (next >> mantissa_bits_shift) as f32 / (1 << f32::MANTISSA_DIGITS) as f32;

                // We have not used the least significant bit while generating
                // the random number, so we can use it to pick the sign.
                let sign = (next & 1) as f32;
                zero_one - sign
            });
            Tensor::from_iter(iter, device).expect("Cannot allocate random tensor")
        }
    }

    #[derive(Debug, Snafu)]
    pub enum PseudoRandomReductionError {
        #[snafu(display("Cannot calculate matmul of tensor and random vector"))]
        MatMul { source: candle_core::Error },

        #[snafu(display("Cannot get size of last dimension, tensor is a scalar"))]
        Scalar { source: candle_core::Error },

        #[snafu(display("Cannot reshape"))]
        Shape { source: candle_core::Error },
    }

    /// Vector reduction using pseudo-random vectors.
    pub trait PseudoRandomReduction {
        /// Reduce the tensor using a pseudo-random vector.
        ///
        /// Compute an aggregate of the last dimension of a tensor by
        /// computing the dot product of last-dimension vectors with a
        /// pseudo-random vector.
        fn pseudo_random_reduction(self) -> Result<Tensor, PseudoRandomReductionError>;
    }

    impl PseudoRandomReduction for &Tensor {
        fn pseudo_random_reduction(self) -> Result<Tensor, PseudoRandomReductionError> {
            let size = self.dim(D::Minus1).context(ScalarSnafu)?;
            let random = Tensor::pseudo_random(size, self.device())
                .reshape((size, 1))
                .context(ShapeSnafu)?;
            self.broadcast_matmul(&random)
                .context(MatMulSnafu)?
                .squeeze(D::Minus1)
                .context(ShapeSnafu)
        }
    }

    /// Sample transformer inputs used for most tests.
    pub fn sample_transformer_inputs() -> Result<(Tensor, AttentionMask), Whatever> {
        let input = Tensor::arange(0i64, 24, &Device::Cpu)
            .and_then(|t| t.reshape((3, 8)))
            .whatever_context("Cannot create input tensor")?;

        let mask = Tensor::from_slice(
            &[
                1u32, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
            ],
            (3, 8),
            &Device::Cpu,
        )
        .whatever_context("Cannot create attention mask tensor")?;

        Ok((
            input,
            AttentionMask::new(mask).whatever_context("Cannot create attention mask")?,
        ))
    }
}
