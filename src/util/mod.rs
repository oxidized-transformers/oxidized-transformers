pub mod renaming_backend;

#[cfg(test)]
pub(crate) mod tests {
    use std::error::Error;
    use std::fmt::Debug;

    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use candle_core::{Tensor, WithDType};
    use ndarray::{ArrayBase, ArrayD, DataOwned, Dimension};

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
}
