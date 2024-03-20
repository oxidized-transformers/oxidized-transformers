use candle_core::{bail, CpuStorage, CustomOp1, Layout, Shape, Tensor, WithDType};
use snafu::{ResultExt, Snafu};

#[cfg(feature = "cuda")]
use candle_core::CudaStorage;

/// Non-zero indices errors.
#[derive(Debug, Snafu)]
pub enum NonzeroError {
    #[snafu(display("Cannot apply nonzero op"))]
    ApplyOp { source: candle_core::Error },

    #[snafu(display("Failed to make tensor contiguous"))]
    MakeContiguous { source: candle_core::Error },
}

/// Get indices of non-zero elements in a tensor.
pub trait Nonzero {
    fn nonzero(self) -> Result<Tensor, NonzeroError>;
}

impl Nonzero for &Tensor {
    /// Get indices of non-zero elements in a tensor.
    ///
    /// Returns a tensor with shape *(n_nonzero, n_dim)* where each row
    /// contains the index of a non-zero element. If the input is a 1D tensor,
    /// a tensor with shape *(n_nonzero)* is returned.
    fn nonzero(self) -> Result<Tensor, NonzeroError> {
        let contiguous = self.contiguous().context(MakeContiguousSnafu)?;
        contiguous.apply_op1(NonzeroOp).context(ApplyOpSnafu)
    }
}

/// Non-zero indices op.
///
/// All op implementations will fail when the input tensor is not contiguous.
/// The public `Nonzero::nonzero` method takes care of the conversion.
struct NonzeroOp;

impl NonzeroOp {
    fn cpu_n_nonzero<T: WithDType>(
        data: &[T],
        layout: &Layout,
    ) -> Result<usize, candle_core::Error> {
        if !layout.is_contiguous() {
            bail!("Nonzero op requires contiguous layout")
        }

        let (start_offset, end_offset) = layout.contiguous_offsets().unwrap();

        let n_nonzero = data[start_offset..end_offset]
            .iter()
            .filter(|x| !x.is_zero())
            .count();

        Ok(n_nonzero)
    }

    fn cpu_fwd_impl<T: WithDType>(
        &self,
        data: &[T],
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape), candle_core::Error> {
        if !layout.is_contiguous() {
            bail!("Nonzero op requires contiguous layout")
        }

        let n_nonzero = Self::cpu_n_nonzero(data, layout)?;
        let n_dim = layout.dims().len();

        let mut dst = Vec::with_capacity(n_nonzero * n_dim);
        let dst_to_set = dst.spare_capacity_mut();
        let dst_to_set = unsafe { std::mem::transmute::<_, &mut [i64]>(dst_to_set) };

        let (start_offset, end_offset) = layout.contiguous_offsets().unwrap();

        let data = &data[start_offset..end_offset];
        let mut offset = 0;
        for (idx, val) in data.iter().enumerate() {
            if !val.is_zero() {
                let mut div = 1;
                for (dim, dim_size) in layout.dims().iter().enumerate().rev() {
                    dst_to_set[offset + dim] = ((idx / div) % dim_size) as i64;
                    div *= dim_size;
                }
                offset += n_dim;
            }

            // Don't process any remaining zero entries when we know
            // that we are done.
            if offset == n_nonzero * n_dim {
                break;
            }
        }
        unsafe { dst.set_len(n_nonzero * n_dim) }

        let shape = if layout.dims().len() == 1 {
            Shape::from_dims(&[n_nonzero])
        } else {
            Shape::from_dims(&[n_nonzero, n_dim])
        };

        Ok((CpuStorage::I64(dst), shape))
    }
}

impl CustomOp1 for NonzeroOp {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape), candle_core::Error> {
        match storage {
            CpuStorage::U8(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::U32(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::I64(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::BF16(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F16(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F32(data) => self.cpu_fwd_impl(data, layout),
            CpuStorage::F64(data) => self.cpu_fwd_impl(data, layout),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape), candle_core::Error> {
        if !layout.is_contiguous() {
            bail!("Nonzero op requires contiguous layout")
        }

        oxidized_cuda_kernels::nonzero(storage, layout)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Tensor};
    use ndarray::{array, ArrayD};
    use snafu::{report, ResultExt, Whatever};

    use crate::{
        ops::nonzero::Nonzero,
        util::{device::tests::test_devices, tests::IntoArrayD},
    };

    #[test]
    #[report]
    fn nonzero_produces_correct_output() -> Result<(), Whatever> {
        for device in test_devices() {
            // We do not have a Metal kernel yet.
            if device.is_metal() {
                continue;
            }

            for dtype in &[DType::F16, DType::BF16, DType::F32, DType::I64] {
                // Check 1D tensor.
                let t = Tensor::from_slice(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0], &[6], &device)
                    .unwrap()
                    .to_dtype(dtype.clone())
                    .unwrap();
                let r: ArrayD<i64> = t
                    .nonzero()
                    .whatever_context("Cannot compute non-zero indices")?
                    .into_arrayd()
                    .whatever_context("Cannot convert indices to array")?;
                assert_eq!(r, array![0, 2, 4].into_arrayd().unwrap());

                // Check 2D tensor.
                let t = Tensor::eye(3, dtype.clone(), &device).unwrap();
                let r: ArrayD<i64> = t.nonzero().unwrap().into_arrayd().unwrap();
                assert_eq!(r, array![[0i64, 0], [1, 1], [2, 2]].into_arrayd().unwrap());

                // Check non-contiguous tensor.
                let t = t.t().unwrap();
                let r: ArrayD<i64> = t
                    .nonzero()
                    .whatever_context("Cannot compute non-zero indices")?
                    .into_arrayd()
                    .whatever_context("Cannot convert indices to array")?;
                assert_eq!(r, array![[0i64, 0], [1, 1], [2, 2]].into_arrayd().unwrap());

                // Check 3D tensorr.
                let t = Tensor::from_slice(
                    &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
                    &[2, 2, 2],
                    &device,
                )
                .unwrap()
                .to_dtype(dtype.clone())
                .unwrap();
                let r: ArrayD<i64> = t
                    .nonzero()
                    .whatever_context("Cannot compute non-zero indices")?
                    .into_arrayd()
                    .whatever_context("Cannot convert indices to array")?;
                assert_eq!(
                    r,
                    array![[0i64, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
                        .into_arrayd()
                        .unwrap()
                );
            }
        }

        Ok(())
    }
}
