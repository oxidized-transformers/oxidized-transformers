use candle_core::{backend::BackendStorage, bail, CudaStorage, DType, Layout, Shape};
use cudarc::driver::{sys::CUdeviceptr, DevicePtr};
use half::{bf16, f16};

use crate::ffi;

/// Returns indices of nonzero elements in the given storage.
///
/// Fails if the data is not contiguous (C-order).
pub fn nonzero(
    storage: &CudaStorage,
    layout: &Layout,
) -> Result<(CudaStorage, Shape), candle_core::Error> {
    if !layout.is_contiguous() {
        bail!("nonzero kernel requires contiguous layout")
    }

    let n_elem = layout.shape().elem_count();
    let n_dim = layout.shape().dims().len();
    let shape = layout.shape().dims();
    let stream = storage.device.cu_stream();

    // Kernel outputs
    let mut n_nonzero = 0;
    let mut indices: CUdeviceptr = 0;

    let offset = layout.contiguous_offsets().unwrap().0 as CUdeviceptr;

    unsafe {
        match storage.dtype() {
            DType::BF16 => ffi::nonzero_bf16(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<bf16>().unwrap().device_ptr() + offset) as *const bf16,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::F16 => ffi::nonzero_f16(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<f16>().unwrap().device_ptr() + offset) as *const f16,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::F32 => ffi::nonzero_f32(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<f32>().unwrap().device_ptr() + offset) as *const f32,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::F64 => ffi::nonzero_f64(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<f64>().unwrap().device_ptr() + offset) as *const f64,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::U8 => ffi::nonzero_u8(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<u8>().unwrap().device_ptr() + offset) as *const u8,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::U32 => ffi::nonzero_u32(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<u32>().unwrap().device_ptr() + offset) as *const u32,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
            DType::I64 => ffi::nonzero_i64(
                n_elem,
                n_dim,
                shape.as_ptr(),
                (storage.as_cuda_slice::<i64>().unwrap().device_ptr() + offset) as *const i64,
                &mut indices,
                &mut n_nonzero,
                *stream,
            ),
        }
    };

    let output_slice = unsafe {
        storage
            .device()
            .upgrade_device_ptr::<i64>(indices, n_nonzero * n_dim)
    };

    let indices_storage = CudaStorage::wrap_cuda_slice(output_slice, storage.device().clone());
    let indices_shape = if n_dim == 1 {
        Shape::from_dims(&[n_nonzero])
    } else {
        Shape::from_dims(&[n_nonzero, n_dim])
    };

    Ok((indices_storage, indices_shape))
}
