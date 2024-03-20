use cudarc::driver::sys::{CUdeviceptr, CUstream};
use half::{bf16, f16};

macro_rules! nonzero {
    ($name:ident, $type:ty) => {
        extern "C" {
            pub(crate) fn $name(
                num_elems: usize,
                num_dims: usize,
                info: *const usize,
                input: *const $type,
                out: *mut CUdeviceptr,
                num_nonzeros: *mut usize,
                stream: CUstream,
            );
        }
    };
}

nonzero!(nonzero_bf16, bf16);
nonzero!(nonzero_f16, f16);
nonzero!(nonzero_f32, f32);
nonzero!(nonzero_f64, f64);
nonzero!(nonzero_u8, u8);
nonzero!(nonzero_u32, u32);
nonzero!(nonzero_i64, i64);
