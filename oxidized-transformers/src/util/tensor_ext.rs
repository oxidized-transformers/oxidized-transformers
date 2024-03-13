/// Tensor extension traits.
use candle_core::{DType, Tensor};
use half::{bf16, f16};

/// Get a tensor with the data types minimum value.
pub trait MinLike: Sized {
    /// Get a new tensor with the data type's minimum value.
    ///
    /// The tensor has the same shape as `self`.
    fn min_like(&self) -> Result<Self, candle_core::Error>;
}

impl MinLike for Tensor {
    fn min_like(&self) -> Result<Self, candle_core::Error> {
        match self.dtype() {
            DType::BF16 => Tensor::try_from(bf16::MIN),
            DType::F16 => Tensor::try_from(f16::MIN),
            DType::F32 => Tensor::try_from(f32::MIN),
            DType::F64 => Tensor::try_from(f64::MIN),
            DType::U8 => Tensor::try_from(u8::MIN),
            DType::U32 => Tensor::try_from(u32::MIN),
            DType::I64 => Tensor::try_from(i64::MIN),
        }
        .and_then(|scalar| scalar.broadcast_as(self.shape()))
        .and_then(|tensor| tensor.to_device(self.device()))
    }
}
