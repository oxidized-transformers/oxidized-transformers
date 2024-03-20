// This kernel is adapted from the PyTorch nonzero kernel:
// https://github.com/pytorch/pytorch/blob/81683380633f3c79cbf0debc0d491612b152f202/aten/src/ATen/native/cuda/Nonzero.cu
//
// Copyrights and license:
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include <cassert>
#include <cub/cub.cuh>

#include "step_output_iterator.cuh"

#if defined(USE_ROCM)
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

template <typename T>
struct NonZeroOp
{
  __host__ __device__ __forceinline__ bool operator()(const T &a) const
  {
    return (a != T(0));
  }
};

// TODO: actually support int64_t index_t
template <typename index_t>
struct TensorDims
{
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
__global__ void write_indices(
    int64_t *inp,
    TensorDims<index_t> dims,
    int ndim,
    index_t n)
{
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
  {
    index_t div = 1;
    int64_t idx_flat = inp[index * ndim];
#pragma unroll
    for (int dim = MAX_DIMS; dim >= 0; dim--)
    {
      if (dim > ndim - 1)
        continue;
      auto dim_size = dims.sizes[dim];
      inp[index * ndim + dim] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  }
}

template <typename scalar_t>
void nonzero_cuda_impl(size_t const num_elems, size_t const num_dims, size_t const *info,
                       scalar_t const *input, int64_t **out, size_t *num_nonzeros, cudaStream_t stream = 0)
{
  int N = num_elems;
  // Compute number of nonzero elements.
  size_t temp_storage_bytes = 0;
  int *device_num_nonzeros = nullptr;
  cudaMalloc(&device_num_nonzeros, sizeof(int));
  cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, const scalar_t *> nonzero_iter(input, NonZeroOp<scalar_t>());
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, nonzero_iter, device_num_nonzeros, N, stream);
  void *temp_storage = NULL;
  cudaMalloc(&temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, nonzero_iter, device_num_nonzeros, N, stream);
  cudaFree(temp_storage);

  // Copy the number of nonzero elements to host memory and synchronize.
  cudaMemcpyAsync(num_nonzeros, device_num_nonzeros, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Allocate the output, for each non-zero element, we store its index in
  // each dimension.
  cudaMalloc(out, num_dims * *num_nonzeros * sizeof(int64_t));

  // Scalars are expected to produce output of size (1,0), so we can't write to it.
  if (num_dims > 0)
  {
    // Get the indices of the non-zero elements, treating the input as an 1D
    // array. The i-th index is placed at out[i*num_dims], we will later expand
    // them to xD array indices in out[i*num_dims:(i+1)*num_dims].
    cub::CountingInputIterator<int64_t> counting_iter(0);
    StepOutputIterator<int64_t> dims_step_output(*out, num_dims);
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_iter, nonzero_iter,
                               dims_step_output, device_num_nonzeros, N, stream);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Flagged(temp_storage, temp_storage_bytes, counting_iter, nonzero_iter,
                               dims_step_output, device_num_nonzeros, N, stream);
    cudaFree(temp_storage);

    // Expand the 1D indices to xD array indices if necessary.
    if (*num_nonzeros > 0 && num_dims > 1)
    {
      TensorDims<size_t> dims;
      for (int i = 0; i < num_dims; i++)
      {
        dims.sizes[i] = info[i];
      }
      const int nthreads = 256;
      const int nblocks = (*num_nonzeros + nthreads - 1) / nthreads;
      write_indices<<<nblocks, nthreads, 0, stream>>>(*out, dims, num_dims, *num_nonzeros);
      // TODO: verify cudaGetLastError result, or do we do this in Rust-land?
    }
  }

  cudaFree(device_num_nonzeros);
}

template <typename scalar_t>
void nonzero_cuda(size_t const num_elems, size_t const num_dims, size_t const *info,
                  scalar_t const *input, int64_t **out, size_t *num_nonzeros, cudaStream_t stream = 0)
{
  assert(num_elems < std::numeric_limits<int>::max());
  assert(num_dims <= MAX_DIMS);
  nonzero_cuda_impl<scalar_t>(num_elems, num_dims, info, input, out, num_nonzeros, stream);
}

#define NONZERO_KERNEL(FN_NAME, TYPENAME)                                                                  \
  extern "C" void FN_NAME(size_t const num_elems, size_t const num_dims, size_t const *info,               \
                          TYPENAME const *input, int64_t **out, size_t *num_nonzeros, cudaStream_t stream) \
  {                                                                                                        \
    nonzero_cuda<TYPENAME>(num_elems, num_dims, info, input, out, num_nonzeros, stream);                   \
  }

NONZERO_KERNEL(nonzero_bf16, __nv_bfloat16)
NONZERO_KERNEL(nonzero_f16, __half)
NONZERO_KERNEL(nonzero_f32, float)
NONZERO_KERNEL(nonzero_f64, double)
NONZERO_KERNEL(nonzero_u8, uint8_t)
NONZERO_KERNEL(nonzero_u32, uint32_t)
NONZERO_KERNEL(nonzero_i64, int64_t)
