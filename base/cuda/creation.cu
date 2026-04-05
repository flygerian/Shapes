#include "../result/result.h"
#include "../shapes.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__global__ static void fillKernel(T *dest, tensor_size_t n, T value) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = value;
}

template <typename T>
static Result launchFillKernel(void *dest, tensor_size_t n, T value) {
  int threadsPerBlock = 256;
  int blocks = (int)((n + (tensor_size_t)threadsPerBlock - 1) /
                     (tensor_size_t)threadsPerBlock);
  fillKernel<<<blocks, threadsPerBlock>>>((T *)dest, n, value);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

__global__ static void arangeKernel(f32 *dest, tensor_size_t n, f32 start,
                                    f32 step) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = start + (f32)idx * step;
}

template <typename IndexType>
__device__ static i64 loadOneHotIndex(const IndexType *indices, size_t idx) {
  return (i64)indices[idx];
}

template <>
__device__ i64 loadOneHotIndex<f32>(const f32 *indices, size_t idx) {
  return (i64)indices[idx];
}

template <>
__device__ i64 loadOneHotIndex<f64>(const f64 *indices, size_t idx) {
  return (i64)indices[idx];
}

template <typename IndexType>
__global__ static void oneHotKernel(const IndexType *indices, tensor_size_t n,
                                    dim_t numClasses, f32 *dest) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  i64 classIdx = loadOneHotIndex(indices, idx);
  if (classIdx < 0 || classIdx >= (i64)numClasses) {
    return;
  }

  dest[idx * (tensor_size_t)numClasses + (tensor_size_t)classIdx] = 1.0f;
}

template <typename IndexType>
static Result launchOneHotKernel(const void *indices, tensor_size_t n,
                                 dim_t numClasses, void *dest) {
  int threadsPerBlock = 256;
  int blocks = (int)((n + (tensor_size_t)threadsPerBlock - 1) /
                     (tensor_size_t)threadsPerBlock);
  oneHotKernel<<<blocks, threadsPerBlock>>>((const IndexType *)indices, n,
                                            numClasses, (f32 *)dest);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaFillTensor(Context *ctx, Dtype dtype, void *dest,
                                    tensor_size_t n, Value value) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case BOOL:
    return launchFillKernel<bool>(dest, n, value.as.boolean);
  case U8:
    return launchFillKernel<u8>(dest, n, value.as.u8);
  case U16:
    return launchFillKernel<u16>(dest, n, value.as.u16);
  case U32:
    return launchFillKernel<u32>(dest, n, value.as.u32);
  case U64:
    return launchFillKernel<u64>(dest, n, value.as.u64);
  case I8:
    return launchFillKernel<i8>(dest, n, value.as.i8);
  case I16:
    return launchFillKernel<i16>(dest, n, value.as.i16);
  case I32:
    return launchFillKernel<i32>(dest, n, value.as.i32);
  case I64:
    return launchFillKernel<i64>(dest, n, value.as.i64);
  case F16:
  case F32:
    return launchFillKernel<f32>(dest, n, value.as.f32);
  case F64:
    return launchFillKernel<f64>(dest, n, value.as.f64);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaArange(Context *ctx, f32 start, f32 step, void *dest,
                                tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  int threadsPerBlock = 256;
  int blocks = (int)((n + (tensor_size_t)threadsPerBlock - 1) /
                     (tensor_size_t)threadsPerBlock);
  arangeKernel<<<blocks, threadsPerBlock>>>((f32 *)dest, n, start, step);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaOneHot(Context *ctx, Dtype indexDtype,
                                const void *indices, tensor_size_t n,
                                dim_t numClasses, void *dest) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (indexDtype) {
  case BOOL:
    return launchOneHotKernel<bool>(indices, n, numClasses, dest);
  case U8:
    return launchOneHotKernel<u8>(indices, n, numClasses, dest);
  case U16:
    return launchOneHotKernel<u16>(indices, n, numClasses, dest);
  case U32:
    return launchOneHotKernel<u32>(indices, n, numClasses, dest);
  case U64:
    return launchOneHotKernel<u64>(indices, n, numClasses, dest);
  case I8:
    return launchOneHotKernel<i8>(indices, n, numClasses, dest);
  case I16:
    return launchOneHotKernel<i16>(indices, n, numClasses, dest);
  case I32:
    return launchOneHotKernel<i32>(indices, n, numClasses, dest);
  case I64:
    return launchOneHotKernel<i64>(indices, n, numClasses, dest);
  case F16:
  case F32:
    return launchOneHotKernel<f32>(indices, n, numClasses, dest);
  case F64:
    return launchOneHotKernel<f64>(indices, n, numClasses, dest);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
