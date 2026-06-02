#include "olib.h"
#include "result.h"
#include "types.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

template <typename T>
__global__ static void reduceDimKernel(const T *src, T *dest,
                                       size_t numBeforeDim, size_t numAfterDim,
                                       size_t reduce, ReductionOpType opType) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t resultSize = numBeforeDim * numAfterDim;
  if (idx >= resultSize) {
    return;
  }

  size_t outer = idx / numAfterDim;
  size_t inner = idx % numAfterDim;
  size_t base = outer * reduce * numAfterDim + inner;

  switch (opType) {
  case REDUCTION_OP_SUM: {
    T acc = (T)0;
    for (size_t r = 0; r < reduce; r++) {
      acc = acc + src[base + r * numAfterDim];
    }
    dest[idx] = acc;
    return;
  }
  case REDUCTION_OP_MEAN: {
    T acc = (T)0;
    for (size_t r = 0; r < reduce; r++) {
      acc = acc + src[base + r * numAfterDim];
    }
    dest[idx] = acc / (T)reduce;
    return;
  }
  case REDUCTION_OP_MAX: {
    T maxValue = src[base];
    for (size_t r = 1; r < reduce; r++) {
      T candidate = src[base + r * numAfterDim];
      if (candidate > maxValue) {
        maxValue = candidate;
      }
    }
    dest[idx] = maxValue;
    return;
  }
  default:
    return;
  }
}

template <typename T>
__global__ static void argmaxDimKernel(const T *src, i64 *dest,
                                       size_t numBeforeDim, size_t numAfterDim,
                                       size_t reduce) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t resultSize = numBeforeDim * numAfterDim;
  if (idx >= resultSize) {
    return;
  }

  size_t outer = idx / numAfterDim;
  size_t inner = idx % numAfterDim;
  size_t base = outer * reduce * numAfterDim + inner;

  T maxValue = src[base];
  i64 maxIndex = 0;
  for (size_t r = 1; r < reduce; r++) {
    T candidate = src[base + r * numAfterDim];
    if (candidate > maxValue) {
      maxValue = candidate;
      maxIndex = (i64)r;
    }
  }

  dest[idx] = maxIndex;
}

template <typename T>
__global__ static void reduceAllKernel(const T *src, T *dest, size_t n,
                                       ReductionOpType opType) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  switch (opType) {
  case REDUCTION_OP_SUM: {
    T acc = (T)0;
    for (size_t i = 0; i < n; i++) {
      acc = acc + src[i];
    }
    dest[0] = acc;
    return;
  }
  case REDUCTION_OP_MEAN: {
    T acc = (T)0;
    for (size_t i = 0; i < n; i++) {
      acc = acc + src[i];
    }
    dest[0] = acc / (T)n;
    return;
  }
  default:
    return;
  }
}

template <typename T>
__global__ static void stdAllKernel(const T *src, T *dest, size_t n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  double mean = 0.0;
  for (size_t i = 0; i < n; i++) {
    mean += (double)src[i];
  }
  mean /= (double)n;

  double deviationSquaredSum = 0.0;
  for (size_t i = 0; i < n; i++) {
    double centered = (double)src[i] - mean;
    deviationSquaredSum += centered * centered;
  }

  double variance = deviationSquaredSum / (double)(n - 1);
  dest[0] = (T)sqrt(variance);
}

template <typename T>
static Result launchReduceDimKernel(const void *src, void *dest,
                                    size_t numBeforeDim, size_t numAfterDim,
                                    size_t reduce, ReductionOpType opType) {
  int threadsPerBlock = 256;
  size_t resultSize = numBeforeDim * numAfterDim;
  int blocks = (int)((resultSize + (size_t)threadsPerBlock - 1) /
                     (size_t)threadsPerBlock);
  reduceDimKernel<<<blocks, threadsPerBlock>>>(
      (const T *)src, (T *)dest, numBeforeDim, numAfterDim, reduce, opType);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchArgmaxDimKernel(const void *src, void *dest,
                                    size_t numBeforeDim, size_t numAfterDim,
                                    size_t reduce) {
  int threadsPerBlock = 256;
  size_t resultSize = numBeforeDim * numAfterDim;
  int blocks = (int)((resultSize + (size_t)threadsPerBlock - 1) /
                     (size_t)threadsPerBlock);
  argmaxDimKernel<<<blocks, threadsPerBlock>>>(
      (const T *)src, (i64 *)dest, numBeforeDim, numAfterDim, reduce);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchReduceAllKernel(const void *src, void *dest, size_t n,
                                    ReductionOpType opType) {
  reduceAllKernel<<<1, 1>>>((const T *)src, (T *)dest, n, opType);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchStdKernel(const void *src, void *dest, size_t n) {
  stdAllKernel<<<1, 1>>>((const T *)src, (T *)dest, n);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

extern "C" Result runCudaReduceDim(Context *ctx, Dtype inputDtype,
                                   Dtype outputDtype, ReductionOpType opType,
                                   const void *src, void *dest,
                                   tensor_size_t numBeforeDim,
                                   tensor_size_t numAfterDim, dim_t reduce) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (opType) {
  case REDUCTION_OP_ARGMAX:
    switch (inputDtype) {
    case BOOL:
      return launchArgmaxDimKernel<bool>(src, dest, numBeforeDim, numAfterDim,
                                         reduce);
    case U8:
      return launchArgmaxDimKernel<u8>(src, dest, numBeforeDim, numAfterDim,
                                       reduce);
    case U16:
      return launchArgmaxDimKernel<u16>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case U32:
      return launchArgmaxDimKernel<u32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case U64:
      return launchArgmaxDimKernel<u64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case I8:
      return launchArgmaxDimKernel<i8>(src, dest, numBeforeDim, numAfterDim,
                                       reduce);
    case I16:
      return launchArgmaxDimKernel<i16>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case I32:
      return launchArgmaxDimKernel<i32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case I64:
      return launchArgmaxDimKernel<i64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case F16:
      return ERR_DTYPE_MISMATCH;
    case F32:
      return launchArgmaxDimKernel<f32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    case F64:
      return launchArgmaxDimKernel<f64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce);
    default:
      return ERR_NO_OP;
    }
  case REDUCTION_OP_SUM:
  case REDUCTION_OP_MEAN:
  case REDUCTION_OP_MAX:
    if (outputDtype != inputDtype) {
      return ERR_NO_OP;
    }
    switch (inputDtype) {
    case BOOL:
      return launchReduceDimKernel<bool>(src, dest, numBeforeDim, numAfterDim,
                                         reduce, opType);
    case U8:
      return launchReduceDimKernel<u8>(src, dest, numBeforeDim, numAfterDim,
                                       reduce, opType);
    case U16:
      return launchReduceDimKernel<u16>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case U32:
      return launchReduceDimKernel<u32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case U64:
      return launchReduceDimKernel<u64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case I8:
      return launchReduceDimKernel<i8>(src, dest, numBeforeDim, numAfterDim,
                                       reduce, opType);
    case I16:
      return launchReduceDimKernel<i16>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case I32:
      return launchReduceDimKernel<i32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case I64:
      return launchReduceDimKernel<i64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case F16:
      return ERR_DTYPE_MISMATCH;
    case F32:
      return launchReduceDimKernel<f32>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    case F64:
      return launchReduceDimKernel<f64>(src, dest, numBeforeDim, numAfterDim,
                                        reduce, opType);
    default:
      return ERR_NO_OP;
    }
  default:
    return ERR_NO_OP;
  }
}

extern "C" Result runCudaReduceAll(Context *ctx, Dtype dtype,
                                   ReductionOpType opType, const void *src,
                                   void *dest, tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (opType) {
  case REDUCTION_OP_SUM:
  case REDUCTION_OP_MEAN:
    switch (dtype) {
    case BOOL:
      return launchReduceAllKernel<bool>(src, dest, n, opType);
    case U8:
      return launchReduceAllKernel<u8>(src, dest, n, opType);
    case U16:
      return launchReduceAllKernel<u16>(src, dest, n, opType);
    case U32:
      return launchReduceAllKernel<u32>(src, dest, n, opType);
    case U64:
      return launchReduceAllKernel<u64>(src, dest, n, opType);
    case I8:
      return launchReduceAllKernel<i8>(src, dest, n, opType);
    case I16:
      return launchReduceAllKernel<i16>(src, dest, n, opType);
    case I32:
      return launchReduceAllKernel<i32>(src, dest, n, opType);
    case I64:
      return launchReduceAllKernel<i64>(src, dest, n, opType);
    case F16:
      return ERR_DTYPE_MISMATCH;
    case F32:
      return launchReduceAllKernel<f32>(src, dest, n, opType);
    case F64:
      return launchReduceAllKernel<f64>(src, dest, n, opType);
    default:
      return ERR_NO_OP;
    }
  default:
    return ERR_NO_OP;
  }
}

extern "C" Result runCudaStd(Context *ctx, Dtype dtype, const void *src,
                             void *dest, tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchStdKernel<f32>(src, dest, n);
  case F64:
    return launchStdKernel<f64>(src, dest, n);
  default:
    return ERR_NO_OP;
  }
}
