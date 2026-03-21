#include "../common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

template <typename T>
__device__ static T applyUnaryOp(T value, UnaryOpType opType, float param) {
  switch (opType) {
    case UNARY_OP_POW: return (T)pow((double)value, (double)param);
    case UNARY_OP_TANH: return (T)tanh((double)value);
    case UNARY_OP_RELU: return value > (T)0 ? value : (T)0;
    case UNARY_OP_NEGATE: return -value;
    case UNARY_OP_EXP: return (T)exp((double)value);
    case UNARY_OP_LOG: return (T)log((double)value);
    case UNARY_OP_ABS: return value < (T)0 ? -value : value;
    default: return value;
  }
}

template <>
__device__ bool applyUnaryOp<bool>(bool value, UnaryOpType opType, float param) {
  (void)opType;
  (void)param;
  return value;
}

template <typename T>
__global__ static void unaryOpKernel(const T *src, T *dest, size_t n, UnaryOpType opType,
                                     float param) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = applyUnaryOp(src[idx], opType, param);
}

template <typename T>
static Result launchUnaryOpKernel(const void *src, void *dest, size_t n, UnaryOpType opType,
                                  float param) {
  int threadsPerBlock = 256;
  int blocks = (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  unaryOpKernel<<<blocks, threadsPerBlock>>>((const T *)src, (T *)dest, n, opType, param);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaUnaryOp(Context *ctx, Dtype dtype, UnaryOpType opType, const void *src,
                                 void *dest, tensor_size_t n, f32 param) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
    case I8: return launchUnaryOpKernel<i8>(src, dest, n, opType, param);
    case I16: return launchUnaryOpKernel<i16>(src, dest, n, opType, param);
    case I32: return launchUnaryOpKernel<i32>(src, dest, n, opType, param);
    case I64: return launchUnaryOpKernel<i64>(src, dest, n, opType, param);
    case F16:
    case F32: return launchUnaryOpKernel<f32>(src, dest, n, opType, param);
    case F64: return launchUnaryOpKernel<f64>(src, dest, n, opType, param);
    default: return ERR_NO_OP;
  }
}
