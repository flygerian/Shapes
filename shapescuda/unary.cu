#include "olib.h"
#include "result.h"
#include "types.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool shouldLogCudaRelu(void) {
  const char *value = getenv("SHAPES_LOG_RELU");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void logCudaReluError(const char *phase, cudaError_t error, size_t n,
                             int blocks, int threadsPerBlock) {
  if (!shouldLogCudaRelu()) {
    return;
  }

  fprintf(stderr,
          "[Relu][CUDA] phase=%s n=%llu blocks=%d threads=%d cudaError=%s "
          "detail=%s code=%d\n",
          phase, (unsigned long long)n, blocks, threadsPerBlock,
          cudaGetErrorName(error), cudaGetErrorString(error), (int)error);
}

template <typename T>
__device__ static T applyUnaryOp(T value, UnaryOpType opType, float param) {
  switch (opType) {
  case UNARY_OP_POW:
    return (T)pow((double)value, (double)param);
  case UNARY_OP_TANH:
    return (T)tanh((double)value);
  case UNARY_OP_RELU:
    return value > (T)0 ? value : (T)0;
  case UNARY_OP_NEGATE:
    return -value;
  case UNARY_OP_EXP:
    return (T)exp((double)value);
  case UNARY_OP_LOG:
    return (T)log((double)value);
  case UNARY_OP_ABS:
    return value < (T)0 ? -value : value;
  case UNARY_OP_SQRT:
    return (T)sqrt((double)value);
  default:
    return value;
  }
}

template <>
__device__ bool applyUnaryOp<bool>(bool value, UnaryOpType opType,
                                   float param) {
  (void)opType;
  (void)param;
  return value;
}

template <typename T>
__global__ static void unaryOpKernel(const T *src, T *dest, size_t n,
                                     UnaryOpType opType, float param) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = applyUnaryOp(src[idx], opType, param);
}

template <typename T>
__global__ static void reluBackwardKernel(const T *output, const T *gradOut,
                                          T *dest, size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = output[idx] > (T)0 ? gradOut[idx] : (T)0;
}

template <typename T>
__global__ static void reluBackwardAccumulateKernel(const T *output,
                                                    const T *gradOut, T *dest,
                                                    size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] += output[idx] > (T)0 ? gradOut[idx] : (T)0;
}

template <typename T>
static Result launchUnaryOpKernel(const void *src, void *dest, size_t n,
                                  UnaryOpType opType, float param) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  cudaError_t pendingError = cudaPeekAtLastError();
  if (opType == UNARY_OP_RELU && pendingError != cudaSuccess) {
    logCudaReluError("prelaunch_pending_error", pendingError, n, blocks,
                     threadsPerBlock);
  }

  unaryOpKernel<<<blocks, threadsPerBlock>>>((const T *)src, (T *)dest, n,
                                             opType, param);

  cudaError_t launchError = cudaGetLastError();
  if (opType == UNARY_OP_RELU && launchError != cudaSuccess) {
    logCudaReluError("launch_error", launchError, n, blocks, threadsPerBlock);
  }

  if (launchError == cudaErrorMemoryAllocation) {
    return ERR_OUT_OF_MEMORY;
  }
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  if (opType == UNARY_OP_RELU && shouldLogCudaRelu()) {
    logCudaReluError("launch_ok", launchError, n, blocks, threadsPerBlock);
  }

  return OK;
}

template <typename T>
static Result launchReluBackwardKernel(const void *output, const void *gradOut,
                                       void *dest, size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  reluBackwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)output, (const T *)gradOut, (T *)dest, n);

  cudaError_t launchError = cudaGetLastError();
  if (launchError == cudaErrorMemoryAllocation) {
    return ERR_OUT_OF_MEMORY;
  }
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchReluBackwardAccumulateKernel(const void *output,
                                                 const void *gradOut,
                                                 void *dest, size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  reluBackwardAccumulateKernel<<<blocks, threadsPerBlock>>>(
      (const T *)output, (const T *)gradOut, (T *)dest, n);

  cudaError_t launchError = cudaGetLastError();
  if (launchError == cudaErrorMemoryAllocation) {
    return ERR_OUT_OF_MEMORY;
  }
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

extern "C" Result runCudaUnaryOp(Context *ctx, Dtype dtype, UnaryOpType opType,
                                 const void *src, void *dest, tensor_size_t n,
                                 f32 param) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case I8:
    return launchUnaryOpKernel<i8>(src, dest, n, opType, param);
  case I16:
    return launchUnaryOpKernel<i16>(src, dest, n, opType, param);
  case I32:
    return launchUnaryOpKernel<i32>(src, dest, n, opType, param);
  case I64:
    return launchUnaryOpKernel<i64>(src, dest, n, opType, param);
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchUnaryOpKernel<f32>(src, dest, n, opType, param);
  case F64:
    return launchUnaryOpKernel<f64>(src, dest, n, opType, param);
  default:
    return ERR_NO_OP;
  }
}

extern "C" Result runCudaReluBackward(Context *ctx, Dtype dtype,
                                      const void *output, const void *gradOut,
                                      void *dest, tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchReluBackwardKernel<f32>(output, gradOut, dest, n);
  case F64:
    return launchReluBackwardKernel<f64>(output, gradOut, dest, n);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaReluBackwardAccumulate(Context *ctx, Dtype dtype,
                                                const void *output,
                                                const void *gradOut, void *dest,
                                                tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchReluBackwardAccumulateKernel<f32>(output, gradOut, dest, n);
  case F64:
    return launchReluBackwardAccumulateKernel<f64>(output, gradOut, dest, n);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
