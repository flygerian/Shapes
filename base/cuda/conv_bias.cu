#include "../common.h"
#include "../result/result.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__global__ static void convBiasAddKernel(T *output, const T *bias, size_t n,
                                         dim_t channels) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  output[idx] += bias[idx % (size_t)channels];
}

template <typename T>
__global__ static void convBiasBackwardKernel(const T *outputGrad, T *dBias,
                                              size_t n, dim_t channels) {
  size_t channel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= (size_t)channels) {
    return;
  }

  T acc = (T)0;
  for (size_t idx = channel; idx < n; idx += (size_t)channels) {
    acc += outputGrad[idx];
  }
  dBias[channel] = acc;
}

template <typename T>
static Result launchConvBiasAdd(Context *ctx, void *output, const void *bias,
                                size_t n, dim_t channels) {
  if (ctx == NULL || channels == 0) {
    return ERR_NO_OP;
  }

  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  convBiasAddKernel<<<blocks, threadsPerBlock>>>((T *)output, (const T *)bias,
                                                 n, channels);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename T>
static Result launchConvBiasBackward(Context *ctx, const void *outputGrad,
                                     void *dBias, size_t n, dim_t channels) {
  if (ctx == NULL || channels == 0) {
    return ERR_NO_OP;
  }

  int threadsPerBlock = 256;
  int blocks = (int)(((size_t)channels + (size_t)threadsPerBlock - 1) /
                     (size_t)threadsPerBlock);
  convBiasBackwardKernel<<<blocks, threadsPerBlock>>>((const T *)outputGrad,
                                                      (T *)dBias, n, channels);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaConvBiasAdd(Context *ctx, Dtype dtype, void *output,
                                     const void *bias, tensor_size_t numValues,
                                     dim_t channels) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
  case F32:
    return launchConvBiasAdd<f32>(ctx, output, bias, numValues, channels);
  case F64:
    return launchConvBiasAdd<f64>(ctx, output, bias, numValues, channels);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaConvBiasBackward(Context *ctx, Dtype dtype,
                                          const void *outputGrad, void *dBias,
                                          tensor_size_t numValues,
                                          dim_t channels) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
  case F32:
    return launchConvBiasBackward<f32>(ctx, outputGrad, dBias, numValues,
                                       channels);
  case F64:
    return launchConvBiasBackward<f64>(ctx, outputGrad, dBias, numValues,
                                       channels);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
