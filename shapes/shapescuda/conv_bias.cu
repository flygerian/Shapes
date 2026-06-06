#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__global__ static void convBiasAddKernel(T *output, const T *bias, size_t n,
                                         size_t channels) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  output[idx] += bias[idx % (size_t)channels];
}

template <typename T>
__global__ static void convBiasBackwardKernel(const T *outputGrad, T *dBias,
                                              size_t n, size_t channels) {
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
static Result launchConvBiasAdd(void *output, const void *bias, size_t n,
                                size_t channels) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  convBiasAddKernel<<<blocks, threadsPerBlock>>>((T *)output, (const T *)bias,
                                                 n, channels);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchConvBiasBackward(const void *outputGrad, void *dBias,
                                     size_t n, size_t channels) {
  int threadsPerBlock = 256;
  int blocks = (int)(((size_t)channels + (size_t)threadsPerBlock - 1) /
                     (size_t)threadsPerBlock);
  convBiasBackwardKernel<<<blocks, threadsPerBlock>>>((const T *)outputGrad,
                                                      (T *)dBias, n, channels);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

extern "C" Result shapescuda_ConvBiasAdd(shapes_Dtype dtype, void *output,
                                         const void *bias, size_t numValues,
                                         size_t channels) {
  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchConvBiasAdd<f32>(output, bias, numValues, channels);
  case F64:
    return launchConvBiasAdd<f64>(output, bias, numValues, channels);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result shapescuda_ConvBiasBackward(shapes_Dtype dtype,
                                              const void *outputGrad,
                                              void *dBias, size_t numValues,
                                              size_t channels) {
  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchConvBiasBackward<f32>(outputGrad, dBias, numValues, channels);
  case F64:
    return launchConvBiasBackward<f64>(outputGrad, dBias, numValues, channels);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
