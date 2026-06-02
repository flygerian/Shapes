#include "olib.h"
#include "result.h"
#include "types.h"
#include <cuda_runtime.h>

template <typename T>
__global__ static void sgdKernel(T *param, const T *grad, tensor_size_t n,
                                 T learningRate) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  param[idx] -= grad[idx] * learningRate;
}

static Result finishSgdLaunch() {
  cudaError_t sync_error = cudaDeviceSynchronize();
  PANIC_WITH_MSG_IF(sync_error != cudaSuccess, cudaGetErrorString(sync_error));

  return OK;
}

template <typename T>
static Result launchSgd(void *param, const void *grad, tensor_size_t n,
                        f32 learningRate) {
  if (n == 0) {
    return OK;
  }

  int threadsPerBlock = 256;
  int blocks = (int)((n + (tensor_size_t)threadsPerBlock - 1) /
                     (tensor_size_t)threadsPerBlock);
  sgdKernel<<<blocks, threadsPerBlock>>>((T *)param, (const T *)grad, n,
                                         (T)learningRate);
  return finishSgdLaunch();
}

extern "C" Result runCudaSgd(Context *ctx, Dtype dtype, void *param,
                             const void *grad, tensor_size_t n,
                             f32 learningRate) {
  if (ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchSgd<f32>(param, grad, n, learningRate);
  case F64:
    return launchSgd<f64>(param, grad, n, learningRate);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
