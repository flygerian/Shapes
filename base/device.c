
#include "common.h"
#include "shapes.h"
#include "utils_lib/cuda_memory.h"
#include <string.h>

static DeviceType getDeviceTypeForPointer(const void *ptr) {
  if (ptr == NULL) {
    return CPU;
  }

  struct cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
  if (result != cudaSuccess) {
    cudaGetLastError();
    return CPU;
  }

#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice ? CUDA : CPU;
#else
  return attributes.memoryType == cudaMemoryTypeDevice ? CUDA : CPU;
#endif
}

void attachCudaDevice(Context *ctx) {
  int deviceCount = 0;
  cudaError_t countResult = cudaGetDeviceCount(&deviceCount);

  if (countResult == cudaSuccess && deviceCount > 0) {
    cudaError_t setDeviceResult = cudaSetDevice(0);
    cudaError_t runtimeInitResult = cudaFree(NULL);
    cublasHandle_t handle;
    cublasStatus_t handleResult = CUBLAS_STATUS_NOT_INITIALIZED;

    if (setDeviceResult == cudaSuccess && runtimeInitResult == cudaSuccess) {
      handleResult = cublasCreate(&handle);
    }

    if (handleResult == CUBLAS_STATUS_SUCCESS) {
      Device *device = allocate(ctx->memory, sizeof(Device));
      *device = (Device){.id = "cuda:0", .type = CUDA};
      ctx->handle = handle;
      ctx->device = device;
      ctx->cudaMemory = Make_CudaMemory(ctx->memory);
    }
  }
}

void attachHostDevice(Context *ctx) {
  Device *device = allocate(ctx->memory, sizeof(Device));
  *device = (Device){.id = "host", .type = CPU};
  ctx->device = device;
}

Result shapes_CopyBetweenDevices(DeviceType srcType, DeviceType destType, void *restrict srcPtr, void *restrict destPtr, size_t size) {
  if (srcPtr == NULL || destPtr == NULL) {
    return ERR_NULL_PTR;
  }

  if (srcType == CPU && destType == CPU) {
    memcpy(destPtr, srcPtr, size);
    return OK;
  }

  if (srcType == CPU && destType == CUDA) {
    cudaError_t err = cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyHostToDevice);
    PANIC_WITH_MSG_IF(err != cudaSuccess, cudaGetErrorString(err));
    return OK;
  }

  if (srcType == CUDA && destType == CPU) {
    cudaError_t err = cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToHost);
    PANIC_WITH_MSG_IF(err != cudaSuccess, cudaGetErrorString(err));
    return OK;
  }

  if (srcType == CUDA && destType == CUDA) {
    cudaError_t err = cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice);
    PANIC_WITH_MSG_IF(err != cudaSuccess, cudaGetErrorString(err));
    return OK;
  }

  return ERR_NO_OP;
}
