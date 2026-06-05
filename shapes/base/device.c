#include "shapes.h"
#ifdef SHAPES_HAS_CUDA 
  #include "shapescuda.h"
  #include <cublas_v2.h>
  #include "cuda_runtime.h"
#endif
#include <string.h>


void attachCudaDevice(shapes_Context *ctx) {
  #ifdef SHAPES_HAS_CUDA 
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
      Device *device = olib_Allocate(ctx->memory, sizeof(Device));
      *device = (Device){.id = "cuda:0", .type = CUDA};
      ctx->handle = handle;
      ctx->device = device;
      ctx->cudaMemory = shapescuda_Make_Memory(ctx->memory);
    }
  }
  #endif
}

void attachHostDevice(shapes_Context *ctx) {
  Device *device = olib_Allocate(ctx->memory, sizeof(Device));
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

  #ifdef SHAPES_HAS_CUDA 
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
  #endif

  return ERR_NO_OP;
}
