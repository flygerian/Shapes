#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "shapes.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

static DeviceType getDeviceTypeForContext(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return CPU;
  }

  return ctx->device->type;
}

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

Result copyBetweenContexts(Context *restrict srcCtx, Context *restrict destCtx,
                           void *restrict srcPtr, void *restrict destPtr, size_t size) {
  if (srcPtr == NULL || destPtr == NULL) {
    return ERR_NULL_PTR;
  }

  DeviceType srcType = srcCtx != NULL ? getDeviceTypeForContext(srcCtx) : getDeviceTypeForPointer(srcPtr);
  DeviceType destType =
      destCtx != NULL ? getDeviceTypeForContext(destCtx) : getDeviceTypeForPointer(destPtr);

  if (srcType == CPU && destType == CPU) {
    memcpy(destPtr, srcPtr, size);
    return OK;
  }

  if (srcType == CPU && destType == CUDA) {
    cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyHostToDevice);
    return OK;
  }

  if (srcType == CUDA && destType == CPU) {
    cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToHost);
    return OK;
  }

  if (srcType == CUDA && destType == CUDA) {
    cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice);
    return OK;
  }

  return ERR_NO_OP;
}

void *allocateOnCtx(Context *ctx, size_t size) {
  if (ctx == NULL) {
    return NULL;
  }

  if (ctx->device == NULL) {
    // CPU allocate by default
    return allocate(ctx->memory, size);
  }

  switch (ctx->device->type) {
    case CPU: return allocate(ctx->memory, size);
    case CUDA: {
      void *locationOnDestCtx = NULL;
      cudaError_t cudaResult = cudaMalloc(&locationOnDestCtx, size);
      if (cudaResult != cudaSuccess) {
        return NULL;
      }
      return locationOnDestCtx;
    }
  }

  return NULL;
}

void freeOnCtx(Context *ctx, void *ptr) {
  if (ctx == NULL || ptr == NULL) {
    return;
  }

  if (ctx->device == NULL) {
    freeAlloc(ctx->memory, ptr);
    return;
  }

  switch (ctx->device->type) {
    case CPU: freeAlloc(ctx->memory, ptr); return;
    case CUDA: cudaFree(ptr); return;
  }
}

Context InitializeContext(size_t arenaSize, size_t minBlockSize, bool withCuda) {
  Memory *memory = initializeArena(arenaSize, minBlockSize);
  Context ctx = {.memory = memory};

  if (withCuda) {
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
        Device *device = allocate(memory, sizeof(Device));
        device->id = "cuda:0";
        device->type = CUDA;
        ctx.handle = handle;
        ctx.device = device;
      }
    }
  }

  return ctx;
}

Context *CreateContext(size_t arenaSize, size_t minBlockSize, bool withCuda) {
  Context initialized = InitializeContext(arenaSize, minBlockSize, withCuda);
  Context *ctx = allocate(initialized.memory, sizeof(Context));
  *ctx = initialized;
  return ctx;
}

void DestroyContext(Context *ctx) {
  if (ctx->device != NULL && ctx->device->type == CUDA) {
    cublasDestroy(ctx->handle);
  }

  freeMemory(ctx->memory);
}

void FreeContext(Context *ctx) {
  if (ctx == NULL) {
    return;
  }

  DestroyContext(ctx);
}

Result Flush(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return OK;
  }

  switch (ctx->device->type) {
    case CPU: return OK;
    case CUDA: {
      cudaError_t syncResult = cudaDeviceSynchronize();
      if (syncResult != cudaSuccess) {
        return ERR_NO_OP;
      }
      return OK;
    }
    default: return OK;
  }
}

Result MoveTensors(Context *destCtx, u8 numTensors, ...) {
  if (destCtx == NULL) {
    return ERR_COPY_CTX_DEVICE_IS_NULL;
  }

  Tensor *tensors[numTensors];

  va_list args;
  va_start(args, numTensors);

  for (u8 x = 0; x < numTensors; x++) {
    tensors[x] = va_arg(args, Tensor *);
  }
  va_end(args);

  for (u8 x = 0; x < numTensors; x++) {
    Tensor *t = tensors[x];
    if (t == NULL || t->context == NULL) {
      return ERR_NULL_TENSOR_PROVIDED;
    }

    size_t valueBytes = t->size * getBytesForDtype(t->dtype);
    void *locationOnDest = allocateOnCtx(destCtx, valueBytes);
    Result copyResult =
        copyBetweenContexts(t->context, destCtx, t->values, locationOnDest, valueBytes);
    if (copyResult != OK) {
      return copyResult;
    }

    freeOnCtx(t->context, t->values);
    t->context = destCtx;
    t->values = locationOnDest;
  }

  return OK;
}
