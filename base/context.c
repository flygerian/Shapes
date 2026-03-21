#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "shapes.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdarg.h>
#include <stdarg.h>
#include <stddef.h>
#include <time.h>

Result copyBetweenContexts(Context* restrict srcCtx, Context* restrict destCtx, void* restrict srcPtr, void* restrict destPtr, size_t size) {
  if (srcCtx->device->type == CPU && destCtx.device->type == CUDA) {
    cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyHostToDevice);
    return OK;
  }

  if (srcCtx->device->type == CUDA && destCtx.device->type == CPU) {
    cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice);
    return  OK;
  }

  return ERR_NO_OP;
}

void *allocateOnCtx(Context *ctx, size_t size) {
  if (ctx->device == NULL) {
    // CPU allocate by default
    return allocate(ctx->memory, size);
  } 

  switch (ctx->device->type) {
    case CPU:
      return allocate(ctx->memory, size);

    case CUDA:
      void *locationOnDestCtx;
      cudaMalloc(locationOnDestCtx, size);
      return locationOnDestCtx;
  }
}


void freeOnCtx(Context *ctx, void* ptr) {
  if (ctx->device == NULL) {
    // CPU allocate by default
    freeAlloc(ctx->memory, ptr);
  } 

  switch (ctx->device->type) {
    case CPU:
      freeAlloc(ctx->memory, ptr);

    case CUDA:
      cudaFree(ptr);
  }
}

Context InitializeContext(size_t arenaSize, size_t minBlockSize, bool withCuda) {
  Memory *memory = initializeArena(arenaSize, minBlockSize);
  Context ctx = {.memory = memory};

  if (withCuda) {
    Device *device = allocate(memory, sizeof(Device));
    device->id = "cuda: 1";
    device->type = CUDA;

    cublasHandle_t handle;
    cublasCreate(&handle);
    ctx.handle = handle;
    ctx.device = device;
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

Result MoveTensors(Context *destCtx, u8 numTensors, ...) {
  if (destCtx == NULL) {
    return ERR_MOVE_CTX_DEVICE_IS_NULL;
  }

  Tensor* tensors[numTensors];

  va_list args;
  va_start(args, numTensors);

  for (u8 x=0; x < numTensors; x++) {
    tensors[x] = va_arg(args, Tensor*);
  }
  va_end(args);

  for (u8 x = 0; x < numTensors; x++) {
    Tensor *t = tensors[x];
    void *locationOnDest = allocateOnCtx(destCtx, t->size * getBytesForDtype(t->dtype));
    copyBetweenContexts(t->context, destCtx, t->values, locationOnDest, t->size);

    freeOnCtx(t->context, t->values);
    t->context = destCtx;
    t->values = locationOnDest;
  }
}

