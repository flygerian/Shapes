#include "result.h"
#include "shapes.h"
#include "memory.h"
#ifdef SHAPES_HAS_CUDA 
  #include "shapescuda.h"
  #include "cuda_runtime.h"
  #include <cublas_v2.h>
#endif
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "shapes_internal.h"

Context shapes_InitializeHostContext(size_t arenaSize, size_t minBlockSize) {
  Memory *memory = initializeArena(arenaSize, minBlockSize);
  Context ctx = {.memory = memory};
  attachHostDevice(&ctx);

  return ctx;
}

Context shapes_InitializeCudaContext(size_t hostArenaSize) {
#ifndef SHAPES_HAS_CUDA
  PANIC_WITH_MSG_IF(1, "CUDA support is not compiled in");
#endif
  Memory *memory = initializeArena(hostArenaSize, 1);
  Context ctx = {.memory = memory};
  attachCudaDevice(&ctx);
  PANIC_WITH_MSG_IF(ctx.device == NULL, "No CUDA device available");
  ctx.cudaMetadataMemory = initializeArena(hostArenaSize, 1);

  return ctx;
}

Context shapes_GetScratchContext(Context *ctx, size_t bufferSize) {
  Context scratch = {
      .device = ctx->device,

      #ifdef SHAPES_HAS_CUDA 
      .handle = ctx->handle,
      #endif
      .isTraining = ctx->isTraining,
      .memory = ctx->memory,
      .cudaMemory = ctx->cudaMemory,
      .cudaMetadataMemory = ctx->cudaMetadataMemory,
      .parent = ctx,
  };

  scratch.memory = GetScratchArena(ctx->memory, bufferSize);
  
  #ifdef SHAPES_HAS_CUDA 
  if(ctx->device->type == CUDA) {
    scratch.cudaMemory = shapescuda_GetMemoryScratchCheckPoint(&ctx->cudaMemory);
  }
  #endif

  return scratch;
}

void shapes_DestroyContext(Context *ctx) {

  #ifdef SHAPES_HAS_CUDA 
  if (ctx->device != NULL && ctx->device->type == CUDA) {
    shapescuda_ReleaseBlocks(&ctx->cudaMemory);
    cublasDestroy(ctx->handle);
  }
  #endif

  freeMemory(ctx->memory);
}

void shapes_FreeContext(Context *ctx) {
  if (ctx == NULL) {
    return;
  }

  shapes_DestroyContext(ctx);
}

Result shapes_Flush(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return OK;
  }

  switch (ctx->device->type) {
    case CPU: return OK;
    case CUDA: {
      #ifdef SHAPES_HAS_CUDA 
      cudaError_t syncResult = cudaDeviceSynchronize();
      PANIC_WITH_MSG_IF(syncResult != cudaSuccess, cudaGetErrorString(syncResult));
      #endif
      return OK;
    }
    default: return OK;
  }
}
