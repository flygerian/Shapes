#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "utils_lib/memory.h"
#include "utils_lib/cuda_memory.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
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
  Memory *memory = initializeArena(hostArenaSize, 1);
  Context ctx = {.memory = memory};
  attachCudaDevice(&ctx); 
  ctx.cudaMetadataMemory = initializeArena(hostArenaSize, 1);

  return ctx;
}

Context shapes_GetScratchContext(Context *ctx, size_t bufferSize) {
  Context scratch = {
      .device = ctx->device, 
      .handle = ctx->handle, 
      .isTraining = ctx->isTraining, 
      .memory = ctx->memory,
      .cudaMemory = ctx->cudaMemory,
      .cudaMetadataMemory = ctx->cudaMetadataMemory,
      .parent = ctx, 
  };

  scratch.cudaMemory = GetCudaMemoryScratchCheckPoint(&ctx->cudaMemory);
  scratch.memory = GetScratchArena(ctx->memory, bufferSize);

  return scratch;
}

void shapes_DestroyContext(Context *ctx) {
  if (ctx->device != NULL && ctx->device->type == CUDA) {
    ReleaseCudaBlocks(&ctx->cudaMemory);
    cublasDestroy(ctx->handle);
  }

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
      cudaError_t syncResult = cudaDeviceSynchronize();
      PANIC_WITH_MSG_IF(syncResult != cudaSuccess, cudaGetErrorString(syncResult));
      return OK;
    }
    default: return OK;
  }
}
