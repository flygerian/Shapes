#include "common.h"
#include "memory.h"
#include <time.h>

Context initializeContext(size_t arenaSize, size_t minBlockSize, bool withCuda) {
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

Context *createContext(size_t arenaSize, size_t minBlockSize, bool withCuda) {
  Context initialized = initializeContext(arenaSize, minBlockSize, withCuda);
  Context *ctx = allocate(initialized.memory, sizeof(Context));
  *ctx = initialized;
  return ctx;
}

void destroyContext(Context *ctx) {
  if (ctx->device != NULL && ctx->device->type == CUDA) {
    cublasDestroy(ctx->handle);
  }

  freeMemory(ctx->memory);
}

void freeContext(Context *ctx) {
  if (ctx == NULL) {
    return;
  }

  destroyContext(ctx);
}
