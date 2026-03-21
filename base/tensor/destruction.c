#include "common.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "../memory.h"
#include <stdio.h>

static Memory *getTensorMetadataMemory(Context *ctx, Tensor *t) {
  if (t != NULL && t->metadataMemory != NULL) {
    return t->metadataMemory;
  }

  if (ctx != NULL) {
    return ctx->memory;
  }

  return NULL;
}

Result FreeViewTensor(Context *ctx, Tensor *t) {
  if (t == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Memory *metadataMemory = getTensorMetadataMemory(ctx, t);

  if (t->shape.dims != NULL) {
    freeAlloc(metadataMemory, t->shape.dims);
  }
  if (t->shape.multipliers != NULL) {
    freeAlloc(metadataMemory, t->shape.multipliers);
  }
  if (t->boundary != NULL) {
    freeAlloc(metadataMemory, t->boundary);
  }

  freeAlloc(metadataMemory, t);
  return OK;
}

Result FreeTensor(Context *ctx, Tensor *t) {
  if (t == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->isView) {
    return ERR_CANNOT_FREE_VIEW_TENSOR;
  }

  if (t->values != NULL) {
    freeOnCtx(t->context != NULL ? t->context : ctx, t->values);
  }
  Memory *metadataMemory = getTensorMetadataMemory(ctx, t);
  if (t->shape.dims != NULL) {
    freeAlloc(metadataMemory, t->shape.dims);
  }
  if (t->shape.multipliers != NULL) {
    freeAlloc(metadataMemory, t->shape.multipliers);
  }
  if (t->boundary != NULL) {
    freeAlloc(metadataMemory, t->boundary);
  }

  freeAlloc(metadataMemory, t);
  return OK;
}

Result freeTensorBuffers(Context *ctx, Tensor *t) {
  Memory *metadataMemory = getTensorMetadataMemory(ctx, t);
  if (!t->isView && t->values)
    freeOnCtx(t->context != NULL ? t->context : ctx, t->values);
  if (t->shape.dims)
    freeAlloc(metadataMemory, t->shape.dims);
  if (t->shape.multipliers)
    freeAlloc(metadataMemory, t->shape.multipliers);
  if (t->boundary)
    freeAlloc(metadataMemory, t->boundary);

  t->values = NULL;
  t->shape.dims = NULL;
  t->shape.multipliers = NULL;
  t->boundary = NULL;

  return OK;
}
