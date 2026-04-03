#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "../memory.h"
#include <sched.h>

static Memory *getTensorMetadataMemory(Context *ctx, Tensor *t) {
  if (t != NULL && t->metadataMemory != NULL) {
    return t->metadataMemory;
  }

  if (ctx != NULL) {
    return ctx->memory;
  }

  return NULL;
}

static Result freeViewTensorInternal(Context *ctx, Tensor *t) {
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

static Result freeTensorInternal(Context *ctx, Tensor *t) {
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

Result FreeTensors(Context *ctx, Tensor **tensors, int numTensors) {
  if (tensors == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  for (int i = 0; i < numTensors; i++) {
    Tensor *t = tensors[i];

    if (t == NULL) {
      return ERR_NULL_TENSOR_PROVIDED;
    }

    if (t->isView) {
      Result res = freeViewTensorInternal(ctx, t);
      if (res != OK) {
        return res;
      }

      continue;
    }

    Result res = freeTensorInternal(ctx, t);
    if (res != OK) {
      return res;
    }
  }

  return OK;
}

Result FreeViewTensor(Context *ctx, Tensor *t) {
  return freeViewTensorInternal(ctx, t);
}

Result FreeTensor(Context *ctx, Tensor *t) {
  return freeTensorInternal(ctx, t);
}

Result freeTensorBuffers(Context *ctx, Tensor *t) {
  Memory *metadataMemory = getTensorMetadataMemory(ctx, t);

  if (!t->isView && t->values) { 
    freeOnCtx(t->context != NULL ? t->context : ctx, t->values); 
  }

  if (t->shape.dims) {
    freeAlloc(metadataMemory, t->shape.dims);
  }

  if (t->shape.multipliers) {
    freeAlloc(metadataMemory, t->shape.multipliers);
  }

  if (t->boundary) {
    freeAlloc(metadataMemory, t->boundary);
  }

  t->values = NULL;
  t->shape.dims = NULL;
  t->shape.multipliers = NULL;
  t->boundary = NULL;

  return OK;
}
