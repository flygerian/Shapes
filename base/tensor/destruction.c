#include "common.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "../memory.h"
#include <stdio.h>

Result FreeViewTensor(Context *ctx, Tensor *t) {
  if (t == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->shape.dims != NULL) {
    freeAlloc(ctx->memory, t->shape.dims);
  }
  if (t->shape.multipliers != NULL) {
    freeAlloc(ctx->memory, t->shape.multipliers);
  }
  if (t->boundary != NULL) {
    freeAlloc(ctx->memory, t->boundary);
  }

  freeAlloc(ctx->memory, t);
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
    freeAlloc(ctx->memory, t->values);
  }
  if (t->shape.dims != NULL) {
    freeAlloc(ctx->memory, t->shape.dims);
  }
  if (t->shape.multipliers != NULL) {
    freeAlloc(ctx->memory, t->shape.multipliers);
  }

  freeAlloc(ctx->memory, t);
  return OK;
}

Result freeTensorBuffers(Context *ctx, Tensor *t) {
  if (!t->isView && t->values)
    freeAlloc(ctx->memory, t->values);
  if (t->shape.dims)
    freeAlloc(ctx->memory, t->shape.dims);
  if (t->shape.multipliers)
    freeAlloc(ctx->memory, t->shape.multipliers);

  t->values = NULL;
  t->shape.dims = NULL;
  t->shape.multipliers = NULL;

  return OK;
}
