#include "tensor_internal.h"
#include "../memory.h"

Result FreeTensor(Context *ctx, Tensor *t) {
  if (t == NULL)
    return ERR_NULL_TENSOR_PROVIDED;

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
