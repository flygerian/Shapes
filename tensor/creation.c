#include <string.h>
#include "tensor_internal.h"
#include "../memory.h"

Tensor* t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims=shape.numOfDims};

  tShape.dims = allocate(ctx->memory, sizeof(u32) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(u8) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(u32) * shape.numOfDims); 
  tensor_size_t size = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);
  size_t bytesRequired = size * getBytesForDtype(type);

  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  *t = (Tensor){.dtype = type, .values=allocate(ctx->memory, bytesRequired), .shape=tShape, .size=size, .isContigous=true};
  memset(t->values, 0, bytesRequired);
  return t;
}

Tensor* T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, U8);
}

Result Clone(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Tensor *source = t;
  if (!t->isContigous) {
    source = copyToContiguous(ctx, t);
  }

  size_t valueBytes = getBytesForDtype(source->dtype) * source->size;
  void *newValues = allocate(ctx->memory, valueBytes);
  memcpy(newValues, source->values, valueBytes);

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  *dest = (Tensor) {
    .dtype = source->dtype,
    .values = newValues,
    .size = source->size,
    .isContigous = true,
    .isView = false,
    .boundary = NULL,
    .shape = {
      .dims = newDims,
      .numOfDims = source->shape.numOfDims,
      .multipliers = newMultipliers
    }
  };

  return OK;
}
