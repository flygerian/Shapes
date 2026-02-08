#include "common.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor/value.h"
#include "tensor_internal.h"
#include "../memory.h"
#include <complex.h>
#include <string.h>

static void initializeGradient(Context *ctx, Tensor *t) {
  if (!ctx->grad) {
    return;
  }

  GraphNode *node = allocate(ctx->memory, sizeof(GraphNode));
  Tensor *grad = allocate(ctx->memory, sizeof(Tensor));

  // Create gradient tensor with same shape and dtype
  size_t valueBytes = getBytesForDtype(t->dtype) * t->size;
  dim_t *gradDims = allocate(ctx->memory, sizeof(dim_t) * t->shape.numOfDims);
  memcpy(gradDims, t->shape.dims, sizeof(dim_t) * t->shape.numOfDims);

  multiplier_t *gradMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * t->shape.numOfDims);
  memcpy(gradMultipliers, t->shape.multipliers, sizeof(multiplier_t) * t->shape.numOfDims);

  *grad = (Tensor){
      .dtype = t->dtype,
      .values = allocate(ctx->memory, valueBytes),
      .size = t->size,
      .isContigous = true,
      .isView = false,
      .boundary = NULL,
      .shape = {.dims = gradDims, .numOfDims = t->shape.numOfDims, .multipliers = gradMultipliers}};
  memset(grad->values, 0, valueBytes);

  *node = (GraphNode){
      .output = t, .grad = grad, .inputs = NULL, .numInputs = 0, .backward = NULL, .optype = 0};

  t->computation = node;
}

Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims = shape.numOfDims};

  tShape.dims = allocate(ctx->memory, sizeof(u32) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(u8) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(u32) * shape.numOfDims);
  tensor_size_t size = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);
  size_t bytesRequired = size * getBytesForDtype(type);

  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  *t = (Tensor){.dtype = type,
                .values = allocate(ctx->memory, bytesRequired),
                .shape = tShape,
                .size = size,
                .isContigous = true};
  memset(t->values, 0, bytesRequired);

  initializeGradient(ctx, t);

  return t;
}

Tensor *T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, F32);
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

  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  *dest = (Tensor){.dtype = source->dtype,
                   .values = newValues,
                   .size = source->size,
                   .isContigous = true,
                   .isView = false,
                   .boundary = NULL,
                   .shape = {.dims = newDims,
                             .numOfDims = source->shape.numOfDims,
                             .multipliers = newMultipliers}};

  initializeGradient(ctx, dest);

  return OK;
}

void SetValues(Tensor *t, Value value) {
  for (tensor_size_t i = 0; i < t->size; i++) {
    VALUE_SET(t->values, i, value);
  }
}

Tensor *T_Int(Context *ctx, Dim shape, i8 initialValue) {
  Tensor *init = t_Zeros(ctx, shape, I8);
  Value v = (Value){.dtype = I8, .as.i8 = initialValue};
  SetValues(init, v);
  return init;
}

Tensor *T_Float(Context *ctx, Dim shape, f32 initialValue) {
  Tensor *init = t_Zeros(ctx, shape, F32);
  Value v = (Value){.dtype = F32, .as.f32 = initialValue};
  SetValues(init, v);
  return init;
}
