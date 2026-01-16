#include "tensor.h"
#include <stddef.h>
#include <string.h>
#include "tensor.h"
#include "stdbool.h"


static size_t getBytesForDtype(Dtype type){
  switch(type) {
    case U8:
      return sizeof(u8);

    case U16:
      return sizeof(u16);

    case U32:
      return sizeof(u32);

    case U64:
      return sizeof(u64);

    case F32:
      return sizeof(float);

    default:
      return -1;
  }
}

static u64 calculateNumValuesAndMultipliers(Dim shape, u8 *multipliers) {
  u64 numberOfValues = 1;

  for (int x = shape.numOfDims - 1; x >= 0; x--) {
    multipliers[x] = numberOfValues;
    numberOfValues *= shape.dims[x];
  }

  return numberOfValues;
}


static Tensor t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims=shape.numOfDims};

  // Dont want to hold on the the original memory space, so it can be freed
  tShape.dims = allocate(ctx->memory, sizeof(u32) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(u8) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(u32) * shape.numOfDims); 
  u64 numberOfValues = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);

  size_t bytesRequired = numberOfValues * getBytesForDtype(type);

  Tensor t = (Tensor){.dtype = type, .values=allocate(ctx->memory, bytesRequired), .shape=tShape};
  memset(t.values, 0, bytesRequired);
  return t;
}

static bool doTensorShapesMatch(Tensor *a, Tensor *b) {
   
  return true;
}

static u64 getIdx(Tensor *t, Dim idx) {
  u64 result = 0;

  for (u8 x = 0; x < t->shape.numOfDims; x++) {
    result += idx.dims[x] * t->shape.multipliers[x];
  }

  return result;
}

static bool isOutOfBounds(Tensor *t, Dim dim) {
  for (u8 i = 0; i < dim.numOfDims; i++) {
    if (dim.dims[i] >= t->shape.dims[i]) {
      return true;
    }
  }

  return false;
}

// Ops
Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {  // Guards check the dimensions
  // MAybe check the datatypes?
 return OK; 
}

Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  // Guards check the dimensions
  // MAybe check the datatypes?
  return OK;
}

Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination) {
  // Guards check the dimensions
  // MAybe check the datatypes?
  return OK;
}

Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  // Guards check the dimensions
// MAybe check the datatypes?
  return OK;
}

Result GetAt(Tensor *t, Dim dim, Value *result) {
  if (t == NULL || result == NULL) {
    return ERR_NULL_PTR;
  }

  if (dim.numOfDims != t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (isOutOfBounds(t, dim)) {
    return ERR_OUT_OF_BOUNDS;
  }

  *result = (Value){.dtype = t->dtype};
  u64 idx = getIdx(t, dim);
  VALUE_GET_FROM_ARR(t->values, idx, result);

  return OK;
}

Result AssignValue(Context *ctx, Tensor *t, Dim dim, Value value) {
  if (t == NULL || t->values == NULL || dim.dims == NULL) {
    return ERR_NULL_PTR;
  }

  if (t->dtype != value.dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (dim.numOfDims != t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (isOutOfBounds(t, dim)) {
    return ERR_OUT_OF_BOUNDS;
  }

  u64 idx = getIdx(t, dim);
  VALUE_SET(t->values, idx, value);

  return OK;
}

// Creation
Tensor T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, U8);
}
