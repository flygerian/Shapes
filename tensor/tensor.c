#include "tensor.h"
#include <stdarg.h>
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
    u64 coord = idx.dims[x];
    if (t->isView) {
      coord += t->boundary[x].start;
    }
    result += coord * t->shape.multipliers[x];
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
// Maybe check the datatypes?
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

  // Apply offset here if viewH
  u64 idx = getIdx(t, dim);
  VALUE_SET(t->values, idx, value);

  return OK;
}

Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...) {
  Range *ranges = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  va_list args;
  va_start(args, dest);

  for(u8 x=0; x < source->shape.numOfDims; x++) {
    // Worried about this
    ranges[x] = va_arg(args, Range);

    if (ranges[x].end < ranges[x].start) {
      va_end(args);
      return ERR_INVALID_RANGE;
    }

    if (ranges[x].start < 0 || ranges[x].start > source->shape.dims[x] || ranges[x].end < 0 || ranges[x].end > source->shape.dims[x]) {
      va_end(args);
      return ERR_DIM_MISMATCH;
    }
  }
  va_end(args);

  // Compute the new shape
  // Since this would be a view and views indexes are converted to the base tensors index range using the boundaries. We keep the old multiplier
  Dim newShape = {.dims=allocate(ctx->memory, sizeof(Dim)* source->shape.numOfDims), .numOfDims=source->shape.numOfDims, .multipliers=source->shape.multipliers}; 
  Range *boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  for(u8 x=0; x < source->shape.numOfDims; x++) {
    Range r = ranges[x];
    u32 dimsize = (r.end - r.start) + 1; // (Range 1-5 should yeild 5)
    newShape.dims[x] = dimsize;

    if (source->isView) {
      boundary[x] = (Range) {.start=source->boundary->start + ranges[x].start, .end=source->boundary->start + ranges[x].end };
    } else {
      boundary[x] = ranges[x];
    }
  }

  *dest = ((Tensor) {.isView = true, .values=source->values, .shape=newShape, .dtype=source->dtype, .boundary=boundary});

  return OK;
}

// Creation
Tensor T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, U8);
}
