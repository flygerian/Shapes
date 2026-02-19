#include "tensor_internal.h"
#include "value.h"

static bool isOutOfBounds(Tensor *t, Dim dim) {
  for (u8 i = 0; i < dim.numOfDims; i++) {
    if (dim.dims[i] >= t->shape.dims[i]) {
      return true;
    }
  }
  return false;
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

  u64 idx = getContigousIdxFromCoord(t, dim.dims);
  
  // Add boundary offset for views
  if (t->isView && t->boundary) {
    idx += t->boundary->start;
  }
  
  VALUE_GET_FROM_ARR(t->values, idx, result, t->dtype);

  return OK;
}

Result GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor *dest) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->shape.numOfDims == 0) {
    return ERR_DIM_MISMATCH;
  }

  if (index >= source->shape.dims[0]) {
    return ERR_OUT_OF_BOUNDS;
  }

  u8 newNumDims = source->shape.numOfDims - 1;
  
  // Handle case where we're reducing to 0-dim (scalar tensor)
  if (newNumDims == 0) {
    *dest = (Tensor){
      .dtype = source->dtype,
      .values = source->values,
      .size = 1,
      .isContigous = source->isContigous,
      .isView = true,
      .shape = {.dims = NULL, .numOfDims = 0, .multipliers = NULL},
      .boundary = NULL
    };
    
    // Calculate boundary for the single element (flat offset)
    Range *boundary = allocate(ctx->memory, sizeof(Range));
    if (source->isView && source->boundary) {
      // For views, add the index offset to the existing offset
      // For 0-dim views, boundary->start is the flat offset
      // For higher-dim views, we need to calculate the stride
      if (source->shape.numOfDims == 0) {
        u64 offset = source->boundary->start + index;
        *boundary = (Range){.start = offset, .end = offset + 1};
      } else {
        u64 offset = source->boundary->start + index * source->shape.multipliers[0];
        *boundary = (Range){.start = offset, .end = offset + 1};
      }
    } else {
      // For non-view source, calculate the flat offset based on index and first dim stride
      u64 offset = index;
      if (source->shape.numOfDims > 1) {
        offset = index * source->shape.multipliers[0];
      }
      *boundary = (Range){.start = offset, .end = offset + 1};
    }
    dest->boundary = boundary;
    
    return OK;
  }

  // Allocate new dims and multipliers for non-scalar result
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  
  for (u8 i = 0; i < newNumDims; i++) {
    newDims[i] = source->shape.dims[i + 1];
  }
  
  calculateNumValuesAndMultipliers(
    (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = NULL},
    newMultipliers
  );

  // Calculate boundary (flat offset for this slice)
  Range *boundary = allocate(ctx->memory, sizeof(Range));
  if (source->isView && source->boundary) {
    // Add the index contribution to the existing offset
    u64 offset = source->boundary->start + index * source->shape.multipliers[0];
    *boundary = (Range){.start = offset, .end = offset + 1};
  } else {
    // For non-view source, the offset is index * stride of first dimension
    u64 offset = index * source->shape.multipliers[0];
    *boundary = (Range){.start = offset, .end = offset + 1};
  }

  *dest = (Tensor){
    .dtype = source->dtype,
    .values = source->values,
    .size = source->size / source->shape.dims[0],
    .isContigous = false,
    .isView = true,
    .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
    .boundary = boundary
  };

  return OK;
}

Result GetScalar(Tensor *t, Value *result) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->shape.numOfDims != 0) {
    return ERR_DIM_MISMATCH;
  }

  if (result == NULL) {
    return ERR_NULL_PTR;
  }

  // For 0-dim tensor, get the single value
  u64 idx = 0;
  if (t->isView && t->boundary) {
    idx = t->boundary->start;
  }
  
  VALUE_GET_FROM_ARR(t->values, idx, result, t->dtype);
  
  return OK;
}

Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
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

  u64 idx = getContigousIdxFromCoord(t, dim.dims);
  VALUE_SET(t->values, idx, value);

  return OK;
}
