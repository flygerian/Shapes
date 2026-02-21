#include "result/result.h"
#include "tensor_internal.h"
#include "value.h"
#include <string.h>

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
    *dest = (Tensor){.dtype = source->dtype,
                     .values = source->values,
                     .size = 1,
                     .isContigous = source->isContigous,
                     .isView = true,
                     .shape = {.dims = NULL, .numOfDims = 0, .multipliers = NULL},
                     .boundary = NULL};

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
      (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = NULL}, newMultipliers);

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

  *dest =
      (Tensor){.dtype = source->dtype,
               .values = source->values,
               .size = source->size / source->shape.dims[0],
               .isContigous = false,
               .isView = true,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
               .boundary = boundary};

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

Result IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor *dest) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isInvalidTensor(indices)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->shape.numOfDims == 0) {
    return ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING;
  }

  if (isIntType(indices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  dim_t firstDim = source->shape.dims[0];

  for (u64 i = 0; i < indices->size; i++) {
    Value idxVal;
    VALUE_GET_FROM_ARR(indices->values, i, &idxVal, indices->dtype);

    dim_t idx;
    switch (indices->dtype) {
      case U8: idx = idxVal.as.u8; break;
      case U16: idx = idxVal.as.u16; break;
      case U32: idx = idxVal.as.u32; break;
      case U64: idx = idxVal.as.u64; break;
      case I8: idx = (dim_t)idxVal.as.i8; break;
      case I16: idx = (dim_t)idxVal.as.i16; break;
      case I32: idx = (dim_t)idxVal.as.i32; break;
      case I64: idx = (dim_t)idxVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    if (idx >= firstDim) {
      return ERR_OUT_OF_BOUNDS;
    }
  }

  u8 newNumDims = source->shape.numOfDims - 1 + indices->shape.numOfDims;

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);

  for (u8 i = 0; i < indices->shape.numOfDims; i++) {
    newDims[i] = indices->shape.dims[i];
  }

  for (u8 i = 0; i < source->shape.numOfDims - 1; i++) {
    newDims[indices->shape.numOfDims + i] = source->shape.dims[i + 1];
  }

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  calculateNumValuesAndMultipliers(
      (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = NULL}, newMultipliers);

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < source->shape.numOfDims; i++) {
    sliceSize *= source->shape.dims[i];
  }

  tensor_size_t destSize = indices->size * sliceSize;

  void *destValues = allocate(ctx->memory, getBytesForDtype(source->dtype) * destSize);

  tensor_size_t destOffset = 0;
  for (u64 i = 0; i < indices->size; i++) {
    Value idxVal;
    VALUE_GET_FROM_ARR(indices->values, i, &idxVal, indices->dtype);

    dim_t idx;
    switch (indices->dtype) {
      case U8: idx = idxVal.as.u8; break;
      case U16: idx = idxVal.as.u16; break;
      case U32: idx = idxVal.as.u32; break;
      case U64: idx = idxVal.as.u64; break;
      case I8: idx = (dim_t)idxVal.as.i8; break;
      case I16: idx = (dim_t)idxVal.as.i16; break;
      case I32: idx = (dim_t)idxVal.as.i32; break;
      case I64: idx = (dim_t)idxVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    u64 srcOffset;
    if (source->isView && source->boundary) {
      srcOffset = source->boundary->start + idx * source->shape.multipliers[0];
    } else {
      srcOffset = idx * source->shape.multipliers[0];
    }

    size_t bytesPerElem = getBytesForDtype(source->dtype);
    memcpy((char *)destValues + destOffset * bytesPerElem,
           (char *)source->values + srcOffset * bytesPerElem, sliceSize * bytesPerElem);

    destOffset += sliceSize;
  }

  *dest =
      (Tensor){.dtype = source->dtype,
               .values = destValues,
               .size = destSize,
               .isContigous = true,
               .isView = false,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
               .boundary = NULL};

  return OK;
}

Result IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *dest) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isInvalidTensor(rowIndices)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isInvalidTensor(colIndices)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->shape.numOfDims < 2) {
    return ERR_DIM_MISMATCH;
  }

  if (isIntType(rowIndices) || isIntType(colIndices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  if (rowIndices->size != colIndices->size) {
    return ERR_DIM_MISMATCH;
  }

  dim_t numRows = source->shape.dims[0];
  dim_t numCols = source->shape.dims[1];

  for (u64 i = 0; i < rowIndices->size; i++) {
    Value rowVal, colVal;
    VALUE_GET_FROM_ARR(rowIndices->values, i, &rowVal, rowIndices->dtype);
    VALUE_GET_FROM_ARR(colIndices->values, i, &colVal, colIndices->dtype);

    dim_t rowIdx, colIdx;
    switch (rowIndices->dtype) {
      case U8: rowIdx = rowVal.as.u8; break;
      case U16: rowIdx = rowVal.as.u16; break;
      case U32: rowIdx = rowVal.as.u32; break;
      case U64: rowIdx = rowVal.as.u64; break;
      case I8: rowIdx = (dim_t)rowVal.as.i8; break;
      case I16: rowIdx = (dim_t)rowVal.as.i16; break;
      case I32: rowIdx = (dim_t)rowVal.as.i32; break;
      case I64: rowIdx = (dim_t)rowVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    switch (colIndices->dtype) {
      case U8: colIdx = colVal.as.u8; break;
      case U16: colIdx = colVal.as.u16; break;
      case U32: colIdx = colVal.as.u32; break;
      case U64: colIdx = colVal.as.u64; break;
      case I8: colIdx = (dim_t)colVal.as.i8; break;
      case I16: colIdx = (dim_t)colVal.as.i16; break;
      case I32: colIdx = (dim_t)colVal.as.i32; break;
      case I64: colIdx = (dim_t)colVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    if (rowIdx >= numRows || colIdx >= numCols) {
      return ERR_OUT_OF_BOUNDS;
    }
  }

  u8 newNumDims = rowIndices->shape.numOfDims + source->shape.numOfDims - 2;

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);

  for (u8 i = 0; i < rowIndices->shape.numOfDims; i++) {
    newDims[i] = rowIndices->shape.dims[i];
  }

  for (u8 i = 0; i < source->shape.numOfDims - 2; i++) {
    newDims[rowIndices->shape.numOfDims + i] = source->shape.dims[i + 2];
  }

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  calculateNumValuesAndMultipliers(
      (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = NULL}, newMultipliers);

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < source->shape.numOfDims; i++) {
    sliceSize *= source->shape.dims[i];
  }

  tensor_size_t destSize = rowIndices->size * sliceSize;

  void *destValues = allocate(ctx->memory, getBytesForDtype(source->dtype) * destSize);

  tensor_size_t destOffset = 0;
  for (u64 i = 0; i < rowIndices->size; i++) {
    Value rowVal, colVal;
    VALUE_GET_FROM_ARR(rowIndices->values, i, &rowVal, rowIndices->dtype);
    VALUE_GET_FROM_ARR(colIndices->values, i, &colVal, colIndices->dtype);

    dim_t rowIdx, colIdx;
    switch (rowIndices->dtype) {
      case U8: rowIdx = rowVal.as.u8; break;
      case U16: rowIdx = rowVal.as.u16; break;
      case U32: rowIdx = rowVal.as.u32; break;
      case U64: rowIdx = rowVal.as.u64; break;
      case I8: rowIdx = (dim_t)rowVal.as.i8; break;
      case I16: rowIdx = (dim_t)rowVal.as.i16; break;
      case I32: rowIdx = (dim_t)rowVal.as.i32; break;
      case I64: rowIdx = (dim_t)rowVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    switch (colIndices->dtype) {
      case U8: colIdx = colVal.as.u8; break;
      case U16: colIdx = colVal.as.u16; break;
      case U32: colIdx = colVal.as.u32; break;
      case U64: colIdx = colVal.as.u64; break;
      case I8: colIdx = (dim_t)colVal.as.i8; break;
      case I16: colIdx = (dim_t)colVal.as.i16; break;
      case I32: colIdx = (dim_t)colVal.as.i32; break;
      case I64: colIdx = (dim_t)colVal.as.i64; break;
      default: return ERR_DTYPE_MISMATCH;
    }

    u64 srcOffset;
    if (source->isView && source->boundary) {
      srcOffset = source->boundary->start + rowIdx * source->shape.multipliers[0] +
                  colIdx * source->shape.multipliers[1];
    } else {
      srcOffset = rowIdx * source->shape.multipliers[0] + colIdx * source->shape.multipliers[1];
    }

    size_t bytesPerElem = getBytesForDtype(source->dtype);
    memcpy((char *)destValues + destOffset * bytesPerElem,
           (char *)source->values + srcOffset * bytesPerElem, sliceSize * bytesPerElem);

    destOffset += sliceSize;
  }

  *dest =
      (Tensor){.dtype = source->dtype,
               .values = destValues,
               .size = destSize,
               .isContigous = true,
               .isView = false,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
               .boundary = NULL};

  return OK;
}
