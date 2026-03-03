
#include "common.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <stddef.h>
#include <string.h>

// IndexAccumulate1d accumulates srcGrad into dest at positions given by indices.
// dest[idx, :] += srcGrad[i, :] for each i, where idx = indices[i].
// Supports repeated indices (scatter-add).
Result IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  (void)ctx;

  if (isInvalidTensor(dest) || isInvalidTensor(indices) || isInvalidTensor(srcGrad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dest->dtype != srcGrad->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (isIntType(indices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

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

    if (idx >= dest->shape.dims[0]) {
      return ERR_OUT_OF_BOUNDS;
    }

    dim_t destCoords[dest->shape.numOfDims];
    destCoords[0] = idx;
    for (u8 d = 1; d < dest->shape.numOfDims; d++) {
      destCoords[d] = 0;
    }
    u64 destBase = getContigousIdxFromCoord(dest, destCoords);
    u64 srcBase = i * sliceSize;

    for (tensor_size_t j = 0; j < sliceSize; j++) {
      Value dVal, sVal, res;
      VALUE_GET_FROM_ARR(dest->values, destBase + j, &dVal, dest->dtype);
      VALUE_GET_FROM_ARR(srcGrad->values, srcBase + j, &sVal, srcGrad->dtype);
      VALUE_BINOP(res, dVal, sVal, +);
      VALUE_SET(dest->values, destBase + j, res);
    }
  }

  return OK;
}

// IndexAccumulate2d accumulates srcGrad into dest at positions (rowIndices[i], colIndices[i]).
// dest[row, col, :] += srcGrad[i, :]. Supports repeated (row, col) pairs.
Result IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *srcGrad) {
  (void)ctx;

  if (isInvalidTensor(dest) || isInvalidTensor(rowIndices) || isInvalidTensor(colIndices) ||
      isInvalidTensor(srcGrad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dest->dtype != srcGrad->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (isIntType(rowIndices) || isIntType(colIndices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  if (rowIndices->size != colIndices->size) {
    return ERR_DIM_MISMATCH;
  }

  if (dest->shape.numOfDims < 2) {
    return ERR_DIM_MISMATCH;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

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

    if (rowIdx >= dest->shape.dims[0] || colIdx >= dest->shape.dims[1]) {
      return ERR_OUT_OF_BOUNDS;
    }

    dim_t destCoords[dest->shape.numOfDims];
    destCoords[0] = rowIdx;
    destCoords[1] = colIdx;
    for (u8 d = 2; d < dest->shape.numOfDims; d++) {
      destCoords[d] = 0;
    }
    u64 destBase = getContigousIdxFromCoord(dest, destCoords);
    u64 srcBase = i * sliceSize;

    for (tensor_size_t j = 0; j < sliceSize; j++) {
      Value dVal, sVal, res;
      VALUE_GET_FROM_ARR(dest->values, destBase + j, &dVal, dest->dtype);
      VALUE_GET_FROM_ARR(srcGrad->values, srcBase + j, &sVal, srcGrad->dtype);
      VALUE_BINOP(res, dVal, sVal, +);
      VALUE_SET(dest->values, destBase + j, res);
    }
  }

  return OK;
}

// SliceAccumulate accumulates srcGrad into dest over a sliced region.
// Equivalent to: dest[ranges...] += srcGrad
Result SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  (void)ctx;

  if (isInvalidTensor(dest) || isInvalidTensor(srcGrad) || ranges == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dest->dtype != srcGrad->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (dest->shape.numOfDims != srcGrad->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  u8 ndims = dest->shape.numOfDims;
  if (ndims == 0) {
    return ERR_DIM_MISMATCH;
  }

  for (u8 i = 0; i < ndims; i++) {
    if (ranges[i].start > ranges[i].end || ranges[i].end > dest->shape.dims[i]) {
      return ERR_OUT_OF_BOUNDS;
    }

    u64 span = ranges[i].end - ranges[i].start;
    if ((u64)srcGrad->shape.dims[i] != span) {
      return ERR_DIM_MISMATCH;
    }
  }

  // Iterate by outer coordinates and accumulate a full strided run on the last
  // dimension to avoid per-element coordinate unraveling.
  u8 lastDim = ndims - 1;
  u64 innerCount = srcGrad->shape.dims[lastDim];
  u64 srcStep = srcGrad->shape.multipliers[lastDim];
  u64 dstStep = dest->shape.multipliers[lastDim];

  dim_t srcCoords[ndims];
  dim_t dstCoords[ndims];
  for (u8 i = 0; i < ndims; i++) {
    srcCoords[i] = 0;
    dstCoords[i] = (dim_t)ranges[i].start;
  }

  u64 outerCount = 1;
  for (u8 i = 0; i < lastDim; i++) {
    outerCount *= srcGrad->shape.dims[i];
  }

  if (innerCount == 0 || outerCount == 0) {
    return OK;
  }

  for (u64 outer = 0; outer < outerCount; outer++) {
    srcCoords[lastDim] = 0;
    dstCoords[lastDim] = (dim_t)ranges[lastDim].start;

    u64 srcBase = getContigousIdxFromCoord(srcGrad, srcCoords);
    u64 dstBase = getContigousIdxFromCoord(dest, dstCoords);
    accumulateStridedByDtype(dest->dtype, dest->values, dstBase, dstStep, srcGrad->values, srcBase,
                             srcStep, innerCount);

    // Odometer increment for outer coordinates [0..lastDim).
    for (i32 d = (i32)lastDim - 1; d >= 0; d--) {
      srcCoords[d]++;
      dstCoords[d]++;
      if (srcCoords[d] < srcGrad->shape.dims[d]) {
        break;
      }
      srcCoords[d] = 0;
      dstCoords[d] = (dim_t)ranges[d].start;
    }
  }

  return OK;
}

