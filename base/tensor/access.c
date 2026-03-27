#include "result/result.h"
#include "tensor_internal.h"
#include "value.h"
#include "../common.h"
#include <string.h>

static bool isOutOfBounds(Tensor *t, Dim dim) {
  for (u8 i = 0; i < dim.numOfDims; i++) {
    if (dim.dims[i] >= t->shape.dims[i]) {
      return true;
    }
  }
  return false;
}

static Result indexValueToDim(Value idxVal, dim_t *idx) {
  if (idx == NULL) {
    return ERR_NULL_PTR;
  }

  switch (idxVal.dtype) {
    case U8: *idx = idxVal.as.u8; return OK;
    case U16: *idx = idxVal.as.u16; return OK;
    case U32: *idx = idxVal.as.u32; return OK;
    case U64: *idx = idxVal.as.u64; return OK;
    case I8: *idx = (dim_t)idxVal.as.i8; return OK;
    case I16: *idx = (dim_t)idxVal.as.i16; return OK;
    case I32: *idx = (dim_t)idxVal.as.i32; return OK;
    case I64: *idx = (dim_t)idxVal.as.i64; return OK;
    default: return ERR_DTYPE_MISMATCH;
  }
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

  return readTensorValueAtFlatIndex(t, idx, result);
}

Result IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor *dest) {
  TensorArg sourceArg = {0};
  TensorArg indicesArg = {0};
  Result result = OK;

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

  result = materializeTensorOnContext(ctx, source, true, &sourceArg);
  if (result != OK) {
    return result;
  }

  result = materializeTensorOnContext(ctx, indices, true, &indicesArg);
  if (result != OK) {
    releaseTensorArg(ctx, &sourceArg);
    return result;
  }

  Tensor *workingSource = sourceArg.tensor;
  Tensor *workingIndices = indicesArg.tensor;
  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingSource->shape.numOfDims - 1 + workingIndices->shape.numOfDims;
  dim_t *newDims = NULL;
  multiplier_t *newMultipliers = NULL;
  if (newNumDims > 0) {
    newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
    newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
    result = ensureAllocated(newDims);
    if (result != OK) {
      goto cleanup_index;
    }
    result = ensureAllocated(newMultipliers);
    if (result != OK) {
      freeAlloc(ctx->memory, newDims);
      goto cleanup_index;
    }
  }

  for (u8 i = 0; i < workingIndices->shape.numOfDims; i++) {
    newDims[i] = workingIndices->shape.dims[i];
  }

  for (u8 i = 0; i < workingSource->shape.numOfDims - 1; i++) {
    newDims[workingIndices->shape.numOfDims + i] = workingSource->shape.dims[i + 1];
  }

  calculateNumValuesAndMultipliers((Dim){.dims = newDims, .numOfDims = newNumDims},
                                   newMultipliers);

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < workingSource->shape.numOfDims; i++) {
    sliceSize *= workingSource->shape.dims[i];
  }

  Dim destShape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers};
  result = initTensor(ctx, dest, destShape, workingSource->dtype);
  if (result != OK) {
    if (newMultipliers != NULL) {
      freeAlloc(ctx->memory, newMultipliers);
    }
    if (newDims != NULL) {
      freeAlloc(ctx->memory, newDims);
    }
    goto cleanup_index;
  }

  if (isCudaCtx) {
    result = runCudaIndexSelect1d(ctx, workingSource->dtype, workingSource->values,
                                  workingIndices->values, workingIndices->dtype, dest->values,
                                  workingIndices->size, sliceSize);
    goto cleanup_index;
  }

  tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (u64 i = 0; i < workingIndices->size; i++) {
    Value idxVal;
    result = readTensorValueAtFlatIndex(workingIndices, i, &idxVal);
    if (result != OK) {
      goto cleanup_index;
    }

    dim_t idx = 0;
    result = indexValueToDim(idxVal, &idx);
    if (result != OK) {
      goto cleanup_index;
    }

    dim_t srcCoords[workingSource->shape.numOfDims];
    srcCoords[0] = idx;
    for (u8 d = 1; d < workingSource->shape.numOfDims; d++) {
      srcCoords[d] = 0;
    }
    u64 srcOffset = getContigousIdxFromCoord(workingSource, srcCoords);

    result = copyBetweenContexts(workingSource->context, dest->context,
                                 (char *)workingSource->values + srcOffset * bytesPerElem,
                                 (char *)dest->values + destOffset * bytesPerElem,
                                 sliceSize * bytesPerElem);
    if (result != OK) {
      goto cleanup_index;
    }

    destOffset += sliceSize;
  }

  result = OK;

cleanup_index:
  releaseTensorArg(ctx, &indicesArg);
  releaseTensorArg(ctx, &sourceArg);
  return result;
}

Result IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *dest) {
  TensorArg sourceArg = {0};
  TensorArg rowArg = {0};
  TensorArg colArg = {0};
  Result result = OK;

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

  result = materializeTensorOnContext(ctx, source, true, &sourceArg);
  if (result != OK) {
    return result;
  }

  result = materializeTensorOnContext(ctx, rowIndices, true, &rowArg);
  if (result != OK) {
    releaseTensorArg(ctx, &sourceArg);
    return result;
  }

  result = materializeTensorOnContext(ctx, colIndices, true, &colArg);
  if (result != OK) {
    releaseTensorArg(ctx, &rowArg);
    releaseTensorArg(ctx, &sourceArg);
    return result;
  }

  Tensor *workingSource = sourceArg.tensor;
  Tensor *workingRows = rowArg.tensor;
  Tensor *workingCols = colArg.tensor;
  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingRows->shape.numOfDims + workingSource->shape.numOfDims - 2;
  dim_t *newDims = NULL;
  multiplier_t *newMultipliers = NULL;
  if (newNumDims > 0) {
    newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
    newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
    result = ensureAllocated(newDims);
    if (result != OK) {
      goto cleanup_index_2d;
    }
    result = ensureAllocated(newMultipliers);
    if (result != OK) {
      freeAlloc(ctx->memory, newDims);
      goto cleanup_index_2d;
    }
  }

  for (u8 i = 0; i < workingRows->shape.numOfDims; i++) {
    newDims[i] = workingRows->shape.dims[i];
  }

  for (u8 i = 0; i < workingSource->shape.numOfDims - 2; i++) {
    newDims[workingRows->shape.numOfDims + i] = workingSource->shape.dims[i + 2];
  }

  calculateNumValuesAndMultipliers((Dim){.dims = newDims, .numOfDims = newNumDims},
                                   newMultipliers);

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < workingSource->shape.numOfDims; i++) {
    sliceSize *= workingSource->shape.dims[i];
  }

  Dim destShape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers};
  result = initTensor(ctx, dest, destShape, workingSource->dtype);
  if (result != OK) {
    if (newMultipliers != NULL) {
      freeAlloc(ctx->memory, newMultipliers);
    }
    if (newDims != NULL) {
      freeAlloc(ctx->memory, newDims);
    }
    goto cleanup_index_2d;
  }

  if (isCudaCtx) {
    result = runCudaIndexSelect2d(ctx, workingSource->dtype, workingSource->values,
                                  workingSource->shape.dims[1], workingRows->values,
                                  workingRows->dtype, workingCols->values, workingCols->dtype,
                                  dest->values, workingRows->size, sliceSize);
    goto cleanup_index_2d;
  }

  tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (u64 i = 0; i < workingRows->size; i++) {
    Value rowVal, colVal;
    result = readTensorValueAtFlatIndex(workingRows, i, &rowVal);
    if (result != OK) {
      goto cleanup_index_2d;
    }
    result = readTensorValueAtFlatIndex(workingCols, i, &colVal);
    if (result != OK) {
      goto cleanup_index_2d;
    }

    dim_t rowIdx = 0;
    dim_t colIdx = 0;
    result = indexValueToDim(rowVal, &rowIdx);
    if (result != OK) {
      goto cleanup_index_2d;
    }
    result = indexValueToDim(colVal, &colIdx);
    if (result != OK) {
      goto cleanup_index_2d;
    }

    dim_t srcCoords[workingSource->shape.numOfDims];
    srcCoords[0] = rowIdx;
    srcCoords[1] = colIdx;
    for (u8 d = 2; d < workingSource->shape.numOfDims; d++) {
      srcCoords[d] = 0;
    }
    u64 srcOffset = getContigousIdxFromCoord(workingSource, srcCoords);

    result = copyBetweenContexts(workingSource->context, dest->context,
                                 (char *)workingSource->values + srcOffset * bytesPerElem,
                                 (char *)dest->values + destOffset * bytesPerElem,
                                 sliceSize * bytesPerElem);
    if (result != OK) {
      goto cleanup_index_2d;
    }

    destOffset += sliceSize;
  }

  result = OK;

cleanup_index_2d:
  releaseTensorArg(ctx, &colArg);
  releaseTensorArg(ctx, &rowArg);
  releaseTensorArg(ctx, &sourceArg);
  return result;
}

Result GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor *dest) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->shape.numOfDims == 0) {
    return ERR_DIM_MISMATCH;
  }

  if (!isSameContext(source->context, ctx)) {
    return ERR_NO_OP;
  }

  if (index >= source->shape.dims[0]) {
    return ERR_OUT_OF_BOUNDS;
  }

  u8 newNumDims = source->shape.numOfDims - 1;
  size_t bytesPerElem = getBytesForDtype(source->dtype);

  // Determine the dim0 base offset in the underlying storage (accounting for any
  // existing per-dim boundary on a view source).
  u64 baseDim0Start = (source->isView && source->boundary) ? source->boundary[0].start : 0;
  u64 selected = baseDim0Start + index;
  u64 offsetElems = selected * source->shape.multipliers[0];

  // Advance the values pointer so the result tensor starts at the correct row.
  void *newValues = (char *)source->values + offsetElems * bytesPerElem;

  // Handle case where we're reducing to 0-dim (scalar tensor)
  if (newNumDims == 0) {
    Value scalar;
    Result result = copyBetweenContexts(source->context, NULL, newValues, &scalar.as, bytesPerElem);
    if (result != OK) {
      return result;
    }
    scalar.dtype = source->dtype;
    *dest = singleValueTensor(ctx, scalar);
    if (dest->values == NULL) {
      return ERR_OUT_OF_MEMORY;
    }
    return OK;
  }

  // Allocate new dims and copy source dims[1:]
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  Result allocRes = ensureAllocated(newDims);
  if (allocRes != OK) {
    return allocRes;
  }
  for (u8 i = 0; i < newNumDims; i++) {
    newDims[i] = source->shape.dims[i + 1];
  }

  // Copy source multipliers[1:] — do NOT recompute; preserves transposed strides.
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  allocRes = ensureAllocated(newMultipliers);
  if (allocRes != OK) {
    freeAlloc(ctx->memory, newDims);
    return allocRes;
  }
  for (u8 i = 0; i < newNumDims; i++) {
    newMultipliers[i] = source->shape.multipliers[i + 1];
  }

  // Copy remaining per-dim boundary entries [1:] if source was a view.
  Range *boundary = NULL;
  if (source->isView && source->boundary) {
    boundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    allocRes = ensureAllocated(boundary);
    if (allocRes != OK) {
      freeAlloc(ctx->memory, newMultipliers);
      freeAlloc(ctx->memory, newDims);
      return allocRes;
    }
    for (u8 i = 0; i < newNumDims; i++) {
      boundary[i] = source->boundary[i + 1];
    }
  }

  *dest =
      tensorView(source->context, ctx->memory, newValues, source->size / source->shape.dims[0],
                 source->dtype,
                 (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                 boundary, false);

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

  return readTensorValueAtFlatIndex(t, 0, result);
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
  return writeTensorValueAtFlatIndex(t, idx, value);
}
