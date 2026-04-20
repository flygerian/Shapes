#include "result/result.h"
#include "tensor_internal.h"
#include "../common.h"
#include "utils_lib/memory.h"
#include <string.h>
#include <stdlib.h>

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

Value *GetAt(Tensor *t, Dim dim) {
  PANIC_IF(dim.numOfDims != t->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(isOutOfBounds(t, dim), ERR_OUT_OF_BOUNDS);

  u64 idx = getContigousIdxFromCoord(t, dim.dims);

  Value *result = allocate(t->context->memory, sizeof(Value));

  readTensorValueAtFlatIndex(t, idx, result);
  return result;
}

Tensor *IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(indices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->shape.numOfDims == 0, ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING);
  PANIC_IF(isIntType(indices), ERR_ONLY_INT_TYPE_ALLOWED);

  Tensor *workingSource = materializeTensorOnContext(ctx, source);
  Tensor *workingIndices = materializeTensorOnContext(ctx, indices);

  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingSource->shape.numOfDims - 1 + workingIndices->shape.numOfDims;
  dim_t *newDims = NULL;
  if (newNumDims > 0) {
    newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
    PANIC_IF(ensureAllocated(newDims) != OK, ALLOCATION_FAILED);
  }

  for (u8 i = 0; i < workingIndices->shape.numOfDims; i++) {
    newDims[i] = workingIndices->shape.dims[i];
  }

  for (u8 i = 0; i < workingSource->shape.numOfDims - 1; i++) {
    newDims[workingIndices->shape.numOfDims + i] = workingSource->shape.dims[i + 1];
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < workingSource->shape.numOfDims; i++) {
    sliceSize *= workingSource->shape.dims[i];
  }

  Dim destShape = {.dims = newDims, .numOfDims = newNumDims};
  Tensor *dest = t_Empty(ctx, destShape, workingSource->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  if (isCudaCtx) {
    Result result = runCudaIndexSelect1d(ctx, workingSource->dtype, workingSource->values,
                                         workingIndices->values, workingIndices->dtype,
                                         dest->values, workingIndices->size, sliceSize);
    PANIC_IF(result != OK, result);
  }

  tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (u64 i = 0; i < workingIndices->size; i++) {
    Value idxVal;
    Result result = readTensorValueAtFlatIndex(workingIndices, i, &idxVal);
    PANIC_IF(result != OK, result);

    dim_t idx = 0;
    result = indexValueToDim(idxVal, &idx);
    PANIC_IF(result != OK, result);

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
    PANIC_IF(result != OK, result);
    destOffset += sliceSize;
  }

  return dest;
}

Tensor *IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(rowIndices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(colIndices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->shape.numOfDims < 2, ERR_DIM_MISMATCH);
  PANIC_IF(isIntType(rowIndices) || isIntType(colIndices), ERR_ONLY_INT_TYPE_ALLOWED);
  PANIC_IF(rowIndices->size != colIndices->size, ERR_DIM_MISMATCH);

  Tensor *workingSource = materializeTensorOnContext(ctx, source);
  Tensor *workingRows = materializeTensorOnContext(ctx, rowIndices);
  Tensor *workingCols = materializeTensorOnContext(ctx, colIndices);

  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingRows->shape.numOfDims + workingSource->shape.numOfDims - 2;
  dim_t *newDims = NULL;
  if (newNumDims > 0) {
    newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
    PANIC_IF(ensureAllocated(newDims) != OK, ALLOCATION_FAILED);
  }

  for (u8 i = 0; i < workingRows->shape.numOfDims; i++) {
    newDims[i] = workingRows->shape.dims[i];
  }

  for (u8 i = 0; i < workingSource->shape.numOfDims - 2; i++) {
    newDims[workingRows->shape.numOfDims + i] = workingSource->shape.dims[i + 2];
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < workingSource->shape.numOfDims; i++) {
    sliceSize *= workingSource->shape.dims[i];
  }

  Dim destShape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers};
  Tensor *dest = t_Empty(ctx, destShape, workingSource->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  if (isCudaCtx) {
    Result result = runCudaIndexSelect2d(
        ctx, workingSource->dtype, workingSource->values, workingSource->shape.dims[1],
        workingRows->values, workingRows->dtype, workingCols->values, workingCols->dtype,
        dest->values, workingRows->size, sliceSize);
    PANIC_IF(result != OK, result);
  }

  tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (u64 i = 0; i < workingRows->size; i++) {
    Value rowVal, colVal;
    Result result = readTensorValueAtFlatIndex(workingRows, i, &rowVal);
    PANIC_IF(result != OK, result);

    result = readTensorValueAtFlatIndex(workingCols, i, &colVal);
    PANIC_IF(result != OK, result);

    dim_t rowIdx = 0;
    dim_t colIdx = 0;
    result = indexValueToDim(rowVal, &rowIdx);
    PANIC_IF(result != OK, result);

    result = indexValueToDim(colVal, &colIdx);
    PANIC_IF(result != OK, result);

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
    PANIC_IF(result != OK, result);
    destOffset += sliceSize;
  }

  freeIfContingousCopy(ctx, workingCols);
  freeIfContingousCopy(ctx, workingRows);
  freeIfContingousCopy(ctx, workingSource);

  return dest;
}

Tensor *GetTensorAt(Context *ctx, Tensor *source, dim_t index) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->shape.numOfDims == 0, ERR_DIM_MISMATCH);
  PANIC_IF(!isSameContext(source->context, ctx), ERR_NO_OP);
  PANIC_IF(index >= source->shape.dims[0], ERR_OUT_OF_BOUNDS);

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
    PANIC_IF(result != OK, result);
    scalar.dtype = source->dtype;
    Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
    PANIC_IF(dest == NULL, ALLOCATION_FAILED);
    *dest = singleValueTensor(ctx, scalar);
    PANIC_IF(dest->values == NULL, ERR_OUT_OF_MEMORY);
    return dest;
  }

  // Allocate new dims and copy source dims[1:]
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  PANIC_IF(ensureAllocated(newDims) != OK, ALLOCATION_FAILED);
  for (u8 i = 0; i < newNumDims; i++) {
    newDims[i] = source->shape.dims[i + 1];
  }

  // Copy source multipliers[1:] — do NOT recompute; preserves transposed strides.
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  PANIC_IF(ensureAllocated(newMultipliers) != OK, ALLOCATION_FAILED);
  for (u8 i = 0; i < newNumDims; i++) {
    newMultipliers[i] = source->shape.multipliers[i + 1];
  }

  // Copy remaining per-dim boundary entries [1:] if source was a view.
  Range *boundary = NULL;
  if (source->isView && source->boundary) {
    boundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    PANIC_IF(ensureAllocated(boundary) != OK, ALLOCATION_FAILED);
    for (u8 i = 0; i < newNumDims; i++) {
      boundary[i] = source->boundary[i + 1];
    }
  }

  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);
  *dest = tensorView(source->context, ctx->memory, newValues, source->size / source->shape.dims[0],
                     source->dtype,
                     (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     boundary, false);

  return dest;
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
