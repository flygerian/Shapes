#include "olib.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "shapes_internal.h"
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

static bool isOutOfBounds(shapes_Tensor *t, shapes_Dim dim) {
  for (u8 i = 0; i < dim.numOfDims; i++) {
    if (dim.dims[i] >= t->shape.dims[i]) {
      return true;
    }
  }
  return false;
}

shapes_Value *shapes_GetAt(shapes_Tensor *t, shapes_Dim dim) {
  PANIC_IF(dim.numOfDims != t->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(isOutOfBounds(t, dim), ERR_OUT_OF_BOUNDS);

  u64 idx = getContigousIdxFromCoord(t, dim.dims);

  shapes_Value *result = olib_Allocate(t->context->memory, sizeof(shapes_Value));
  size_t valueBytes = getBytesForDtype(t->dtype);

  byte *values;
  if (t->context->device != NULL && t->context->device->type != CPU) {
    size_t size = t->size * valueBytes;
    values = olib_Allocate(t->context->memory, size);
    shapes_CopyBetweenDevices(t->context->device->type, CPU, t->values, values, size);
  } else {
    values = t->values;
  }

  memcpy(&result->as, values + idx * valueBytes, valueBytes);
  result->dtype = t->dtype;
  return result;
}

shapes_Tensor shapes_IndexWithTensor(shapes_Context *ctx, shapes_Tensor *source, shapes_Tensor *indices) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(indices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->shape.numOfDims == 0, ERR_ZERO_DIM_TENSOR_ADVANCED_INDEXING);
  PANIC_IF(isIntType(indices), ERR_ONLY_INT_TYPE_ALLOWED);

  shapes_Tensor *workingSource = materializeTensorOnContext(ctx, source);
  shapes_Tensor *workingIndices = materializeTensorOnContext(ctx, indices);

  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingSource->shape.numOfDims - 1 + workingIndices->shape.numOfDims;
  shapes_dim_t *newDims = NULL;
  if (newNumDims > 0) {
    newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newNumDims);
  }

  for (RANGE(i, workingIndices->shape.numOfDims)) {
    newDims[i] = workingIndices->shape.dims[i];
  }

  for (RANGE(i, workingIndices->shape.numOfDims)) {
    newDims[workingIndices->shape.numOfDims + i] = workingSource->shape.dims[i + 1];
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  shapes_tensor_size_t sliceSize = 1;
  for (RANGE(i, workingIndices->shape.numOfDims)) {
    sliceSize *= workingSource->shape.dims[i];
  }

  shapes_Dim destShape = {.dims = newDims, .numOfDims = newNumDims};
  shapes_Tensor dest = t_Empty(ctx, destShape, workingSource->dtype);
  if (isCudaCtx) {
    Result result = shapescuda_IndexSelect1d(workingSource->dtype, workingSource->values, workingIndices->values, workingIndices->dtype, dest.values, workingIndices->size, sliceSize);
    PANIC_IF(result != OK, result);
  }

  shapes_tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (RANGE(i, workingIndices->size)) {
    shapes_Value idxVal;
    Result result = readTensorValueAtFlatIndex(workingIndices, i, &idxVal);
    PANIC_IF(result != OK, result);

    shapes_dim_t idx = indexValueToDim(idxVal, workingIndices->dtype);
    PANIC_IF(result != OK, result);

    shapes_dim_t srcCoords[workingSource->shape.numOfDims];
    srcCoords[0] = idx;
    for (RANGE(d, workingSource->shape.numOfDims)) {
      srcCoords[d] = 0;
    }
    u64 srcOffset = getContigousIdxFromCoord(workingSource, srcCoords);

    memcpy((char *)dest.values + destOffset * bytesPerElem, (char *)workingSource->values + srcOffset * bytesPerElem, sliceSize * bytesPerElem);
    destOffset += sliceSize;
  }

  return dest;
}

shapes_Tensor shapes_IndexWithTensor2d(shapes_Context *ctx, shapes_Tensor *source, shapes_Tensor *rowIndices, shapes_Tensor *colIndices) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(rowIndices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(isInvalidTensor(colIndices), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->shape.numOfDims < 2, ERR_DIM_MISMATCH);
  PANIC_IF(isIntType(rowIndices) || isIntType(colIndices), ERR_ONLY_INT_TYPE_ALLOWED);
  PANIC_IF(rowIndices->size != colIndices->size, ERR_DIM_MISMATCH);

  shapes_Tensor *workingSource = materializeTensorOnContext(ctx, source);
  shapes_Tensor *workingRows = materializeTensorOnContext(ctx, rowIndices);
  shapes_Tensor *workingCols = materializeTensorOnContext(ctx, colIndices);

  bool isCudaCtx = ctx->device != NULL && ctx->device->type == CUDA;

  u8 newNumDims = workingRows->shape.numOfDims + workingSource->shape.numOfDims - 2;
  shapes_dim_t *newDims = NULL;
  if (newNumDims > 0) {
    newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newNumDims);
  }

  for (RANGE(i, workingRows->shape.numOfDims)) {
    newDims[i] = workingRows->shape.dims[i];
  }

  for (RANGE(i, workingSource->shape.numOfDims)) {
    newDims[workingRows->shape.numOfDims + i] = workingSource->shape.dims[i + 2];
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  shapes_tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < workingSource->shape.numOfDims; i++) {
    sliceSize *= workingSource->shape.dims[i];
  }

  shapes_Dim destShape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers};
  shapes_Tensor dest = t_Empty(ctx, destShape, workingSource->dtype);
  if (isCudaCtx) {
    Result result = shapescuda_IndexSelect2d(workingSource->dtype, workingSource->values, workingSource->shape.dims[1], workingRows->values, workingRows->dtype, workingCols->values,
                                         workingCols->dtype, dest.values, workingRows->size, sliceSize);
    PANIC_IF(result != OK, result);
  }

  shapes_tensor_size_t destOffset = 0;
  size_t bytesPerElem = getBytesForDtype(workingSource->dtype);
  for (u64 i = 0; i < workingRows->size; i++) {
    shapes_Value rowVal, colVal;
    Result result = readTensorValueAtFlatIndex(workingRows, i, &rowVal);
    PANIC_IF(result != OK, result);

    result = readTensorValueAtFlatIndex(workingCols, i, &colVal);
    PANIC_IF(result != OK, result);

    shapes_dim_t rowIdx = indexValueToDim(rowVal, workingRows->dtype);
    shapes_dim_t colIdx = indexValueToDim(colVal, workingCols->dtype);

    shapes_dim_t srcCoords[workingSource->shape.numOfDims];
    srcCoords[0] = rowIdx;
    srcCoords[1] = colIdx;

    for (u8 d = 2; d < workingSource->shape.numOfDims; d++) {
      srcCoords[d] = 0;
    }
    u64 srcOffset = getContigousIdxFromCoord(workingSource, srcCoords);

    memcpy((char *)dest.values + destOffset * bytesPerElem, (char *)workingSource->values + srcOffset * bytesPerElem, sliceSize * bytesPerElem);
    destOffset += sliceSize;
  }

  return dest;
}

Result shapes_AssignValueAt(shapes_Context *ctx, shapes_Tensor *t, shapes_Dim dim, shapes_Value value) {
  (void)ctx;
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
