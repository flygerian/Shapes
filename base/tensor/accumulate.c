#include "common.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <stddef.h>

static DeviceType getAccumulateDispatchDevice(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return CPU;
  }

  return ctx->device->type;
}

static DeviceType getTensorDeviceType(Tensor *t) {
  if (t == NULL || t->context == NULL || t->context->device == NULL) {
    return CPU;
  }

  return t->context->device->type;
}

static bool isDestOnDispatchDevice(Context *ctx, Tensor *dest) {
  return getAccumulateDispatchDevice(ctx) == getTensorDeviceType(dest);
}

static Result validateAccumulateTensorArgs(Tensor *dest, Tensor *srcGrad) {
  if (dest->dtype != srcGrad->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  return OK;
}

static Result indexValueToDim(Value *idxVal, Dtype dtype, dim_t *idx) {
  switch (dtype) {
    case U8: *idx = idxVal->as.u8; return OK;
    case U16: *idx = idxVal->as.u16; return OK;
    case U32: *idx = idxVal->as.u32; return OK;
    case U64: *idx = idxVal->as.u64; return OK;
    case I8: *idx = (dim_t)idxVal->as.i8; return OK;
    case I16: *idx = (dim_t)idxVal->as.i16; return OK;
    case I32: *idx = (dim_t)idxVal->as.i32; return OK;
    case I64: *idx = (dim_t)idxVal->as.i64; return OK;
    default: return ERR_DTYPE_MISMATCH;
  }
}

static Result indexAccumulate1dCpu(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  TensorArg indicesArg = {0};
  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, indices, true, &indicesArg);
  if (result != OK) {
    return result;
  }

  result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    releaseTensorArg(ctx, &indicesArg);
    return result;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  for (u64 i = 0; i < indicesArg.tensor->size; i++) {
    Value idxVal;
    VALUE_GET_FROM_ARR(indicesArg.tensor->values, i, &idxVal, indicesArg.tensor->dtype);

    dim_t idx = 0;
    result = indexValueToDim(&idxVal, indicesArg.tensor->dtype, &idx);
    if (result != OK) {
      releaseTensorArg(ctx, &indicesArg);
      releaseTensorArg(ctx, &srcArg);
      return result;
    }

    if (idx >= dest->shape.dims[0]) {
      releaseTensorArg(ctx, &indicesArg);
      releaseTensorArg(ctx, &srcArg);
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
      Value destValue, srcValue, resultValue;
      VALUE_GET_FROM_ARR(dest->values, destBase + j, &destValue, dest->dtype);
      VALUE_GET_FROM_ARR(srcArg.tensor->values, srcBase + j, &srcValue, srcArg.tensor->dtype);
      VALUE_BINOP(resultValue, destValue, srcValue, +);
      VALUE_SET(dest->values, destBase + j, resultValue);
    }
  }

  releaseTensorArg(ctx, &indicesArg);
  releaseTensorArg(ctx, &srcArg);
  return OK;
}

static Result indexAccumulate2dCpu(Context *ctx, Tensor *dest, Tensor *rowIndices,
                                   Tensor *colIndices, Tensor *srcGrad) {
  TensorArg rowArg = {0};
  TensorArg colArg = {0};
  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, rowIndices, true, &rowArg);
  if (result != OK) {
    return result;
  }
  result = materializeTensorOnContext(ctx, colIndices, true, &colArg);
  if (result != OK) {
    releaseTensorArg(ctx, &rowArg);
    return result;
  }
  result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    releaseTensorArg(ctx, &rowArg);
    releaseTensorArg(ctx, &colArg);
    return result;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  for (u64 i = 0; i < rowArg.tensor->size; i++) {
    Value rowValue, colValue;
    VALUE_GET_FROM_ARR(rowArg.tensor->values, i, &rowValue, rowArg.tensor->dtype);
    VALUE_GET_FROM_ARR(colArg.tensor->values, i, &colValue, colArg.tensor->dtype);

    dim_t row = 0;
    dim_t col = 0;
    result = indexValueToDim(&rowValue, rowArg.tensor->dtype, &row);
    if (result != OK) {
      releaseTensorArg(ctx, &rowArg);
      releaseTensorArg(ctx, &colArg);
      releaseTensorArg(ctx, &srcArg);
      return result;
    }
    result = indexValueToDim(&colValue, colArg.tensor->dtype, &col);
    if (result != OK) {
      releaseTensorArg(ctx, &rowArg);
      releaseTensorArg(ctx, &colArg);
      releaseTensorArg(ctx, &srcArg);
      return result;
    }

    if (row >= dest->shape.dims[0] || col >= dest->shape.dims[1]) {
      releaseTensorArg(ctx, &rowArg);
      releaseTensorArg(ctx, &colArg);
      releaseTensorArg(ctx, &srcArg);
      return ERR_OUT_OF_BOUNDS;
    }

    dim_t destCoords[dest->shape.numOfDims];
    destCoords[0] = row;
    destCoords[1] = col;
    for (u8 d = 2; d < dest->shape.numOfDims; d++) {
      destCoords[d] = 0;
    }

    u64 destBase = getContigousIdxFromCoord(dest, destCoords);
    u64 srcBase = i * sliceSize;
    for (tensor_size_t j = 0; j < sliceSize; j++) {
      Value destValue, srcValue, resultValue;
      VALUE_GET_FROM_ARR(dest->values, destBase + j, &destValue, dest->dtype);
      VALUE_GET_FROM_ARR(srcArg.tensor->values, srcBase + j, &srcValue, srcArg.tensor->dtype);
      VALUE_BINOP(resultValue, destValue, srcValue, +);
      VALUE_SET(dest->values, destBase + j, resultValue);
    }
  }

  releaseTensorArg(ctx, &rowArg);
  releaseTensorArg(ctx, &colArg);
  releaseTensorArg(ctx, &srcArg);
  return OK;
}

static Result sliceAccumulateCpu(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    return result;
  }

  u8 ndims = dest->shape.numOfDims;
  u8 lastDim = ndims - 1;
  u64 innerCount = srcArg.tensor->shape.dims[lastDim];
  u64 srcStep = srcArg.tensor->shape.multipliers[lastDim];
  u64 dstStep = dest->shape.multipliers[lastDim];

  dim_t srcCoords[ndims];
  dim_t dstCoords[ndims];
  for (u8 i = 0; i < ndims; i++) {
    srcCoords[i] = 0;
    dstCoords[i] = (dim_t)ranges[i].start;
  }

  u64 outerCount = 1;
  for (u8 i = 0; i < lastDim; i++) {
    outerCount *= srcArg.tensor->shape.dims[i];
  }

  if (innerCount == 0 || outerCount == 0) {
    releaseTensorArg(ctx, &srcArg);
    return OK;
  }

  for (u64 outer = 0; outer < outerCount; outer++) {
    srcCoords[lastDim] = 0;
    dstCoords[lastDim] = (dim_t)ranges[lastDim].start;

    u64 srcBase = getContigousIdxFromCoord(srcArg.tensor, srcCoords);
    u64 dstBase = getContigousIdxFromCoord(dest, dstCoords);
    accumulateStridedByDtype(dest->dtype, dest->values, dstBase, dstStep, srcArg.tensor->values,
                             srcBase, srcStep, innerCount);

    for (i32 d = (i32)lastDim - 1; d >= 0; d--) {
      srcCoords[d]++;
      dstCoords[d]++;
      if (srcCoords[d] < srcArg.tensor->shape.dims[d]) {
        break;
      }
      srcCoords[d] = 0;
      dstCoords[d] = (dim_t)ranges[d].start;
    }
  }

  releaseTensorArg(ctx, &srcArg);
  return OK;
}

static Result indexAccumulate1dCuda(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  TensorArg indicesArg = {0};
  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, indices, true, &indicesArg);
  if (result != OK) {
    return result;
  }

  result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    releaseTensorArg(ctx, &indicesArg);
    return result;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  result = runCudaIndexAccumulate1d(ctx, dest->dtype, dest->values, indicesArg.tensor->values,
                                    indicesArg.tensor->dtype, srcArg.tensor->values,
                                    indicesArg.tensor->size, sliceSize);

  releaseTensorArg(ctx, &indicesArg);
  releaseTensorArg(ctx, &srcArg);
  return result;
}

static Result indexAccumulate2dCuda(Context *ctx, Tensor *dest, Tensor *rowIndices,
                                    Tensor *colIndices, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  TensorArg rowArg = {0};
  TensorArg colArg = {0};
  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, rowIndices, true, &rowArg);
  if (result != OK) {
    return result;
  }
  result = materializeTensorOnContext(ctx, colIndices, true, &colArg);
  if (result != OK) {
    releaseTensorArg(ctx, &rowArg);
    return result;
  }
  result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    releaseTensorArg(ctx, &rowArg);
    releaseTensorArg(ctx, &colArg);
    return result;
  }

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  result = runCudaIndexAccumulate2d(ctx, dest->dtype, dest->values, dest->shape.dims[1],
                                    rowArg.tensor->values, rowArg.tensor->dtype, colArg.tensor->values,
                                    colArg.tensor->dtype, srcArg.tensor->values, rowArg.tensor->size,
                                    sliceSize);

  releaseTensorArg(ctx, &rowArg);
  releaseTensorArg(ctx, &colArg);
  releaseTensorArg(ctx, &srcArg);
  return result;
}

static Result sliceAccumulateCuda(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  TensorArg srcArg = {0};
  Result result = materializeTensorOnContext(ctx, srcGrad, true, &srcArg);
  if (result != OK) {
    return result;
  }

  result = runCudaSliceAccumulate(ctx, dest->dtype, dest->values, dest->shape.numOfDims,
                                  dest->shape.multipliers, ranges, srcArg.tensor->values,
                                  srcArg.tensor->shape.dims, srcArg.tensor->shape.numOfDims,
                                  srcArg.tensor->size);

  releaseTensorArg(ctx, &srcArg);
  return result;
}

Result IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  if (isInvalidTensor(dest) || isInvalidTensor(indices) || isInvalidTensor(srcGrad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  if (result != OK) {
    return result;
  }

  if (isIntType(indices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  if (!isDestOnDispatchDevice(ctx, dest)) {
    return ERR_NO_OP;
  }

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA: return indexAccumulate1dCuda(ctx, dest, indices, srcGrad);
    case CPU:
    default: return indexAccumulate1dCpu(ctx, dest, indices, srcGrad);
  }
}

Result IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *srcGrad) {
  if (isInvalidTensor(dest) || isInvalidTensor(rowIndices) || isInvalidTensor(colIndices) ||
      isInvalidTensor(srcGrad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  if (result != OK) {
    return result;
  }

  if (isIntType(rowIndices) || isIntType(colIndices)) {
    return ERR_ONLY_INT_TYPE_ALLOWED;
  }

  if (rowIndices->size != colIndices->size || dest->shape.numOfDims < 2) {
    return ERR_DIM_MISMATCH;
  }

  if (!isDestOnDispatchDevice(ctx, dest)) {
    return ERR_NO_OP;
  }

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA: return indexAccumulate2dCuda(ctx, dest, rowIndices, colIndices, srcGrad);
    case CPU:
    default: return indexAccumulate2dCpu(ctx, dest, rowIndices, colIndices, srcGrad);
  }
}

Result SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  if (isInvalidTensor(dest) || isInvalidTensor(srcGrad) || ranges == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  if (result != OK) {
    return result;
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

  if (!isDestOnDispatchDevice(ctx, dest)) {
    return ERR_NO_OP;
  }

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA: return sliceAccumulateCuda(ctx, dest, ranges, srcGrad);
    case CPU:
    default: return sliceAccumulateCpu(ctx, dest, ranges, srcGrad);
  }
}
