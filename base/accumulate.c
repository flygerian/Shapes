#include "result/result.h"
#include "tensor_internal.h"
#include "value.h"
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

static dim_t indexValueToDim(Value *idxVal, Dtype dtype) {
  switch (dtype) {
    case U8: return idxVal->as.u8; ;
    case U16: return idxVal->as.u16;
    case U32: return idxVal->as.u32; return OK;
    case U64: return idxVal->as.u64; return OK;
    case I8: return (dim_t)idxVal->as.i8; return OK;
    case I16: return (dim_t)idxVal->as.i16; return OK;
    case I32: return (dim_t)idxVal->as.i32; return OK;
    case I64: return (dim_t)idxVal->as.i64; return OK;
  }

  PANIC_IF(true, ERR_DTYPE_MISMATCH);
  return -1;
}

static Result indexAccumulate1dCpu(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {

  Tensor *indicesContig = materializeTensorOnContext(ctx, indices);
  Tensor *srcContig = materializeTensorOnContext(ctx, srcGrad);

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  for (u64 i = 0; i < indicesContig->size; i++) {
    Value idxVal;
    VALUE_GET_FROM_ARR(indicesContig->values, i, &idxVal, indicesContig->dtype);

    dim_t idx = indexValueToDim(&idxVal, indicesContig->dtype);
    PANIC_IF(idx >= dest->shape.dims[0], ERR_OUT_OF_BOUNDS);

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
      VALUE_GET_FROM_ARR(srcContig->values, srcBase + j, &srcValue, srcContig->dtype);
      VALUE_BINOP(resultValue, destValue, srcValue, +);
      VALUE_SET(dest->values, destBase + j, resultValue);
    }
  }

  return OK;
}

static Result indexAccumulate2dCpu(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad) {

  Tensor *rowContig = materializeTensorOnContext(ctx, rowIndices);
  Tensor *colContig = materializeTensorOnContext(ctx, colIndices);
  Tensor *srcContig = materializeTensorOnContext(ctx, srcGrad);

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  for (u64 i = 0; i < rowContig->size; i++) {
    Value rowValue, colValue;
    VALUE_GET_FROM_ARR(rowContig->values, i, &rowValue, rowContig->dtype);
    VALUE_GET_FROM_ARR(colContig->values, i, &colValue, colContig->dtype);

    dim_t row = indexValueToDim(&rowValue, rowContig->dtype);
    dim_t col = indexValueToDim(&colValue, colContig->dtype);

    PANIC_IF(row >= dest->shape.dims[0] || col >= dest->shape.dims[1], ERR_OUT_OF_BOUNDS);

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
      VALUE_GET_FROM_ARR(srcContig->values, srcBase + j, &srcValue, srcContig->dtype);
      VALUE_BINOP(resultValue, destValue, srcValue, +);
      VALUE_SET(dest->values, destBase + j, resultValue);
    }
  }

  return OK;
}

static Result sliceAccumulateCpu(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  Tensor *srcGradContig = materializeTensorOnContext(ctx, srcGrad);

  u8 ndims = dest->shape.numOfDims;
  u8 lastDim = ndims - 1;
  u64 innerCount = srcGradContig->shape.dims[lastDim];
  u64 srcStep = srcGradContig->shape.multipliers[lastDim];
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

    u64 srcBase = getContigousIdxFromCoord(srcGradContig, srcCoords);
    u64 dstBase = getContigousIdxFromCoord(dest, dstCoords);
    accumulateStridedByDtype(dest->dtype, dest->values, dstBase, dstStep, srcGradContig->values, srcBase, srcStep, innerCount);

    for (i32 d = (i32)lastDim - 1; d >= 0; d--) {
      srcCoords[d]++;
      dstCoords[d]++;
      if (srcCoords[d] < srcGradContig->shape.dims[d]) {
        break;
      }
      srcCoords[d] = 0;
      dstCoords[d] = (dim_t)ranges[d].start;
    }
  }

  return OK;
}

static Result indexAccumulate1dCuda(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  Tensor *indicesContig = materializeTensorOnContext(ctx, indices);
  Tensor *srcContig = materializeTensorOnContext(ctx, srcGrad);

  tensor_size_t sliceSize = 1;
  for (u8 i = 1; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  Result result = runCudaIndexAccumulate1d(ctx, dest->dtype, dest->values, indicesContig->values, indicesContig->dtype, srcContig->values, indicesContig->size, sliceSize);

  return result;
}

static Result indexAccumulate2dCuda(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  Tensor *rowContig = materializeTensorOnContext(ctx, rowIndices);
  Tensor *colConfig = materializeTensorOnContext(ctx, colIndices);
  Tensor *srcGradContig = materializeTensorOnContext(ctx, srcGrad);

  tensor_size_t sliceSize = 1;
  for (u8 i = 2; i < dest->shape.numOfDims; i++) {
    sliceSize *= dest->shape.dims[i];
  }

  Result result = runCudaIndexAccumulate2d(ctx, dest->dtype, dest->values, dest->shape.dims[1], rowContig->values, rowContig->dtype, colConfig->values, colConfig->dtype, srcGradContig->values,
                                           rowContig->size, sliceSize);

  return result;
}

static Result sliceAccumulateCuda(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  if (!dest->isContigous || dest->isView) {
    return ERR_NO_OP;
  }

  Tensor *srcContig = materializeTensorOnContext(ctx, srcGrad);

  Result result = runCudaSliceAccumulate(ctx, dest->dtype, dest->values, dest->shape.numOfDims, dest->shape.multipliers, ranges, srcContig->values, srcContig->shape.dims, srcContig->shape.numOfDims,
                                         srcContig->size);

  return result;
}

void shapes_IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  PANIC_IF(isInvalidTensor(dest) || isInvalidTensor(indices) || isInvalidTensor(srcGrad), ERR_NULL_TENSOR_PROVIDED);

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  PANIC_IF(result != OK, result);

  PANIC_IF(isIntType(indices), ERR_ONLY_INT_TYPE_ALLOWED);
  PANIC_IF(!isDestOnDispatchDevice(ctx, dest), ERR_NO_OP);

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA:
      result = indexAccumulate1dCuda(ctx, dest, indices, srcGrad);
      PANIC_IF(result != OK, result);
      return;
    case CPU:
    default: result = indexAccumulate1dCpu(ctx, dest, indices, srcGrad); PANIC_IF(result != OK, result);
  }
}

void shapes_IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad) {
  PANIC_IF(isInvalidTensor(dest) || isInvalidTensor(rowIndices) || isInvalidTensor(colIndices) || isInvalidTensor(srcGrad), ERR_NULL_TENSOR_PROVIDED);

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  PANIC_IF(result != OK, result);

  PANIC_IF(isIntType(rowIndices) || isIntType(colIndices), ERR_ONLY_INT_TYPE_ALLOWED);
  PANIC_IF(rowIndices->size != colIndices->size || dest->shape.numOfDims < 2, ERR_DIM_MISMATCH);
  PANIC_IF(!isDestOnDispatchDevice(ctx, dest), ERR_NO_OP);

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA:
      result = indexAccumulate2dCuda(ctx, dest, rowIndices, colIndices, srcGrad);
      PANIC_IF(result != OK, result);
      return;
    case CPU:
    default: result = indexAccumulate2dCpu(ctx, dest, rowIndices, colIndices, srcGrad); PANIC_IF(result != OK, result);
  }
}

void shapes_SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  PANIC_IF(isInvalidTensor(dest) || isInvalidTensor(srcGrad) || ranges == NULL, ERR_NULL_TENSOR_PROVIDED);

  Result result = validateAccumulateTensorArgs(dest, srcGrad);
  PANIC_IF(result != OK, result);

  PANIC_IF(dest->shape.numOfDims != srcGrad->shape.numOfDims, ERR_DIM_MISMATCH);

  u8 ndims = dest->shape.numOfDims;
  PANIC_IF(ndims == 0, ERR_DIM_MISMATCH);

  for (u8 i = 0; i < ndims; i++) {
    PANIC_IF(ranges[i].start > ranges[i].end || ranges[i].end > dest->shape.dims[i], ERR_OUT_OF_BOUNDS);

    u64 span = ranges[i].end - ranges[i].start;
    PANIC_IF((u64)srcGrad->shape.dims[i] != span, ERR_DIM_MISMATCH);
  }

  PANIC_IF(!isDestOnDispatchDevice(ctx, dest), ERR_NO_OP);

  switch (getAccumulateDispatchDevice(ctx)) {
    case CUDA:
      result = sliceAccumulateCuda(ctx, dest, ranges, srcGrad);
      PANIC_IF(result != OK, result);
      return;
    case CPU:
    default: result = sliceAccumulateCpu(ctx, dest, ranges, srcGrad); PANIC_IF(result != OK, result);
  }
}
