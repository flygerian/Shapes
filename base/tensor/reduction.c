#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "common.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor_internal.h"
#include "value.h"
#include "../memory.h"

static Value *contigousSum(Memory *m, void *postion, tensor_size_t limit, Dtype dtype) {
  tensor_size_t x = 0;
  Value sums[MAX_PARALLEL_SUMS] = {};

  for (u8 i = 0; i < MAX_PARALLEL_SUMS; i++) {
    sums[i] = VALUE(dtype, 0);
  }

  while (x < limit) {
    u8 numComputations = 0;
    for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
      if (x + y >= limit) {
        break;
      }

      Value val;
      VALUE_GET_FROM_ARR(postion, x + y, &val, dtype);
      VALUE_BINOP(sums[y], sums[y], val, +);
      numComputations++;
    }
    x += numComputations;
  }

  Value *sum = allocate(m, sizeof(Value));
  *sum = VALUE(dtype, 0);
  for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
    VALUE_BINOP(*sum, *sum, sums[y], +);
  }

  return sum;
}

static Value *sumTensor(Context *ctx, Tensor *t) {
  return contigousSum(ctx->memory, t->values, t->size, t->dtype);
}

Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_SUM_DIM_OUT_OF_BOUNDS;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    return numBeforeResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest =
      (Tensor){.dtype = workingTensor->dtype,
               .isContigous = true,
               .isView = false,
               .size = (resultSize),
               .values = allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * resultSize),
               .boundary = NULL,
               };

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      Value acc = VALUE(workingTensor->dtype, 0);
      for (dim_t r = 0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;

        Value v;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &v, workingTensor->dtype);

        VALUE_BINOP(acc, acc, v, +);
      }
      VALUE_UNBOX(acc, dest->values +
                           (outer * numAfterDim + inner) * getBytesForDtype(workingTensor->dtype));
    }
  }

  dest->shape = (Dim){
      .numOfDims = workingTensor->shape.numOfDims,
      .dims = allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims),
      .multipliers = allocate(ctx->memory, sizeof(multiplier_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims,
         sizeof(dim_t) * workingTensor->shape.numOfDims);
  dest->shape.dims[dim] = 1;

  // Recalculate multipliers for the new shape
  calculateNumValuesAndMultipliers(dest->shape, dest->shape.multipliers);

  if (!t->isContigous) {
    Result freeRes = FreeTensor(ctx, workingTensor);
    if (freeRes != OK) {
      return freeRes;
    }
  }

  return OK;
}

Result Mean(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_MEAN_VALUE_NOT_FLOAT;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  Value *tensorSum = sumTensor(ctx, workingTensor);
  Value size = VALUE(workingTensor->dtype, workingTensor->size);

  Value avg = VALUE(t->dtype, 0);
  VALUE_BINOP(avg, *tensorSum, size, /);

  *dest = singleValueTensor(ctx, avg);

  if (!t->isContigous) {
    Result result = FreeTensor(ctx, workingTensor);
    if (result != OK) {
      return result;
    }
  }

  freeAlloc(ctx->memory, tensorSum);
  return OK;
}

Result MeanDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_MEAN_VALUE_NOT_FLOAT;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    if (!t->isContigous) {
      FreeTensor(ctx, workingTensor);
    }

    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    if (!t->isContigous) {
      FreeTensor(ctx, workingTensor);
    }
    return numAfterResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest =
      (Tensor){.dtype = workingTensor->dtype,
               .isContigous = true,
               .isView = false,
               .size = resultSize,
               .values = allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * resultSize),
               .boundary = NULL,
      };

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      f64 sum = 0.0;
      for (dim_t r = 0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value v;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &v, workingTensor->dtype);
        switch (workingTensor->dtype) {
          case F16: sum += (f64)v.as.f16; break;
          case F32: sum += (f64)v.as.f32; break;
          case F64: sum += v.as.f64; break;
          default: break;
        }
      }
      f64 mean = sum / (f64)reduce;
      tensor_size_t destIdx = outer * numAfterDim + inner;
      switch (workingTensor->dtype) {
        case F16: ((f16 *)dest->values)[destIdx] = (f16)mean; break;
        case F32: ((f32 *)dest->values)[destIdx] = (f32)mean; break;
        case F64: ((f64 *)dest->values)[destIdx] = mean; break;
        default: break;
      }
    }
  }

  dest->shape = (Dim){
      .numOfDims = workingTensor->shape.numOfDims,
      .dims = allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims),
      .multipliers = allocate(ctx->memory, sizeof(multiplier_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims,
         sizeof(dim_t) * workingTensor->shape.numOfDims);
  dest->shape.dims[dim] = 1;
  calculateNumValuesAndMultipliers(dest->shape, dest->shape.multipliers);

  if (!t->isContigous) {
    FreeTensor(ctx, workingTensor);
  }

  return OK;
}

static inline void cleanupStd(Context *ctx, bool isCopiedToContigous, Tensor *workingTensor, Tensor *mean) {
  if (isCopiedToContigous) {
    FreeTensor(ctx, workingTensor);
  }

  freeTensorBuffers(ctx, mean);
}

Result Std(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->size < 2) {
    return ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES;
  }

  if (isFloatType(t)) {
    return ERR_STD_NOT_FLOAT_TYPE;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  Tensor mean;
  Result meanRes = Mean(ctx, workingTensor, &mean);
  if (meanRes != OK) {
    if (!t->isContigous) {
      FreeTensor(ctx, workingTensor);
    }
    return meanRes;
  }

  // Store the value of  Σ(xᵢ - x̄)²
  Value deviationSquaredSum = VALUE(workingTensor->dtype, 0);

  for (tensor_size_t i = 0; i < workingTensor->size; i++) {
    Value curVal = VALUE(workingTensor->dtype, 0);
    VALUE_GET_FROM_ARR(workingTensor->values, i, &curVal, workingTensor->dtype);

    Value meanVal = VALUE(F64, 0);
    VALUE_GET_FROM_ARR(mean.values, 0, &meanVal, mean.dtype);

    // (xᵢ - x̄)
    Value diviation = VALUE(F64, 0);
    VALUE_BINOP(diviation, curVal, meanVal, -);

    // (xᵢ - x̄)²
    Result res = powValue(&diviation, 2);
    if (res != OK) {
      cleanupStd(ctx, !t->isContigous, workingTensor, &mean);
      return res;
    }

    VALUE_BINOP(deviationSquaredSum, deviationSquaredSum, diviation, +);
  }

  Value v;
  Value deg = VALUE(workingTensor->dtype, workingTensor->size - 1);
  VALUE_BINOP(v, deviationSquaredSum, deg, /);

  Result res = sqrtValue(&v);
  if (res != OK) {
    cleanupStd(ctx, !t->isContigous, workingTensor, &mean);
    return res;
  }

  *dest = singleValueTensor(ctx, v);

  cleanupStd(ctx, !t->isContigous, workingTensor, &mean);

  return OK;
}

Result Max(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (t->size == 0) {
    return ERR_NO_OP;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    if (!t->isContigous)
      FreeTensor(ctx, workingTensor);
    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    if (!t->isContigous)
      FreeTensor(ctx, workingTensor);
    return numAfterResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest =
      (Tensor){.dtype = workingTensor->dtype,
               .isContigous = true,
               .isView = false,
               .size = resultSize,
               .values = allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * resultSize),
               .boundary = NULL,
      };

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      Value max;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &max, workingTensor->dtype);

      for (dim_t r = 1; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value candidate;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &candidate, workingTensor->dtype);
        if (VALUE_CMP(candidate, max, >, workingTensor->dtype)) {
          max = candidate;
        }
      }

      tensor_size_t destIdx = outer * numAfterDim + inner;
      VALUE_SET(dest->values, destIdx, max);
    }
  }

  dest->shape = (Dim){
      .numOfDims = workingTensor->shape.numOfDims,
      .dims = allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims),
      .multipliers = allocate(ctx->memory, sizeof(multiplier_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims,
         sizeof(dim_t) * workingTensor->shape.numOfDims);
  dest->shape.dims[dim] = 1;
  calculateNumValuesAndMultipliers(dest->shape, dest->shape.multipliers);

  if (!t->isContigous) {
    FreeTensor(ctx, workingTensor);
  }

  return OK;
}

Result ArgMax(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (t->size == 0) {
    return ERR_NO_OP;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    if (!t->isContigous)
      FreeTensor(ctx, workingTensor);
    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    if (!t->isContigous)
      FreeTensor(ctx, workingTensor);
    return numAfterResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest = (Tensor){.dtype = I64,
                   .isContigous = true,
                   .isView = false,
                   .size = resultSize,
                   .values = allocate(ctx->memory, sizeof(i64) * resultSize),
                   .boundary = NULL,
                   };

  i64 *destValues = (i64 *)dest->values;
  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      Value max;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &max, workingTensor->dtype);

      dim_t argmax = 0;
      for (dim_t r = 1; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value candidate;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &candidate, workingTensor->dtype);
        if (VALUE_CMP(candidate, max, >, workingTensor->dtype)) {
          max = candidate;
          argmax = r;
        }
      }

      tensor_size_t destIdx = outer * numAfterDim + inner;
      destValues[destIdx] = (i64)argmax;
    }
  }

  dest->shape = (Dim){
      .numOfDims = workingTensor->shape.numOfDims,
      .dims = allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims),
      .multipliers = allocate(ctx->memory, sizeof(multiplier_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims,
         sizeof(dim_t) * workingTensor->shape.numOfDims);
  dest->shape.dims[dim] = 1;
  calculateNumValuesAndMultipliers(dest->shape, dest->shape.multipliers);

  if (!t->isContigous) {
    FreeTensor(ctx, workingTensor);
  }

  return OK;
}
