#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static Result powValue(Value *v, f32 power) {
  switch (v->dtype) {
    COMPUTE_POW(v, power, F16, f16, pow);
    COMPUTE_POW(v, power, F32, f32, pow);
    COMPUTE_POW(v, power, F64, f64, pow);

    default: return ERR_POW_VALUE_NOT_FLOAT;
  }
}

static Result absValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_ABS(v, I8, i8, abs);
    COMPUTE_ABS(v, I16, i16, abs);
    COMPUTE_ABS(v, I32, i32, abs);
    COMPUTE_ABS(v, I64, i64, llabs);
    COMPUTE_ABS(v, F16, f16, fabsf);
    COMPUTE_ABS(v, F32, f32, fabsf);
    COMPUTE_ABS(v, F64, f64, fabs);

    default: return ERR_ABS_VALUE_NOT_SIGNED;
  }
}

Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_POW_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = powValue(&val, power);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
}

static Result expValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_EXP(v, F16, f16, exp);
    COMPUTE_EXP(v, F32, f32, exp);
    COMPUTE_EXP(v, F64, f64, exp);

    default: return ERR_EXP_VALUE_NOT_FLOAT;
  }
}

static Result negateValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_NEGATE(v, F16, f16);
    COMPUTE_NEGATE(v, F32, f32);
    COMPUTE_NEGATE(v, F64, f64);
    COMPUTE_NEGATE(v, I8, i8);
    COMPUTE_NEGATE(v, I16, i16);
    COMPUTE_NEGATE(v, I32, i32);
    COMPUTE_NEGATE(v, I64, i64);

    default: return ERR_NEGATE_UNSUPPORTED_DTYPE;
  }
}

Result Negate(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype == U8 || t->dtype == U16 || t->dtype == U32 || t->dtype == U64) {
    return ERR_NEGATE_UNSUPPORTED_DTYPE;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = negateValue(&val);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
}

Result Exp(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_EXP_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = expValue(&val);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
}

static Result logValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_LOG(v, F16, f16, log);
    COMPUTE_LOG(v, F32, f32, log);
    COMPUTE_LOG(v, F64, f64, log);

    default: return ERR_LOG_VALUE_NOT_FLOAT;
  }
}

Result Log(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_LOG_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = logValue(&val);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
}

Result Mean(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
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

static void maxValueForType(Value *max, void *values, tensor_size_t idx, Dtype dtype) {
  switch (dtype) {
    case U8:
      if (((u8 *)values)[idx] > max->as.u8)
        max->as.u8 = ((u8 *)values)[idx];
      break;
    case U16:
      if (((u16 *)values)[idx] > max->as.u16)
        max->as.u16 = ((u16 *)values)[idx];
      break;
    case U32:
      if (((u32 *)values)[idx] > max->as.u32)
        max->as.u32 = ((u32 *)values)[idx];
      break;
    case U64:
      if (((u64 *)values)[idx] > max->as.u64)
        max->as.u64 = ((u64 *)values)[idx];
      break;
    case I8:
      if (((i8 *)values)[idx] > max->as.i8)
        max->as.i8 = ((i8 *)values)[idx];
      break;
    case I16:
      if (((i16 *)values)[idx] > max->as.i16)
        max->as.i16 = ((i16 *)values)[idx];
      break;
    case I32:
      if (((i32 *)values)[idx] > max->as.i32)
        max->as.i32 = ((i32 *)values)[idx];
      break;
    case I64:
      if (((i64 *)values)[idx] > max->as.i64)
        max->as.i64 = ((i64 *)values)[idx];
      break;
    case F16:
      if (((f16 *)values)[idx] > max->as.f16)
        max->as.f16 = ((f16 *)values)[idx];
      break;
    case F32:
      if (((f32 *)values)[idx] > max->as.f32)
        max->as.f32 = ((f32 *)values)[idx];
      break;
    case F64:
      if (((f64 *)values)[idx] > max->as.f64)
        max->as.f64 = ((f64 *)values)[idx];
      break;
  }
}

static void setMaxValue(void *values, tensor_size_t idx, Value max, Dtype dtype) {
  switch (dtype) {
    case U8: ((u8 *)values)[idx] = max.as.u8; break;
    case U16: ((u16 *)values)[idx] = max.as.u16; break;
    case U32: ((u32 *)values)[idx] = max.as.u32; break;
    case U64: ((u64 *)values)[idx] = max.as.u64; break;
    case I8: ((i8 *)values)[idx] = max.as.i8; break;
    case I16: ((i16 *)values)[idx] = max.as.i16; break;
    case I32: ((i32 *)values)[idx] = max.as.i32; break;
    case I64: ((i64 *)values)[idx] = max.as.i64; break;
    case F16: ((f16 *)values)[idx] = max.as.f16; break;
    case F32: ((f32 *)values)[idx] = max.as.f32; break;
    case F64: ((f64 *)values)[idx] = max.as.f64; break;
  }
}

static bool valueGreaterForType(Value candidate, Value currentMax, Dtype dtype) {
  switch (dtype) {
    case U8: return candidate.as.u8 > currentMax.as.u8;
    case U16: return candidate.as.u16 > currentMax.as.u16;
    case U32: return candidate.as.u32 > currentMax.as.u32;
    case U64: return candidate.as.u64 > currentMax.as.u64;
    case I8: return candidate.as.i8 > currentMax.as.i8;
    case I16: return candidate.as.i16 > currentMax.as.i16;
    case I32: return candidate.as.i32 > currentMax.as.i32;
    case I64: return candidate.as.i64 > currentMax.as.i64;
    case F16: return candidate.as.f16 > currentMax.as.f16;
    case F32: return candidate.as.f32 > currentMax.as.f32;
    case F64: return candidate.as.f64 > currentMax.as.f64;
    default: return false;
  }
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
        maxValueForType(&max, workingTensor->values, sourceIdx, workingTensor->dtype);
      }

      tensor_size_t destIdx = outer * numAfterDim + inner;
      setMaxValue(dest->values, destIdx, max, workingTensor->dtype);
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
        if (valueGreaterForType(candidate, max, workingTensor->dtype)) {
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

Result Abs(Context *ctx, Tensor *t, Tensor *dest) {

  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->size == 0) {
    return ERR_NO_OP;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  *dest = (Tensor) {
    .dtype = t->dtype, 
      .size = t->size, 
      .isView = false,
      .isContigous = true,
      .shape = {
        .dims = allocate(ctx->memory, t->shape.numOfDims * sizeof(dim_t)),
        .numOfDims = t->shape.numOfDims,
        .multipliers = t->shape.multipliers
      },
      .values = allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * workingTensor->size)
  };

  memcpy(dest->shape.dims, t->shape.dims, t->shape.numOfDims * sizeof(dim_t));
  memcpy(dest->shape.multipliers, t->shape.multipliers, t->shape.numOfDims * sizeof(multiplier_t));

  for (size_t i = 0; i < workingTensor->size; i++) {
    Value v;
    VALUE_GET_FROM_ARR(workingTensor->values, i, &v, workingTensor->dtype);

    Result result = absValue(&v);
    if (result != OK) {
      return result;
    }

    VALUE_SET(dest->values, i, v);
  }

  if (!t->isContigous) {
    FreeTensor(ctx, workingTensor);
  }

  return OK;
}
