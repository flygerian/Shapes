#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

Result powValue(Value *v, f32 power) {
  switch (v->dtype) {
    COMPUTE_POW(v, power, F16, f16, pow);
    COMPUTE_POW(v, power, F32, f32, pow);
    COMPUTE_POW(v, power, F64, f64, pow);

    default: return ERR_POW_VALUE_NOT_FLOAT;
  }
}

Result sqrtValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_SQRT(v, F16, f16, sqrt);
    COMPUTE_SQRT(v, F32, f32, sqrt);
    COMPUTE_SQRT(v, F64, f64, sqrt);

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

Result Tanh(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_TANH_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  if (t->dtype == F64) {
    f64 *in = t->values;
    f64 *out = output->values;
    for (tensor_size_t i = 0; i < t->size; i++) {
      out[i] = tanh(in[i]);
    }
  } else {
    f32 *in = t->values;
    f32 *out = output->values;
    for (tensor_size_t i = 0; i < t->size; i++) {
      out[i] = tanhf(in[i]);
    }
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
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

  *dest = (Tensor){.dtype = t->dtype,
                   .size = t->size,
                   .isView = false,
                   .isContigous = true,
                   .shape = {.dims = allocate(ctx->memory, t->shape.numOfDims * sizeof(dim_t)),
                             .numOfDims = t->shape.numOfDims,
                             .multipliers = t->shape.multipliers},
                   .values = allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) *
                                                       workingTensor->size)};

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
