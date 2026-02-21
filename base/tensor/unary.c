#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <math.h>
#include <stddef.h>

static Result powValue(Value *v, f32 power) {
  switch (v->dtype) {
    COMPUTE_POW(v, power, F16, f16, pow);
    COMPUTE_POW(v, power, F32, f32, pow);
    COMPUTE_POW(v, power, F64, f64, pow);

    default: return ERR_POW_VALUE_NOT_FLOAT;
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

Result Mean(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_MEAN_VALUE_NOT_FLOAT;
  }

  f64 sum = 0.0;

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    switch (t->dtype) {
      case F16: sum += (f64)val.as.f16; break;
      case F32: sum += (f64)val.as.f32; break;
      case F64: sum += val.as.f64; break;
      default: return ERR_MEAN_VALUE_NOT_FLOAT;
    }
  }

  f64 mean = sum / (f64)t->size;

  dim_t dims[] = {1};
  Tensor *output = t_Zeros(ctx, (Dim){.dims = dims, .numOfDims = 1}, t->dtype);

  switch (t->dtype) {
    case F16: ((f16 *)output->values)[0] = (f16)mean; break;
    case F32: ((f32 *)output->values)[0] = (f32)mean; break;
    case F64: ((f64 *)output->values)[0] = mean; break;
    default: return ERR_MEAN_VALUE_NOT_FLOAT;
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);

  return OK;
}
