#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <stdio.h>
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

static Result logValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_LOG(v, F16, f16, log);
    COMPUTE_LOG(v, F32, f32, log);
    COMPUTE_LOG(v, F64, f64, log);

    default: return ERR_LOG_VALUE_NOT_FLOAT;
  }
}

static DeviceType getUnaryDispatchDevice(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return CPU;
  }

  return ctx->device->type;
}

static const char *unaryDeviceTypeName(DeviceType type) {
  switch (type) {
    case CPU: return "CPU";
    case CUDA: return "CUDA";
    default: return "UNKNOWN";
  }
}

static const char *unaryContextDeviceName(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return "CPU(default)";
  }

  return unaryDeviceTypeName(ctx->device->type);
}

static const char *unaryDtypeName(Dtype dtype) {
  switch (dtype) {
    case F16: return "F16";
    case F32: return "F32";
    case F64: return "F64";
    case U8: return "U8";
    case U16: return "U16";
    case U32: return "U32";
    case U64: return "U64";
    case I8: return "I8";
    case I16: return "I16";
    case I32: return "I32";
    case I64: return "I64";
    case BOOL: return "BOOL";
    default: return "UNKNOWN";
  }
}

static const char *unaryResultName(Result result) {
  switch (result) {
    case OK: return "OK";
    case ERR_NO_OP: return "ERR_NO_OP";
    case ERR_NULL_TENSOR_PROVIDED: return "ERR_NULL_TENSOR_PROVIDED";
    case ERR_RELU_VALUE_NOT_FLOAT: return "ERR_RELU_VALUE_NOT_FLOAT";
    default: return "UNKNOWN_RESULT";
  }
}

static bool shouldLogRelu(void) {
  const char *value = getenv("SHAPES_LOG_RELU");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void logReluTensorState(const char *phase, Context *ctx, Tensor *t, Result result) {
  if (!shouldLogRelu()) {
    return;
  }

  Context *tensorCtx = t != NULL ? t->context : NULL;
  const char *dtypeName = t != NULL ? unaryDtypeName(t->dtype) : "NULL";
  unsigned long long size = t != NULL ? (unsigned long long)t->size : 0ULL;
  int contiguous = t != NULL && t->isContigous ? 1 : 0;

  fprintf(stderr,
          "[Relu] phase=%s ctx=%p ctxDevice=%s tensorCtx=%p tensorDevice=%s dtype=%s "
          "size=%llu contiguous=%d result=%s(%d)\n",
          phase, (void *)ctx, unaryContextDeviceName(ctx), (void *)tensorCtx,
          unaryContextDeviceName(tensorCtx), dtypeName, size, contiguous, unaryResultName(result),
          (int)result);
}

static Result validateUnaryOpTensor(Tensor *t) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  return OK;
}

static Result validatePowTensor(Tensor *t) {
  Result result = validateUnaryOpTensor(t);
  if (result != OK) {
    return result;
  }

  switch (t->dtype) {
    case F16:
    case F32:
    case F64: return OK;
    default: return ERR_POW_VALUE_NOT_FLOAT;
  }
}

static Result validateFloatUnaryTensor(Tensor *t, Result invalidResult) {
  Result result = validateUnaryOpTensor(t);
  if (result != OK) {
    return result;
  }

  switch (t->dtype) {
    case F16:
    case F32:
    case F64: return OK;
    default: return invalidResult;
  }
}

static Result validateNegateTensor(Tensor *t) {
  Result result = validateUnaryOpTensor(t);
  if (result != OK) {
    return result;
  }

  switch (t->dtype) {
    case U8:
    case U16:
    case U32:
    case U64: return ERR_NEGATE_UNSUPPORTED_DTYPE;
    default: return OK;
  }
}

static Result validateAbsTensor(Tensor *t) {
  Result result = validateUnaryOpTensor(t);
  if (result != OK) {
    return result;
  }

  if (t->size == 0) {
    return ERR_NO_OP;
  }

  switch (t->dtype) {
    case I8:
    case I16:
    case I32:
    case I64:
    case F16:
    case F32:
    case F64: return OK;
    default: return ERR_ABS_VALUE_NOT_SIGNED;
  }
}

static Result applyUnaryCpuValue(Value *value, UnaryOpType opType, f32 param) {
  switch (opType) {
    case UNARY_OP_POW: return powValue(value, param);
    case UNARY_OP_TANH:
      switch (value->dtype) {
        case F16: value->as.f16 = (f16)tanh((double)value->as.f16); return OK;
        case F32: value->as.f32 = (f32)tanh((double)value->as.f32); return OK;
        case F64: value->as.f64 = (f64)tanh((double)value->as.f64); return OK;
        default: return ERR_TANH_VALUE_NOT_FLOAT;
      }
    case UNARY_OP_RELU:
      switch (value->dtype) {
        case F16: value->as.f16 = value->as.f16 > 0 ? value->as.f16 : 0; return OK;
        case F32: value->as.f32 = value->as.f32 > 0 ? value->as.f32 : 0; return OK;
        case F64: value->as.f64 = value->as.f64 > 0 ? value->as.f64 : 0; return OK;
        default: return ERR_RELU_VALUE_NOT_FLOAT;
      }
    case UNARY_OP_NEGATE: return negateValue(value);
    case UNARY_OP_EXP: return expValue(value);
    case UNARY_OP_LOG: return logValue(value);
    case UNARY_OP_ABS: return absValue(value);
    default: return ERR_NO_OP;
  }
}

static Result unaryOpCpu(Context *ctx, Tensor *t, Tensor *dest, UnaryOpType opType, f32 param) {
  TensorArg inputArg = {0};
  Result result = materializeTensorOnContext(ctx, t, true, &inputArg);
  if (opType == UNARY_OP_RELU && shouldLogRelu()) {
    fprintf(stderr,
            "[Relu] phase=materialize backend=CPU inputCtx=%p inputDevice=%s ownsTensor=%d "
            "result=%s(%d)\n",
            (void *)(inputArg.tensor != NULL ? inputArg.tensor->context : NULL),
            unaryContextDeviceName(inputArg.tensor != NULL ? inputArg.tensor->context : NULL),
            inputArg.ownsTensor ? 1 : 0, unaryResultName(result), (int)result);
  }
  if (result != OK) {
    return result;
  }

  Tensor *input = inputArg.tensor;
  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);

  for (tensor_size_t i = 0; i < input->size; i++) {
    Value value;
    VALUE_GET_FROM_ARR(input->values, i, &value, input->dtype);

    result = applyUnaryCpuValue(&value, opType, param);
    if (result != OK) {
      FreeTensor(ctx, output);
      releaseTensorArg(ctx, &inputArg);
      return result;
    }

    VALUE_SET(output->values, i, value);
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);
  releaseTensorArg(ctx, &inputArg);

  return OK;
}

static Result unaryOpCuda(Context *ctx, Tensor *t, Tensor *dest, UnaryOpType opType, f32 param) {
  TensorArg inputArg = {0};
  Result result = materializeTensorOnContext(ctx, t, true, &inputArg);
  if (opType == UNARY_OP_RELU && shouldLogRelu()) {
    fprintf(stderr,
            "[Relu] phase=materialize backend=CUDA inputCtx=%p inputDevice=%s ownsTensor=%d "
            "result=%s(%d)\n",
            (void *)(inputArg.tensor != NULL ? inputArg.tensor->context : NULL),
            unaryContextDeviceName(inputArg.tensor != NULL ? inputArg.tensor->context : NULL),
            inputArg.ownsTensor ? 1 : 0, unaryResultName(result), (int)result);
  }
  if (result != OK) {
    return result;
  }

  Tensor *input = inputArg.tensor;
  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);

  result = runCudaUnaryOp(ctx, input->dtype, opType, input->values, output->values, input->size,
                          param);
  if (opType == UNARY_OP_RELU) {
    logReluTensorState("cuda_kernel", ctx, input, result);
  }
  if (result != OK) {
    FreeTensor(ctx, output);
    releaseTensorArg(ctx, &inputArg);
    return result;
  }

  *dest = *output;
  freeAlloc(ctx->memory, output);
  releaseTensorArg(ctx, &inputArg);

  return OK;
}

static Result dispatchUnaryOp(Context *ctx, Tensor *t, Tensor *dest, UnaryOpType opType, f32 param) {
  if (opType == UNARY_OP_RELU && shouldLogRelu()) {
    fprintf(stderr, "[Relu] phase=dispatch device=%s\n",
            unaryDeviceTypeName(getUnaryDispatchDevice(ctx)));
  }

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, dest, opType, param);
    case CPU:
    default: return unaryOpCpu(ctx, t, dest, opType, param);
  }
}

Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest) {
  Result result = validatePowTensor(t);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_POW, power);
}

Result Tanh(Context *ctx, Tensor *t, Tensor *dest) {
  Result result = validateFloatUnaryTensor(t, ERR_TANH_VALUE_NOT_FLOAT);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_TANH, 0.0f);
}

Result Relu(Context *ctx, Tensor *t, Tensor *dest) {
  logReluTensorState("entry", ctx, t, OK);

  Result result = validateFloatUnaryTensor(t, ERR_RELU_VALUE_NOT_FLOAT);
  if (result != OK) {
    logReluTensorState("validate", ctx, t, result);
    return result;
  }

  result = dispatchUnaryOp(ctx, t, dest, UNARY_OP_RELU, 0.0f);
  if (shouldLogRelu()) {
    fprintf(stderr, "[Relu] phase=return result=%s(%d)\n", unaryResultName(result), (int)result);
  }

  return result;
}

Result Negate(Context *ctx, Tensor *t, Tensor *dest) {
  Result result = validateNegateTensor(t);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_NEGATE, 0.0f);
}

Result Exp(Context *ctx, Tensor *t, Tensor *dest) {
  Result result = validateFloatUnaryTensor(t, ERR_EXP_VALUE_NOT_FLOAT);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_EXP, 0.0f);
}

Result Log(Context *ctx, Tensor *t, Tensor *dest) {
  Result result = validateFloatUnaryTensor(t, ERR_LOG_VALUE_NOT_FLOAT);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_LOG, 0.0f);
}

Result Abs(Context *ctx, Tensor *t, Tensor *dest) {
  Result result = validateAbsTensor(t);
  if (result != OK) {
    return result;
  }

  return dispatchUnaryOp(ctx, t, dest, UNARY_OP_ABS, 0.0f);
}
