#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <vecLib/vForce.h>

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
    case ERR_OUT_OF_MEMORY: return " ERR_OUT_OF_MEMORY";
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
          "[shapes_Relu] phase=%s ctx=%p ctxDevice=%s tensorCtx=%p tensorDevice=%s dtype=%s "
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

static Result applyUnaryCpuValue(Value *value, UnaryOpType opType) {
  switch (opType) {
    case UNARY_OP_NEGATE: return negateValue(value);
    case UNARY_OP_EXP: return expValue(value);
    case UNARY_OP_LOG: return logValue(value);
    case UNARY_OP_ABS: return absValue(value);
    case UNARY_OP_SQRT: return sqrtValue(value);
    default: return ERR_NO_OP;
  }
}

static void tanhCpuF32(const f32 *src, f32 *dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvtanhf(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = tanhf(src[i]);
  }
  #endif
}

static void tanhCpuF64(const f64 *src, f64 *dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvtanh(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = tanh(src[i]);
  }
  #endif
}

static Tensor *tanhCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: tanhCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: tanhCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    default: PANIC_IF(true, ERR_TANH_VALUE_NOT_FLOAT);
  }

  return output;
}

static void reluCpuF32(const f32 *src, f32 *dst, tensor_size_t n) {
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
  }
}

static void reluCpuF64(const f64 *src, f64 *dst, tensor_size_t n) {
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = src[i] > 0.0 ? src[i] : 0.0;
  }
}

static Tensor *reluCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: reluCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: reluCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    default: PANIC_IF(true, ERR_RELU_VALUE_NOT_FLOAT);
  }

  return output;
}

static void powCpuF32(const f32 *src, f32 *dst, tensor_size_t n, f32 power) {
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = powf(src[i], power);
  }
}

static void powCpuF64(const f64 *src, f64 *dst, tensor_size_t n, f64 power) {
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = pow(src[i], power);
  }
}

static Tensor *powCpu(Context *ctx, Tensor *t, f32 power) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32:
      powCpuF32((const f32 *)input->values, (f32 *)output->values, input->size, power);
      break;
    case F64:
      powCpuF64((const f64 *)input->values, (f64 *)output->values, input->size, (f64)power);
      break;
    default: PANIC_IF(true, ERR_POW_VALUE_NOT_FLOAT);
  }

  return output;
}

static Tensor *unaryOpCpu(Context *ctx, Tensor *t, UnaryOpType opType) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  for (tensor_size_t i = 0; i < input->size; i++) {
    Value value;
    VALUE_GET_FROM_ARR(input->values, i, &value, input->dtype);

    Result result = applyUnaryCpuValue(&value, opType);
    PANIC_IF(result != OK, result);

    VALUE_SET(output->values, i, value);
  }

  return output;
}

static Tensor *unaryOpCuda(Context *ctx, Tensor *t, UnaryOpType opType, f32 param) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ERR_OUT_OF_MEMORY);

  Result result =
      runCudaUnaryOp(ctx, input->dtype, opType, input->values, output->values, input->size, param);

  if (opType == UNARY_OP_RELU) {
    logReluTensorState("cuda_kernel", ctx, input, result);
  }

  PANIC_IF(result != OK, CUDA_OP_FAILED);

  return output;
}

static Tensor *dispatchUnaryOp(Context *ctx, Tensor *t, UnaryOpType opType, f32 param) {
  if (opType == UNARY_OP_RELU && shouldLogRelu()) {
    fprintf(stderr, "[shapes_Relu] phase=dispatch device=%s\n",
            unaryDeviceTypeName(getUnaryDispatchDevice(ctx)));
  }

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, opType, param);
    case CPU:
    default: return unaryOpCpu(ctx, t, opType);
  }
}

Tensor *shapes_Pow(Context *ctx, Tensor *t, f32 power) {
  Result result = validatePowTensor(t);
  PANIC_IF(result != OK, result);

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_POW, power);
    case CPU:
    default: return powCpu(ctx, t, power);
  }
}

Tensor *shapes_Tanh(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_TANH_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_TANH, 0.0f);
    case CPU:
    default: return tanhCpu(ctx, t);
  }
}

Tensor *shapes_Relu(Context *ctx, Tensor *t) {
  logReluTensorState("entry", ctx, t, OK);
  Result result = validateFloatUnaryTensor(t, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  Tensor *out;
  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: out = unaryOpCuda(ctx, t, UNARY_OP_RELU, 0.0f); break;
    case CPU:
    default: out = reluCpu(ctx, t); break;
  }

  if (shouldLogRelu()) {
    fprintf(stderr, "[shapes_Relu] phase=return result=OK(0)\n");
  }
  return out;
}

Tensor *shapes_ReluBackward(Context *ctx, Tensor *output, Tensor *gradOut) {
  Result result = validateFloatUnaryTensor(output, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  result = validateFloatUnaryTensor(gradOut, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  PANIC_IF(output->dtype != gradOut->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(output->size != gradOut->size || output->shape.numOfDims != gradOut->shape.numOfDims,
           ERR_DIM_MISMATCH);

  for (u8 i = 0; i < output->shape.numOfDims; i++) {
    PANIC_IF(output->shape.dims[i] != gradOut->shape.dims[i], ERR_DIM_MISMATCH);
  }

  Tensor *outputWork = materializeTensorOnContext(ctx, output);
  Tensor *gradWork = materializeTensorOnContext(ctx, gradOut);

  Tensor *dInput = t_Zeros(ctx, outputWork->shape, outputWork->dtype);
  PANIC_IF(dInput == NULL, ERR_OUT_OF_MEMORY);

  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    result = runCudaReluBackward(ctx, outputWork->dtype, outputWork->values, gradWork->values,
                                 dInput->values, outputWork->size);
    PANIC_IF(result != OK, CUDA_OP_FAILED);

  } else if (outputWork->dtype == F64) {
    f64 *outputValues = outputWork->values;
    f64 *gradValues = gradWork->values;
    f64 *destValues = dInput->values;
    for (tensor_size_t i = 0; i < outputWork->size; i++) {
      destValues[i] = outputValues[i] > 0.0 ? gradValues[i] : 0.0;
    }
  } else {
    f32 *outputValues = outputWork->values;
    f32 *gradValues = gradWork->values;
    f32 *destValues = dInput->values;
    for (tensor_size_t i = 0; i < outputWork->size; i++) {
      destValues[i] = outputValues[i] > 0.0f ? gradValues[i] : 0.0f;
    }
  }

  return dInput;
}

void shapes_ReluBackwardAccumulate(Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest) {
  Result result = validateFloatUnaryTensor(output, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  result = validateFloatUnaryTensor(gradOut, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  result = validateFloatUnaryTensor(dest, ERR_RELU_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  PANIC_IF(output->dtype != gradOut->dtype || output->dtype != dest->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(output->size != gradOut->size || output->size != dest->size ||
               output->shape.numOfDims != gradOut->shape.numOfDims ||
               output->shape.numOfDims != dest->shape.numOfDims,
           ERR_DIM_MISMATCH);

  for (u8 i = 0; i < output->shape.numOfDims; i++) {
    PANIC_IF(output->shape.dims[i] != gradOut->shape.dims[i] ||
                 output->shape.dims[i] != dest->shape.dims[i],
             ERR_DIM_MISMATCH);
  }

  PANIC_IF(ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA &&
               (!dest->isContigous || dest->isView),
           ERR_NO_OP);

  Tensor *outputWork = materializeTensorOnContext(ctx, output);
  Tensor *gradWork = materializeTensorOnContext(ctx, gradOut);

  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    result = runCudaReluBackwardAccumulate(ctx, outputWork->dtype, outputWork->values,
                                           gradWork->values, dest->values, outputWork->size);
    PANIC_IF(result != OK, CUDA_OP_FAILED);
    return;
  }

  PANIC_IF(!dest->isContigous || dest->isView, ERR_NO_OP);

  if (outputWork->dtype == F64) {
    f64 *outputValues = outputWork->values;
    f64 *gradValues = gradWork->values;
    f64 *destValues = dest->values;
    for (tensor_size_t i = 0; i < outputWork->size; i++) {
      destValues[i] += outputValues[i] > 0.0 ? gradValues[i] : 0.0;
    }
  } else {
    f32 *outputValues = outputWork->values;
    f32 *gradValues = gradWork->values;
    f32 *destValues = dest->values;
    for (tensor_size_t i = 0; i < outputWork->size; i++) {
      destValues[i] += outputValues[i] > 0.0f ? gradValues[i] : 0.0f;
    }
  }
}

Tensor *shapes_Negate(Context *ctx, Tensor *t) {
  Result result = validateNegateTensor(t);
  PANIC_IF(result != OK, result);
  return dispatchUnaryOp(ctx, t, UNARY_OP_NEGATE, 0.0f);
}

Tensor *shapes_Exp(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_EXP_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);
  return dispatchUnaryOp(ctx, t, UNARY_OP_EXP, 0.0f);
}

Tensor *shapes_Log(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_LOG_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);
  return dispatchUnaryOp(ctx, t, UNARY_OP_LOG, 0.0f);
}

Tensor *shapes_Abs(Context *ctx, Tensor *t) {
  Result result = validateAbsTensor(t);
  PANIC_IF(result != OK, result);
  return dispatchUnaryOp(ctx, t, UNARY_OP_ABS, 0.0f);
}

void shapes_SqrtBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = shapes_Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input == NULL || input->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *two = shapes_Make_FloatTensor(ctx, SHAPE1D(1), 2.0f);
  Tensor *twoTimesOutput = shapes_Multiply(ctx, two, tensor);
  Tensor *gradInput = shapes_Divide(ctx, tensor->grad, twoTimesOutput);

  Tensor *reducedGrad = shapes_ReduceBroadcast(ctx, input, gradInput);
  shapes_AddInPlace(ctx, input->grad, reducedGrad);
}

Tensor *shapes_Sqrt(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_SQRT_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  Tensor *out = dispatchUnaryOp(ctx, t, UNARY_OP_SQRT, 0.0f);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  shapes_Array_AppendTensor(out->inputs, t);
  out->opType = OP_SQRT;
  out->grad = shapes_Make_ZerosTensor(ctx, out->shape);

  return out;
}
