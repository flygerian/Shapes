#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
  #include <vecLib/vDSP.h>
  #include <vecLib/vForce.h>
#endif

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

static void tanhCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
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

static void tanhCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
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

// Left scalar deliberately: -O3 auto-vectorizes to fmaxnm.4s with a 16-wide unrolled main loop.
// vDSP/vForce have no ReLU equivalent that beats this.
static void reluCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
  }
}

static void reluCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
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

static void powCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n, f32 power) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvpowsf(dst, &power, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = powf(src[i], power);
  }
  #endif
}

static void powCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n, f64 power) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvpows(dst, &power, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = pow(src[i], power);
  }
  #endif
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

static void sqrtCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvsqrtf(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = sqrtf(src[i]);
  }
  #endif
}

static void sqrtCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvsqrt(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = sqrt(src[i]);
  }
  #endif
}

static Tensor *sqrtCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: sqrtCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: sqrtCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    default: PANIC_IF(true, ERR_SQRT_VALUE_NOT_FLOAT);
  }

  return output;
}

static void expCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvexpf(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = expf(src[i]);
  }
  #endif
}

static void expCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvexp(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = exp(src[i]);
  }
  #endif
}

static Tensor *expCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: expCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: expCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    default: PANIC_IF(true, ERR_EXP_VALUE_NOT_FLOAT);
  }

  return output;
}

static void logCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvlogf(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = logf(src[i]);
  }
  #endif
}

static void logCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvlog(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = log(src[i]);
  }
  #endif
}

static Tensor *logCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: logCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: logCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    default: PANIC_IF(true, ERR_LOG_VALUE_NOT_FLOAT);
  }

  return output;
}

static void absCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvfabsf(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = fabsf(src[i]);
  }
  #endif
}

static void absCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    int ni = (int)n;
    PANIC_IF(ni <= 0, ERR_NO_OP);
    vvfabs(dst, src, &ni);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = fabs(src[i]);
  }
  #endif
}

static Tensor *absCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32: absCpuF32((const f32 *)input->values, (f32 *)output->values, input->size); break;
    case F64: absCpuF64((const f64 *)input->values, (f64 *)output->values, input->size); break;
    case I8: {
      const i8 *s = input->values;
      i8 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = s[i] < 0 ? (i8)-s[i] : s[i];
      }
      break;
    }
    case I16: {
      const i16 *s = input->values;
      i16 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = s[i] < 0 ? (i16)-s[i] : s[i];
      }
      break;
    }
    case I32: {
      const i32 *s = input->values;
      i32 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = s[i] < 0 ? -s[i] : s[i];
      }
      break;
    }
    case I64: {
      const i64 *s = input->values;
      i64 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = s[i] < 0 ? -s[i] : s[i];
      }
      break;
    }
    default: PANIC_IF(true, ERR_ABS_VALUE_NOT_SIGNED);
  }

  return output;
}

static void negateCpuF32(const f32 *restrict src, f32 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    PANIC_IF(n == 0, ERR_NO_OP);
    vDSP_vneg(src, 1, dst, 1, (vDSP_Length)n);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = -src[i];
  }
  #endif
}

static void negateCpuF64(const f64 *restrict src, f64 *restrict dst, tensor_size_t n) {
  #ifdef __APPLE__
    PANIC_IF(n == 0, ERR_NO_OP);
    vDSP_vnegD(src, 1, dst, 1, (vDSP_Length)n);
  #else
  for (tensor_size_t i = 0; i < n; i++) {
    dst[i] = -src[i];
  }
  #endif
}

static Tensor *negateCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *output = t_Zeros(ctx, input->shape, input->dtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16:
    case F32:
      negateCpuF32((const f32 *)input->values, (f32 *)output->values, input->size);
      break;
    case F64:
      negateCpuF64((const f64 *)input->values, (f64 *)output->values, input->size);
      break;
    case I8: {
      const i8 *s = input->values;
      i8 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = (i8)-s[i];
      }
      break;
    }
    case I16: {
      const i16 *s = input->values;
      i16 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = (i16)-s[i];
      }
      break;
    }
    case I32: {
      const i32 *s = input->values;
      i32 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = -s[i];
      }
      break;
    }
    case I64: {
      const i64 *s = input->values;
      i64 *d = output->values;
      for (tensor_size_t i = 0; i < input->size; i++) {
        d[i] = -s[i];
      }
      break;
    }
    default: PANIC_IF(true, ERR_NEGATE_UNSUPPORTED_DTYPE);
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

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_NEGATE, 0.0f);
    case CPU:
    default: return negateCpu(ctx, t);
  }
}

Tensor *shapes_Exp(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_EXP_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_EXP, 0.0f);
    case CPU:
    default: return expCpu(ctx, t);
  }
}

Tensor *shapes_Log(Context *ctx, Tensor *t) {
  Result result = validateFloatUnaryTensor(t, ERR_LOG_VALUE_NOT_FLOAT);
  PANIC_IF(result != OK, result);

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_LOG, 0.0f);
    case CPU:
    default: return logCpu(ctx, t);
  }
}

Tensor *shapes_Abs(Context *ctx, Tensor *t) {
  Result result = validateAbsTensor(t);
  PANIC_IF(result != OK, result);

  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: return unaryOpCuda(ctx, t, UNARY_OP_ABS, 0.0f);
    case CPU:
    default: return absCpu(ctx, t);
  }
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

  Tensor *out;
  switch (getUnaryDispatchDevice(ctx)) {
    case CUDA: out = unaryOpCuda(ctx, t, UNARY_OP_SQRT, 0.0f); break;
    case CPU:
    default: out = sqrtCpu(ctx, t); break;
  }

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  shapes_Array_AppendTensor(out->inputs, t);
  out->opType = OP_SQRT;
  out->grad = shapes_Make_ZerosTensor(ctx, out->shape);

  return out;
}
