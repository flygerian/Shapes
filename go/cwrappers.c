#include "cwrappers.h"
#include "layer/im2col.h"
#include <string.h>

Result clearTensorValues(Tensor *t);

Dim *makeDim(Memory *mem, dim_t *dims, u8 numDims) {
  Dim *d = allocate(mem, sizeof(Dim));
  d->dims = allocate(mem, sizeof(dim_t) * numDims);
  memcpy(d->dims, dims, sizeof(dim_t) * numDims);
  d->numOfDims = numDims;
  d->multipliers = NULL;
  return d;
}

void zeroTensorValues(Tensor *t) {
  clearTensorValues(t);
}

Result copyTensorValuesFromHost(Tensor *t, void *src, size_t size) {
  if (t == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  return copyBetweenContexts(NULL, t->context, src, t->values, size);
}

Result copyTensorValuesToHost(Tensor *t, void *dest, size_t size) {
  if (t == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  return copyBetweenContexts(t->context, NULL, t->values, dest, size);
}

Result wrap_CopyShape(Tensor *t, dim_t *destDims, u8 *numDims) { return CopyShape(t, destDims, numDims); }

Result wrap_GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = GetTensorAt(ctx, source, index, dest);
  *out = dest;
  return r;
}

Result wrap_IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = IndexWithTensor(ctx, source, indices, dest);
  *out = dest;
  return r;
}

Result wrap_IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices,
                              Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = IndexWithTensor2d(ctx, source, rowIndices, colIndices, dest);
  *out = dest;
  return r;
}

Result wrap_GetScalar(Tensor *t, Value *result) { return GetScalar(t, result); }

double value_as_double(Value v) {
  switch (v.dtype) {
    case BOOL: return v.as.boolean ? 1.0 : 0.0;
    case F16: return (double)v.as.f16;
    case F32: return (double)v.as.f32;
    case F64: return (double)v.as.f64;
    case U8: return (double)v.as.u8;
    case U16: return (double)v.as.u16;
    case U32: return (double)v.as.u32;
    case U64: return (double)v.as.u64;
    case I8: return (double)v.as.i8;
    case I16: return (double)v.as.i16;
    case I32: return (double)v.as.i32;
    case I64: return (double)v.as.i64;
    default: return 0.0;
  }
}

u64 value_as_u64(Value v) {
  switch (v.dtype) {
    case BOOL: return v.as.boolean ? 1ULL : 0ULL;
    case U8: return (u64)v.as.u8;
    case U16: return (u64)v.as.u16;
    case U32: return (u64)v.as.u32;
    case U64: return (u64)v.as.u64;
    case I8: return (u64)v.as.i8;
    case I16: return (u64)v.as.i16;
    case I32: return (u64)v.as.i32;
    case I64: return (u64)v.as.i64;
    default: return 0;
  }
}

i64 value_as_i64(Value v) {
  switch (v.dtype) {
    case BOOL: return v.as.boolean ? 1LL : 0LL;
    case I8: return (i64)v.as.i8;
    case I16: return (i64)v.as.i16;
    case I32: return (i64)v.as.i32;
    case I64: return (i64)v.as.i64;
    case U8: return (i64)v.as.u8;
    case U16: return (i64)v.as.u16;
    case U32: return (i64)v.as.u32;
    case U64: return (i64)v.as.u64;
    default: return 0;
  }
}

bool value_as_bool(Value v) {
  switch (v.dtype) {
    case BOOL: return v.as.boolean;
    default: return false;
  }
}

Result wrap_Add(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Add(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Subtract(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Multiply(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_Divide(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Divide(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_GreaterThan(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = GreaterThan(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = GreaterThanOrEqual(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_LessThan(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = LessThan(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = LessThanOrEqual(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_AddInPlace(Context *ctx, Tensor *a, Tensor *b) { return AddInPlace(ctx, a, b); }
Result wrap_SubtractInPlace(Context *ctx, Tensor *a, Tensor *b) { return SubtractInPlace(ctx, a, b); }
Result wrap_MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b) { return MultiplyInPlace(ctx, a, b); }

Result wrap_Cast(Context *ctx, Tensor *src, Dtype target, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Cast(ctx, src, dest, target);
  *out = dest;
  return r;
}

Result wrap_IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad) {
  return IndexAccumulate1d(ctx, dest, indices, srcGrad);
}

Result wrap_IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices,
                              Tensor *srcGrad) {
  return IndexAccumulate2d(ctx, dest, rowIndices, colIndices, srcGrad);
}

Result wrap_SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad) {
  return SliceAccumulate(ctx, dest, ranges, srcGrad);
}

Result wrap_Pow(Context *ctx, Tensor *t, f32 power, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Pow(ctx, t, power, dest);
  *out = dest;
  return r;
}

Result wrap_Exp(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Exp(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_Tanh(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Tanh(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_Relu(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Relu(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_Negate(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Negate(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_MeanWithDim(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = MeanDim(ctx, t, dest, dim);
  *out = dest;
  return r;
}

Result wrap_Log(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Log(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_Abs(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Abs(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits, Tensor **loss,
                                Tensor **probs) {
  Tensor *lossLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *probsLocal = allocate(ctx->memory, sizeof(Tensor));

  Result r = CrossEntropyForward(ctx, yGround, logits, lossLocal, probsLocal);
  *loss = lossLocal;
  *probs = probsLocal;
  return r;
}

Result wrap_CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut,
                                 Tensor **dLogits) {
  Tensor *dLogitsLocal = allocate(ctx->memory, sizeof(Tensor));
  Result r = CrossEntropyBackward(ctx, yGround, probs, gradOut, dLogitsLocal);
  *dLogits = dLogitsLocal;
  return r;
}

Result wrap_Sum(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Sum(ctx, t, dest, dim);
  *out = dest;
  return r;
}

Result wrap_Mean(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = MeanDim(ctx, t, dest, dim);
  *out = dest;
  return r;
}

Result wrap_Std(Context *ctx, Tensor *t, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Std(ctx, t, dest);
  *out = dest;
  return r;
}

Result wrap_Max(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Max(ctx, t, dest, dim);
  *out = dest;
  return r;
}

Result wrap_ArgMax(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = ArgMax(ctx, t, dest, dim);
  *out = dest;
  return r;
}

Tensor *wrap_T_Zeros(Context *ctx, Dim *shape) {
  return T_Zeros(ctx, *shape);
}

Tensor *wrap_T_Int(Context *ctx, Dim *shape, i8 value) {
  return T_Int(ctx, *shape, value);
}

Tensor *wrap_T_UInt(Context *ctx, Dim *shape, u8 value) {
  return T_UInt(ctx, *shape, value);
}

Tensor *wrap_T_Float(Context *ctx, Dim *shape, f32 value) {
  return T_Float(ctx, *shape, value);
}

Result wrap_Clone(Context *ctx, Tensor *src, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Clone(ctx, src, dest);
  *out = dest;
  return r;
}

Result wrap_Copy(Context *ctx, Tensor *src, Tensor *dest) { return Clone(ctx, src, dest); }

Tensor *wrap_T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses) {
  return T_OneHot(ctx, indices, numClasses);
}

Tensor *wrap_T_Arange(Context *ctx, f32 start, f32 end, f32 step) {
  return T_Arange(ctx, start, end, step);
}

Result wrap_MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = MatMul(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_Dot(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Dot(ctx, a, b, dest);
  *out = dest;
  return r;
}

Result wrap_DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = DenseLinear(ctx, x, w, b, withBias, dest);
  *out = dest;
  return r;
}

Result wrap_DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor **dX, Tensor **dW,
                          Tensor **dB) {
  Tensor *dxLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *dwLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *dbLocal = allocate(ctx->memory, sizeof(Tensor));

  Result r = DenseBackward(ctx, x, w, gradOut, dxLocal, dwLocal, dbLocal);
  *dX = dxLocal;
  *dW = dwLocal;
  *dB = dbLocal;
  return r;
}

Result wrap_BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon,
                                     Tensor **out, Tensor **mean, Tensor **variance) {
  Tensor *outLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *meanLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *varLocal = allocate(ctx->memory, sizeof(Tensor));

  Result r = BatchNormForwardTraining(ctx, x2d, gamma, beta, epsilon, outLocal, meanLocal, varLocal);
  *out = outLocal;
  *mean = meanLocal;
  *variance = varLocal;
  return r;
}

Result wrap_BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon,
                              Tensor **dX, Tensor **dGamma, Tensor **dBeta) {
  Tensor *dx = allocate(ctx->memory, sizeof(Tensor));
  Tensor *dGammaLocal = allocate(ctx->memory, sizeof(Tensor));
  Tensor *dBetaLocal = allocate(ctx->memory, sizeof(Tensor));

  Result r = BatchNormBackward(ctx, x2d, grad2d, gamma, epsilon, dx, dGammaLocal, dBetaLocal);
  *dX = dx;
  *dGamma = dGammaLocal;
  *dBeta = dBetaLocal;
  return r;
}

Result wrap_Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride,
                   Tensor *kernels, Tensor *bias, bool withBias, Tensor *x, Tensor **out,
                   Tensor **destColBuffer) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Tensor *colBuffer = allocate(ctx->memory, sizeof(Tensor));

  Result r = Conv2d(ctx, inChannels, outChannels, stride, kernels, bias, withBias, x, dest,
                    colBuffer);
  *out = dest;
  *destColBuffer = colBuffer;
  return r;
}

Result wrap_Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels,
                           Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer, Tensor *dBias,
                           bool withBias, u8 stride) {
  Result r = Conv2dBackward(ctx, input, dInput, kernels, dKernels, outputGrad, colBuffer, dBias,
                            withBias, stride);
  return r;
}

Result wrap_ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride,
                            Tensor *kernels, dim_t kernelH, dim_t kernelW, Tensor *x,
                            Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  dim_t kernelDims[2] = {kernelH, kernelW};
  Dim kernel = {.dims = kernelDims, .numOfDims = 2, .multipliers = NULL};
  Result r = ConvTranspose2d(ctx, inChannels, outChannels, stride, kernels, kernel, x, dest);
  *out = dest;
  return r;
}

Result wrap_ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut,
                                    u8 stride, Tensor **dX, Tensor **dKernels) {
  Tensor *dx = allocate(ctx->memory, sizeof(Tensor));
  Tensor *dKernelsLocal = allocate(ctx->memory, sizeof(Tensor));

  Result r = ConvTranspose2dBackward(ctx, x, kernels, gradOut, stride, dx, dKernelsLocal);
  *dX = dx;
  *dKernels = dKernelsLocal;
  return r;
}

Result wrap_MaxPool2d(Context *ctx, Tensor *x, dim_t kernelH, dim_t kernelW, u8 stride,
                      Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  dim_t kernelDims[2] = {kernelH, kernelW};
  Dim kernel = {.dims = kernelDims, .numOfDims = 2, .multipliers = NULL};
  Result r = MaxPool2d(ctx, x, kernel, stride, dest);
  *out = dest;
  return r;
}

Result wrap_MaxPool2dWithIndices(Context *ctx, Tensor *x, dim_t kernelH, dim_t kernelW, u8 stride,
                                 Tensor **out, Tensor **indices) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Tensor *argmax = allocate(ctx->memory, sizeof(Tensor));
  dim_t kernelDims[2] = {kernelH, kernelW};
  Dim kernel = {.dims = kernelDims, .numOfDims = 2, .multipliers = NULL};
  Result r = MaxPool2dWithIndices(ctx, x, kernel, stride, dest, argmax);
  *out = dest;
  *indices = argmax;
  return r;
}

Result wrap_MaxPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t kernelH,
                              dim_t kernelW, u8 stride, Tensor **dX) {
  Tensor *dx = allocate(ctx->memory, sizeof(Tensor));
  dim_t kernelDims[2] = {kernelH, kernelW};
  Dim kernel = {.dims = kernelDims, .numOfDims = 2, .multipliers = NULL};

  Result r = MaxPool2dBackward(ctx, x, gradOut, kernel, stride, dx);
  *dX = dx;
  return r;
}

Result wrap_MaxPool2dBackwardWithIndices(Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices,
                                         Tensor **dX) {
  Tensor *dx = allocate(ctx->memory, sizeof(Tensor));
  Result r = MaxPool2dBackwardWithIndices(ctx, x, gradOut, indices, dx);
  *dX = dx;
  return r;
}

Result wrap_AdaptiveAvgPool2d(Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = AdaptiveAvgPool2d(ctx, x, outH, outW, dest);
  *out = dest;
  return r;
}

Result wrap_AdaptiveAvgPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH,
                                      dim_t outW, Tensor **dX) {
  Tensor *dx = allocate(ctx->memory, sizeof(Tensor));
  Result r = AdaptiveAvgPool2dBackward(ctx, x, gradOut, outH, outW, dx);
  *dX = dx;
  return r;
}

Result wrap_Slice1(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0]);
  *out = dest;
  return res;
}

Result wrap_Slice2(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1]);
  *out = dest;
  return res;
}

Result wrap_Slice3(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2]);
  *out = dest;
  return res;
}

Result wrap_Slice4(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3]);
  *out = dest;
  return res;
}

Result wrap_Slice5(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4]);
  *out = dest;
  return res;
}

Result wrap_Slice6(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5]);
  *out = dest;
  return res;
}

Result wrap_Slice7(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
  *out = dest;
  return res;
}

Result wrap_Slice8(Context *ctx, Tensor *src, Tensor **out, Range *r) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
  *out = dest;
  return res;
}

Result wrap_Reshape(Context *ctx, Tensor *src, Tensor **out, Dim *newShape) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Reshape(ctx, src, dest, *newShape);
  *out = dest;
  freeAlloc(ctx->memory, newShape);
  return r;
}

Result wrap_Transpose(Context *ctx, Tensor *src, Tensor **out, dim_t d0, dim_t d1) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Transpose(ctx, src, dest, d0, d1);
  *out = dest;
  return r;
}

Result wrap_Permute(Context *ctx, Tensor *src, Tensor **out, Dim *order) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Permute(ctx, src, dest, *order);
  *out = dest;
  freeAlloc(ctx->memory, order);
  return r;
}

Result wrap_Squeeze(Context *ctx, Tensor *src, Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Squeeze(ctx, src, dest);
  *out = dest;
  return r;
}

Result wrap_SqueezeDim(Context *ctx, Tensor *src, Tensor **out, dim_t d) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = SqueezeDim(ctx, src, dest, d);
  *out = dest;
  return r;
}

Result wrap_UnSqueeze(Context *ctx, Tensor *src, Tensor **out, dim_t d) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = UnSqueeze(ctx, src, dest, d);
  *out = dest;
  return r;
}

Result wrap_Concat(Context *ctx, Tensor *target, dim_t targetDim, Tensor **tensors, u32 numTensors,
                   Tensor **out) {
  Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
  Result r = Concat(ctx, target, targetDim, tensors, numTensors, dest);
  *out = dest;
  return r;
}
