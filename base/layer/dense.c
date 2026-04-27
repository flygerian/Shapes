#include "common.h"
#include "result/result.h"
#include "shapes.h"

#include "tensor/tensor_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Build a destination tensor that matches x's rank and leading dims, but swaps
// the last dim (feature width). Dense uses this to preserve any batch axes.
static Dim swapLastDim(Context *ctx, Dim dim, dim_t lastDim) {
  u8 numDims = dim.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  PANIC_IF(dims == NULL, ALLOCATION_FAILED);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = dim.dims[i];
  }
  dims[numDims - 1] = lastDim;
  Dim shape = {.dims = dims, .numOfDims = numDims};

  return shape;
}

static Result validateDenseGradBuffer(Tensor *grad, Tensor *reference, Dtype dtype) {
  if (isInvalidTensor(grad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (grad->dtype != dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (grad->shape.numOfDims != reference->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  for (u8 i = 0; i < reference->shape.numOfDims; i++) {
    if (grad->shape.dims[i] != reference->shape.dims[i]) {
      return ERR_DIM_MISMATCH;
    }
  }

  return OK;
}

static Result validateDenseBiasGradBuffer(Tensor *grad, dim_t outputSize, Dtype dtype) {
  if (grad == NULL) {
    return OK;
  }

  if (isInvalidTensor(grad)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (grad->dtype != dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (grad->shape.numOfDims != 1 || grad->shape.dims[0] != outputSize) {
    return ERR_DIM_MISMATCH;
  }

  return OK;
}

static void promoteDenseBackwardInput(Context *ctx, Tensor *src, Tensor *dest) {
  if (src->shape.numOfDims == 1) {
    *dest = *UnSqueeze(ctx, src, 0);
    return;
  }

  *dest = *src;
}

static bool shouldLogOpTiming(void) {
  const char *value = getenv("SHAPES_LOG_OP_TIMES");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static double opTimingNowMs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void logOpTiming(Context *ctx, const char *opName, const char *phase, double startMs) {
  if (!shouldLogOpTiming()) {
    return;
  }

  const char *deviceName = "CPU(default)";
  if (ctx != NULL && ctx->device != NULL) {
    switch (ctx->device->type) {
      case CPU: deviceName = "CPU"; break;
      case CUDA: deviceName = "CUDA"; break;
      default: deviceName = "UNKNOWN"; break;
    }
  }

  fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", opName, deviceName, phase,
          opTimingNowMs() - startMs);
}

Tensor *DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias) {
  // Dense expects:
  // x: [..., inputSize]
  // w: [outputSize, inputSize]
  // b: [outputSize] (optional)
  // out: [..., outputSize]
  PANIC_IF(isInvalidTensor(x) || isInvalidTensor(w) || (withBias && isInvalidTensor(b)),
           ERR_NULL_TENSOR_PROVIDED);

  PANIC_IF(x->shape.numOfDims < 2 || w->shape.numOfDims != 2, ERR_MATMUL_MIN_2D);

  PANIC_IF(x->dtype != w->dtype || (withBias && b->dtype != x->dtype), ERR_DTYPE_MISMATCH);

  PANIC_IF(x->dtype != F16 && x->dtype != F32 && x->dtype != F64, ERR_DTYPE_MISMATCH);

  dim_t inputSize = x->shape.dims[x->shape.numOfDims - 1];
  dim_t outputSize = w->shape.dims[0];

  PANIC_IF(w->shape.dims[1] != inputSize, ERR_MATMUL_INNER_DIM_MISMATCH);

  PANIC_IF(withBias && (b->shape.numOfDims != 1 || b->shape.dims[0] != outputSize),
           ERR_DIM_MISMATCH);
  // BLAS expects dense row-major buffers. Views/slices from Go can be
  // non-contiguous, so we materialize contiguous copies when needed.
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }

  Tensor *xContig = materializeTensorOnContext(ctx, x);
  Tensor *wContig = materializeTensorOnContext(ctx, w);

  logOpTiming(ctx, "DenseLinear", "materialize", totalStartMs);

  tensor_size_t rows = x->size / inputSize;

  phaseStartMs = opTimingNowMs();

  Dim newDims = swapLastDim(ctx, x->shape, outputSize);
  Tensor *out = T_Zeros(ctx, newDims);

  logOpTiming(ctx, "DenseLinear", "alloc_output", phaseStartMs);

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  phaseStartMs = opTimingNowMs();
  runGemm(ctx, x->dtype, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize,
          xContig->values, (int)inputSize, wContig->values, (int)inputSize, false, out->values,
          (int)outputSize);

  logOpTiming(ctx, "DenseLinear", "gemm", phaseStartMs);

  if (withBias) {
    AddInPlace(ctx, out, b);
  }

  return out;
}

Result DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW,
                     Tensor *dB) {
  // DenseBackward accumulates gradients into preallocated buffers:
  // x: [..., inputSize], w: [outputSize, inputSize], gradOut: [..., outputSize]
  // dX: [..., inputSize], dW: [outputSize, inputSize], dB: [outputSize] or NULL
  if (ctx == NULL || isInvalidTensor(x) || isInvalidTensor(w) || isInvalidTensor(gradOut) ||
      isInvalidTensor(dX) || isInvalidTensor(dW)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Tensor x2d = {0};
  promoteDenseBackwardInput(ctx, x, &x2d);

  Tensor gradOut2d = {0};
  promoteDenseBackwardInput(ctx, gradOut, &gradOut2d);

  if (x2d.shape.numOfDims < 2 || gradOut2d.shape.numOfDims < 2 || w->shape.numOfDims != 2) {
    return ERR_MATMUL_MIN_2D;
  }

  if (x->dtype != gradOut->dtype || x->dtype != w->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x->dtype != F16 && x->dtype != F32 && x->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t inputSize = x2d.shape.dims[x2d.shape.numOfDims - 1];
  dim_t outputSize = w->shape.dims[0];

  if (w->shape.dims[1] != inputSize) {
    return ERR_MATMUL_INNER_DIM_MISMATCH;
  }
  if (gradOut2d.shape.dims[gradOut2d.shape.numOfDims - 1] != outputSize) {
    return ERR_DIM_MISMATCH;
  }

  Result res = validateDenseGradBuffer(dX, x, x->dtype);
  if (res != OK) {
    return res;
  }
  res = validateDenseGradBuffer(dW, w, w->dtype);
  if (res != OK) {
    return res;
  }
  res = validateDenseBiasGradBuffer(dB, outputSize, x->dtype);
  if (res != OK) {
    return res;
  }

  tensor_size_t rows = x2d.size / inputSize;
  if (gradOut2d.size != rows * outputSize) {
    return ERR_DIM_MISMATCH;
  }

  // Same contiguous requirement as forward: BLAS kernels consume packed rows.
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }

  Tensor *xContig = materializeTensorOnContext(ctx, &x2d);
  Tensor *wContig = materializeTensorOnContext(ctx, w);
  Tensor *gContig = materializeTensorOnContext(ctx, &gradOut2d);

  logOpTiming(ctx, "DenseBackward", "materialize", totalStartMs);

  phaseStartMs = opTimingNowMs();
  Tensor dX2d = {0};
  Tensor *createdDX2d = t_Empty(ctx, swapLastDim(ctx, x2d.shape, inputSize), x->dtype);
  PANIC_IF(createdDX2d == NULL, ALLOCATION_FAILED);
  dX2d = *createdDX2d;

  Tensor dWRaw = {0};
  Tensor *createdDWRaw = t_Empty(ctx, swapLastDim(ctx, w->shape, inputSize), w->dtype);
  PANIC_IF(createdDWRaw == NULL, ALLOCATION_FAILED);
  dWRaw = *createdDWRaw;

  logOpTiming(ctx, "DenseBackward", "alloc_outputs", phaseStartMs);

  phaseStartMs = opTimingNowMs();
  runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize,
          gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX2d.values,
          (int)inputSize);

  logOpTiming(ctx, "DenseBackward", "gemm_dx", phaseStartMs);

  // dW = gradOut^T * x
  phaseStartMs = opTimingNowMs();
  runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows,
          gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dWRaw.values,
          (int)inputSize);
  logOpTiming(ctx, "DenseBackward", "gemm_dw", phaseStartMs);

  phaseStartMs = opTimingNowMs();
  Tensor *dXReduced = ReduceBroadcast(ctx, x, &dX2d);
  AddInPlace(ctx, dX, dXReduced);

  Tensor *dWReduced = ReduceBroadcast(ctx, w, &dWRaw);
  AddInPlace(ctx, dW, dWReduced);

  if (dB != NULL) {
    Tensor *dBRaw = Sum(ctx, gContig, 0);
    Tensor *dBReduced = ReduceBroadcast(ctx, dB, dBRaw);
    AddInPlace(ctx, dB, dBReduced);
  }

  logOpTiming(ctx, "DenseBackward", "bias_grad", phaseStartMs);
  logOpTiming(ctx, "DenseBackward", "total", totalStartMs);

  return OK;
}
