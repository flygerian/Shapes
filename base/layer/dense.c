#include "shapes.h"

#include "../memory.h"
#include "tensor/tensor_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Build a destination tensor that matches x's rank and leading dims, but swaps
// the last dim (feature width). Dense uses this to preserve any batch axes.
static Result initTensorLikeInputWithLastDim(Context *ctx, Tensor *dest, Tensor *x, dim_t lastDim,
                                             Dtype dtype) {
  u8 numDims = x->shape.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  Result allocRes = ensureAllocated(dims);
  if (allocRes != OK) {
    return allocRes;
  }
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * numDims);
  allocRes = ensureAllocated(multipliers);
  if (allocRes != OK) {
    freeAlloc(ctx->memory, dims);
    return allocRes;
  }

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = x->shape.dims[i];
  }
  dims[numDims - 1] = lastDim;

  Dim shape = {.dims = dims, .numOfDims = numDims, .multipliers = multipliers};
  calculateNumValuesAndMultipliers(shape, multipliers);

  Result initRes = initTensor(ctx, dest, shape, dtype);
  if (initRes != OK) {
    freeAlloc(ctx->memory, multipliers);
    freeAlloc(ctx->memory, dims);
  }
  return initRes;
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

static Result syncForOpTiming(Context *ctx) {
  if (!shouldLogOpTiming()) {
    return OK;
  }

  return Flush(ctx);
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

Result DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor *dest) {
  // Dense expects:
  // x: [..., inputSize]
  // w: [outputSize, inputSize]
  // b: [outputSize] (optional)
  // out: [..., outputSize]
  if (isInvalidTensor(x) || isInvalidTensor(w) || (withBias && isInvalidTensor(b))) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (x->shape.numOfDims < 2 || w->shape.numOfDims != 2) {
    return ERR_MATMUL_MIN_2D;
  }

  if (x->dtype != w->dtype || (withBias && b->dtype != x->dtype)) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x->dtype != F16 && x->dtype != F32 && x->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t inputSize = x->shape.dims[x->shape.numOfDims - 1];
  dim_t outputSize = w->shape.dims[0];

  if (w->shape.dims[1] != inputSize) {
    return ERR_MATMUL_INNER_DIM_MISMATCH;
  }

  if (withBias && (b->shape.numOfDims != 1 || b->shape.dims[0] != outputSize)) {
    return ERR_DIM_MISMATCH;
  }

  // BLAS expects dense row-major buffers. Views/slices from Go can be
  // non-contiguous, so we materialize contiguous copies when needed.
  TensorArg xArg = {0};
  TensorArg wArg = {0};
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }
  Result res = materializeTensorOnContext(ctx, x, true, &xArg);
  if (res != OK) {
    return res;
  }
  res = materializeTensorOnContext(ctx, w, true, &wArg);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    return res;
  }
  Tensor *xContig = xArg.tensor;
  Tensor *wContig = wArg.tensor;
  logOpTiming(ctx, "DenseLinear", "materialize", totalStartMs);

  tensor_size_t rows = x->size / inputSize;
  res = syncForOpTiming(ctx);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    releaseTensorArg(ctx, &wArg);
    return res;
  }
  phaseStartMs = opTimingNowMs();
  res = initTensorLikeInputWithLastDim(ctx, dest, x, outputSize, x->dtype);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    releaseTensorArg(ctx, &wArg);
    return res;
  }
  logOpTiming(ctx, "DenseLinear", "alloc_output", phaseStartMs);

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  res = syncForOpTiming(ctx);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    releaseTensorArg(ctx, &wArg);
    return res;
  }
  phaseStartMs = opTimingNowMs();
  runGemm(ctx, x->dtype, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize,
          xContig->values, (int)inputSize, wContig->values, (int)inputSize, false, dest->values,
          (int)outputSize);
  res = syncForOpTiming(ctx);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    releaseTensorArg(ctx, &wArg);
    return res;
  }
  logOpTiming(ctx, "DenseLinear", "gemm", phaseStartMs);

  if (withBias) {
    res = syncForOpTiming(ctx);
    if (res != OK) {
      releaseTensorArg(ctx, &xArg);
      releaseTensorArg(ctx, &wArg);
      return res;
    }
    phaseStartMs = opTimingNowMs();
    res = AddInPlace(ctx, dest, b);
    if (res != OK) {
      releaseTensorArg(ctx, &xArg);
      releaseTensorArg(ctx, &wArg);
      return res;
    }
    logOpTiming(ctx, "DenseLinear", "bias_add", phaseStartMs);
  }

  releaseTensorArg(ctx, &xArg);
  releaseTensorArg(ctx, &wArg);

  logOpTiming(ctx, "DenseLinear", "total", totalStartMs);

  return OK;
}

Result DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW,
                     Tensor *dB) {
  // Backward inputs/outputs:
  // x: [..., inputSize], w: [outputSize, inputSize], gradOut: [..., outputSize]
  // dX: [..., inputSize], dW: [outputSize, inputSize], dB: [outputSize]
  if (isInvalidTensor(x) || isInvalidTensor(w) || isInvalidTensor(gradOut)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (x->shape.numOfDims < 2 || gradOut->shape.numOfDims < 2 || w->shape.numOfDims != 2) {
    return ERR_MATMUL_MIN_2D;
  }

  if (x->dtype != gradOut->dtype || x->dtype != w->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x->dtype != F16 && x->dtype != F32 && x->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t inputSize = x->shape.dims[x->shape.numOfDims - 1];
  dim_t outputSize = w->shape.dims[0];
  if (w->shape.dims[1] != inputSize) {
    return ERR_MATMUL_INNER_DIM_MISMATCH;
  }

  if (gradOut->shape.dims[gradOut->shape.numOfDims - 1] != outputSize) {
    return ERR_DIM_MISMATCH;
  }

  tensor_size_t rows = x->size / inputSize;
  if (gradOut->size != rows * outputSize) {
    return ERR_DIM_MISMATCH;
  }

  // Same contiguous requirement as forward: BLAS kernels consume packed rows.
  TensorArg xArg = {0};
  TensorArg wArg = {0};
  TensorArg gArg = {0};
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }
  Result res = materializeTensorOnContext(ctx, x, true, &xArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, w, true, &wArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, gradOut, true, &gArg);
  if (res != OK) {
    goto cleanup;
  }
  Tensor *xContig = xArg.tensor;
  Tensor *wContig = wArg.tensor;
  Tensor *gContig = gArg.tensor;
  logOpTiming(ctx, "DenseBackward", "materialize", totalStartMs);

  res = syncForOpTiming(ctx);
  if (res != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  res = initTensorLikeInputWithLastDim(ctx, dX, x, inputSize, x->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = initTensorLikeInputWithLastDim(ctx, dW, w, inputSize, w->dtype);
  if (res != OK) {
    goto cleanup;
  }
  logOpTiming(ctx, "DenseBackward", "alloc_outputs", phaseStartMs);
  if (x->dtype == F64) {
    // dX = gradOut * w
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize,
            gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX->values,
            (int)inputSize);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "DenseBackward", "gemm_dx", phaseStartMs);

    // dW = gradOut^T * x
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows,
            gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dW->values,
            (int)inputSize);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "DenseBackward", "gemm_dw", phaseStartMs);
  } else {
    // dX = gradOut * w
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize,
            gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX->values,
            (int)inputSize);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "DenseBackward", "gemm_dx", phaseStartMs);

    // dW = gradOut^T * x
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows,
            gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dW->values,
            (int)inputSize);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "DenseBackward", "gemm_dw", phaseStartMs);
  }

  dim_t *grad2dDims = allocate(ctx->memory, sizeof(dim_t) * 2);
  res = ensureAllocated(grad2dDims);
  if (res != OK) {
    goto cleanup;
  }
  grad2dDims[0] = rows;
  grad2dDims[1] = outputSize;
  Tensor grad2dView = {0};
  res = syncForOpTiming(ctx);
  if (res != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  res = Reshape(ctx, gContig, &grad2dView, (Dim){.dims = grad2dDims, .numOfDims = 2});
  if (res != OK) {
    goto cleanup;
  }

  res = Sum(ctx, &grad2dView, dB, 0);
  if (res != OK) {
    goto cleanup;
  }
  logOpTiming(ctx, "DenseBackward", "bias_grad", phaseStartMs);
  logOpTiming(ctx, "DenseBackward", "total", totalStartMs);

cleanup:
  releaseTensorArg(ctx, &xArg);
  releaseTensorArg(ctx, &wArg);
  releaseTensorArg(ctx, &gArg);
  return res;
}
