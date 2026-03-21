#include "shapes.h"

#include "../memory.h"
#include "tensor/tensor_internal.h"

// Build a destination tensor that matches x's rank and leading dims, but swaps
// the last dim (feature width). Dense uses this to preserve any batch axes.
static Result initTensorLikeInputWithLastDim(Context *ctx, Tensor *dest, Tensor *x, dim_t lastDim,
                                             Dtype dtype) {
  u8 numDims = x->shape.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * numDims);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = x->shape.dims[i];
  }
  dims[numDims - 1] = lastDim;

  Dim shape = {.dims = dims, .numOfDims = numDims, .multipliers = multipliers};
  calculateNumValuesAndMultipliers(shape, multipliers);

  return initTensor(ctx, dest, shape, dtype);
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

  tensor_size_t rows = x->size / inputSize;
  res = initTensorLikeInputWithLastDim(ctx, dest, x, outputSize, x->dtype);
  if (res != OK) {
    releaseTensorArg(ctx, &xArg);
    releaseTensorArg(ctx, &wArg);
    return res;
  }

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  runGemm(ctx, x->dtype, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize,
          xContig->values, (int)inputSize, wContig->values, (int)inputSize, false, dest->values,
          (int)outputSize);

  if (withBias) {
    // Add bias per output feature for every flattened row.
    if (x->dtype == F64) {
      double *out = dest->values;
      double *bias = b->values;
      for (tensor_size_t r = 0; r < rows; r++) {
        tensor_size_t base = r * outputSize;
        for (tensor_size_t c = 0; c < outputSize; c++) {
          out[base + c] += bias[c];
        }
      }
    } else {
      float *out = dest->values;
      float *bias = b->values;
      for (tensor_size_t r = 0; r < rows; r++) {
        tensor_size_t base = r * outputSize;
        for (tensor_size_t c = 0; c < outputSize; c++) {
          out[base + c] += bias[c];
        }
      }
    }
  }

  releaseTensorArg(ctx, &xArg);
  releaseTensorArg(ctx, &wArg);

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

  res = initTensorLikeInputWithLastDim(ctx, dX, x, inputSize, x->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = initTensorLikeInputWithLastDim(ctx, dW, w, inputSize, w->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = init1DTensor(ctx, dB, outputSize, gradOut->dtype);
  if (res != OK) {
    goto cleanup;
  }

  if (x->dtype == F64) {
    // dX = gradOut * w
    runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize,
            gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX->values,
            (int)inputSize);

    // dW = gradOut^T * x
    runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows,
            gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dW->values,
            (int)inputSize);

    // dB is the row-wise sum of gradOut (one accumulator per output feature).
    double *gVals = gContig->values;
    double *dbVals = dB->values;
    for (tensor_size_t j = 0; j < outputSize; j++) {
      dbVals[j] = 0.0;
    }
    for (tensor_size_t r = 0; r < rows; r++) {
      tensor_size_t base = r * outputSize;
      for (tensor_size_t j = 0; j < outputSize; j++) {
        dbVals[j] += gVals[base + j];
      }
    }
  } else {
    // dX = gradOut * w
    runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize,
            gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX->values,
            (int)inputSize);

    // dW = gradOut^T * x
    runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows,
            gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dW->values,
            (int)inputSize);

    // dB is the row-wise sum of gradOut (one accumulator per output feature).
    float *gVals = gContig->values;
    float *dbVals = dB->values;
    for (tensor_size_t j = 0; j < outputSize; j++) {
      dbVals[j] = 0.0f;
    }
    for (tensor_size_t r = 0; r < rows; r++) {
      tensor_size_t base = r * outputSize;
      for (tensor_size_t j = 0; j < outputSize; j++) {
        dbVals[j] += gVals[base + j];
      }
    }
  }

cleanup:
  releaseTensorArg(ctx, &xArg);
  releaseTensorArg(ctx, &wArg);
  releaseTensorArg(ctx, &gArg);
  return res;
}
