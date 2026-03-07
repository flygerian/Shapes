#include "dense.h"

#include "../memory.h"
#include "tensor/blas.h"
#include "tensor/tensor_internal.h"

// Build a destination tensor that matches x's rank and leading dims, but swaps
// the last dim (feature width). Dense uses this to preserve any batch axes.
static Result initTensorLikeInputWithLastDim(Context *ctx, Tensor *dest, Tensor *x, dim_t lastDim,
                                             Dtype dtype) {
  u8 numDims = x->shape.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * numDims);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = x->shape.dims[i];
  }
  dims[numDims - 1] = lastDim;

  Dim shape = {.dims = dims, .numOfDims = numDims, .multipliers = multipliers};
  tensor_size_t size = calculateNumValuesAndMultipliers(shape, multipliers);

  *dest = (Tensor){.dtype = dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

static Result init1DTensor(Context *ctx, Tensor *dest, dim_t size, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  u8 *multipliers = allocate(ctx->memory, sizeof(u8));
  dims[0] = size;
  multipliers[0] = 1;

  *dest = (Tensor){.dtype = dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = (Dim){.dims = dims, .numOfDims = 1, .multipliers = multipliers},
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};
  return OK;
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
  Tensor *xContig = x;
  Tensor *wContig = w;
  if (!xContig->isContigous) {
    xContig = copyToContiguous(ctx, xContig);
  }
  if (!wContig->isContigous) {
    wContig = copyToContiguous(ctx, wContig);
  }

  tensor_size_t rows = x->size / inputSize;
  Result res = initTensorLikeInputWithLastDim(ctx, dest, x, outputSize, x->dtype);
  if (res != OK) {
    if (xContig != x) {
      FreeTensor(ctx, xContig);
    }
    if (wContig != w) {
      FreeTensor(ctx, wContig);
    }
    return res;
  }

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  if (x->dtype == F64) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize,
                1.0, (double *)xContig->values, (int)inputSize, (double *)wContig->values,
                (int)inputSize, 0.0, (double *)dest->values, (int)outputSize);
  } else {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize,
                1.0f, (float *)xContig->values, (int)inputSize, (float *)wContig->values,
                (int)inputSize, 0.0f, (float *)dest->values, (int)outputSize);
  }

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

  if (xContig != x) {
    FreeTensor(ctx, xContig);
  }
  if (wContig != w) {
    FreeTensor(ctx, wContig);
  }

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
  Tensor *xContig = x;
  Tensor *wContig = w;
  Tensor *gContig = gradOut;
  if (!xContig->isContigous) {
    xContig = copyToContiguous(ctx, xContig);
  }
  if (!wContig->isContigous) {
    wContig = copyToContiguous(ctx, wContig);
  }
  if (!gContig->isContigous) {
    gContig = copyToContiguous(ctx, gContig);
  }

  Result res = initTensorLikeInputWithLastDim(ctx, dX, x, inputSize, x->dtype);
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize,
                (int)outputSize, 1.0, (double *)gContig->values, (int)outputSize,
                (double *)wContig->values, (int)inputSize, 0.0, (double *)dX->values,
                (int)inputSize);

    // dW = gradOut^T * x
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize,
                (int)rows, 1.0, (double *)gContig->values, (int)outputSize,
                (double *)xContig->values, (int)inputSize, 0.0, (double *)dW->values,
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize,
                (int)outputSize, 1.0f, (float *)gContig->values, (int)outputSize,
                (float *)wContig->values, (int)inputSize, 0.0f, (float *)dX->values,
                (int)inputSize);

    // dW = gradOut^T * x
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize,
                (int)rows, 1.0f, (float *)gContig->values, (int)outputSize,
                (float *)xContig->values, (int)inputSize, 0.0f, (float *)dW->values,
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
  if (xContig != x) {
    FreeTensor(ctx, xContig);
  }
  if (wContig != w) {
    FreeTensor(ctx, wContig);
  }
  if (gContig != gradOut) {
    FreeTensor(ctx, gContig);
  }
  return res;
}
