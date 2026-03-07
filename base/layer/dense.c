#include "dense.h"

#include "../memory.h"
#include "tensor/blas.h"
#include "tensor/tensor_internal.h"

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

Result DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor *dest) {
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
