#include "result/result.h"
#include "shapes.h"

#include "tensor_internal.h"
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
    *dest = shapes_UnSqueeze(ctx, src, 0);
    return;
  }

  *dest = *src;
}

Tensor shapes_layer_DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias) {
  // Dense expects:
  // x: [..., inputSize]
  // w: [outputSize, inputSize]
  // b: [outputSize] (optional)
  // out: [..., outputSize]
  PANIC_IF(isInvalidTensor(x) || isInvalidTensor(w) || (withBias && isInvalidTensor(b)), ERR_NULL_TENSOR_PROVIDED);

  PANIC_IF(x->shape.numOfDims < 2 || w->shape.numOfDims != 2, ERR_MATMUL_MIN_2D);

  PANIC_IF(x->dtype != w->dtype || (withBias && b->dtype != x->dtype), ERR_DTYPE_MISMATCH);

  PANIC_IF(x->dtype != F16 && x->dtype != F32 && x->dtype != F64, ERR_DTYPE_MISMATCH);

  dim_t inputSize = x->shape.dims[x->shape.numOfDims - 1];
  dim_t outputSize = w->shape.dims[0];

  PANIC_IF(w->shape.dims[1] != inputSize, ERR_MATMUL_INNER_DIM_MISMATCH);

  PANIC_IF(withBias && (b->shape.numOfDims != 1 || b->shape.dims[0] != outputSize), ERR_DIM_MISMATCH);
  // BLAS expects dense row-major buffers. Views/slices from Go can be
  // non-contiguous, so we materialize contiguous copies when needed.

  Tensor *xContig = materializeTensorOnContext(ctx, x);
  Tensor *wContig = materializeTensorOnContext(ctx, w);

  tensor_size_t rows = x->size / inputSize;

  Dim newDims = swapLastDim(ctx, x->shape, outputSize);
  Tensor out = shapes_Make_ZerosTensor(ctx, newDims);

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  runGemm(ctx, x->dtype, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize, xContig->values, (int)inputSize, wContig->values, (int)inputSize, false, out.values, (int)outputSize);

  if (withBias) {
    shapes_AddInPlace(ctx, &out, b);
  }

  return out;
}

Result shapes_layer_DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW, Tensor *dB) {
  // DenseBackward accumulates gradients into preallocated buffers:
  // x: [..., inputSize], w: [outputSize, inputSize], gradOut: [..., outputSize]
  // dX: [..., inputSize], dW: [outputSize, inputSize], dB: [outputSize] or NULL
  if (ctx == NULL || isInvalidTensor(x) || isInvalidTensor(w) || isInvalidTensor(gradOut) || isInvalidTensor(dX) || isInvalidTensor(dW)) {
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

  Tensor *xContig = materializeTensorOnContext(ctx, &x2d);
  Tensor *wContig = materializeTensorOnContext(ctx, w);
  Tensor *gContig = materializeTensorOnContext(ctx, &gradOut2d);

  Tensor dX2d = t_Empty(ctx, swapLastDim(ctx, x2d.shape, inputSize), x->dtype);
  Tensor dWRaw = t_Empty(ctx, swapLastDim(ctx, w->shape, inputSize), w->dtype);

  runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize, gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX2d.values, (int)inputSize);

  // dW = gradOut^T * x
  runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows, gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dWRaw.values, (int)inputSize);

  Tensor dXReduced = shapes_ReduceBroadcast(ctx, x, &dX2d);
  shapes_AddInPlace(ctx, dX, &dXReduced);

  Tensor dWReduced = shapes_ReduceBroadcast(ctx, w, &dWRaw);
  shapes_AddInPlace(ctx, dW, &dWReduced);

  if (dB != NULL) {
    Tensor dBRaw = shapes_Sum(ctx, gContig, 0);
    Tensor dBReduced = shapes_ReduceBroadcast(ctx, dB, &dBRaw);
    shapes_AddInPlace(ctx, dB, &dBReduced);
  }

  return OK;
}
