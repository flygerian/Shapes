#include "result.h"
#include "shapes_internal.h"
#include <time.h>
#include "shapes.h"

// Build a destination tensor that matches x's rank and leading dims, but swaps
// the last dim (feature width). Dense uses this to preserve any batch axes.
static shapes_Dim swapLastDim(shapes_Context *ctx, shapes_Dim dim, shapes_dim_t lastDim) {
  u8 numDims = dim.numOfDims;
  shapes_dim_t *dims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * numDims);
  PANIC_IF(dims == NULL, ALLOCATION_FAILED);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = dim.dims[i];
  }
  dims[numDims - 1] = lastDim;
  shapes_Dim shape = {.dims = dims, .numOfDims = numDims};

  return shape;
}

static Result validateDenseGradBuffer(shapes_Tensor *grad, shapes_Tensor *reference, shapes_Dtype dtype) {
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

static Result validateDenseBiasGradBuffer(shapes_Tensor *grad, shapes_dim_t outputSize, shapes_Dtype dtype) {
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

static void promoteDenseBackwardInput(shapes_Context *ctx, shapes_Tensor *src, shapes_Tensor *dest) {
  if (src->shape.numOfDims == 1) {
    *dest = shapes_UnSqueeze(ctx, src, 0);
    return;
  }

  *dest = *src;
}

shapes_Tensor shapes_DenseLinear(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *w, shapes_Tensor *b, bool withBias) {
  PANIC_IF(isInvalidTensor(x) || isInvalidTensor(w) || (withBias && isInvalidTensor(b)), ERR_NULL_TENSOR_PROVIDED);

  PANIC_IF(x->shape.numOfDims < 2 || w->shape.numOfDims != 2, ERR_MATMUL_MIN_2D);

  PANIC_IF(x->dtype != w->dtype || (withBias && b->dtype != x->dtype), ERR_DTYPE_MISMATCH);

  PANIC_IF(x->dtype != F16 && x->dtype != F32 && x->dtype != F64, ERR_DTYPE_MISMATCH);

  shapes_dim_t inputSize = x->shape.dims[x->shape.numOfDims - 1];
  shapes_dim_t outputSize = w->shape.dims[0];

  PANIC_IF(w->shape.dims[1] != inputSize, ERR_MATMUL_INNER_DIM_MISMATCH);

  PANIC_IF(withBias && (b->shape.numOfDims != 1 || b->shape.dims[0] != outputSize), ERR_DIM_MISMATCH);
  // BLAS expects dense row-major buffers. Views/slices from Go can be
  // non-contiguous, so we materialize contiguous copies when needed.

  shapes_Tensor *xContig = materializeTensorOnContext(ctx, x);
  shapes_Tensor *wContig = materializeTensorOnContext(ctx, w);

  shapes_tensor_size_t rows = x->size / inputSize;

  shapes_Dim newDims = swapLastDim(ctx, x->shape, outputSize);
  shapes_Tensor out = shapes_MakeZerosTensor(ctx, newDims);

  // Flatten all leading dims into a single "rows" dimension and run:
  // out(rows x outputSize) = x(rows x inputSize) * w^T(inputSize x outputSize)
  runGemm(ctx, x->dtype, CblasNoTrans, CblasTrans, (int)rows, (int)outputSize, (int)inputSize, xContig->values, (int)inputSize, wContig->values, (int)inputSize, false, out.values, (int)outputSize);

  if (withBias) {
    shapes_AddInPlace(ctx, &out, b);
  }

  return out;
}

Result shapes_DenseBackward(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *w, shapes_Tensor *gradOut, shapes_Tensor *dX, shapes_Tensor *dW, shapes_Tensor *dB) {
  // DenseBackward accumulates gradients into preallocated buffers:
  // x: [..., inputSize], w: [outputSize, inputSize], gradOut: [..., outputSize]
  // dX: [..., inputSize], dW: [outputSize, inputSize], dB: [outputSize] or NULL
  if (ctx == NULL || isInvalidTensor(x) || isInvalidTensor(w) || isInvalidTensor(gradOut) || isInvalidTensor(dX) || isInvalidTensor(dW)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  shapes_Tensor x2d = {0};
  promoteDenseBackwardInput(ctx, x, &x2d);

  shapes_Tensor gradOut2d = {0};
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

  shapes_dim_t inputSize = x2d.shape.dims[x2d.shape.numOfDims - 1];
  shapes_dim_t outputSize = w->shape.dims[0];

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

  shapes_tensor_size_t rows = x2d.size / inputSize;
  if (gradOut2d.size != rows * outputSize) {
    return ERR_DIM_MISMATCH;
  }

  // Same contiguous requirement as forward: BLAS kernels consume packed rows.

  shapes_Tensor *xContig = materializeTensorOnContext(ctx, &x2d);
  shapes_Tensor *wContig = materializeTensorOnContext(ctx, w);
  shapes_Tensor *gContig = materializeTensorOnContext(ctx, &gradOut2d);

  shapes_Tensor dX2d = t_Empty(ctx, swapLastDim(ctx, x2d.shape, inputSize), x->dtype);
  shapes_Tensor dWRaw = t_Empty(ctx, swapLastDim(ctx, w->shape, inputSize), w->dtype);

  runGemm(ctx, x->dtype, CblasNoTrans, CblasNoTrans, (int)rows, (int)inputSize, (int)outputSize, gContig->values, (int)outputSize, wContig->values, (int)inputSize, false, dX2d.values, (int)inputSize);

  // dW = gradOut^T * x
  runGemm(ctx, x->dtype, CblasTrans, CblasNoTrans, (int)outputSize, (int)inputSize, (int)rows, gContig->values, (int)outputSize, xContig->values, (int)inputSize, false, dWRaw.values, (int)inputSize);

  shapes_Tensor dXReduced = shapes_ReduceBroadcast(ctx, x, &dX2d);
  shapes_AddInPlace(ctx, dX, &dXReduced);

  shapes_Tensor dWReduced = shapes_ReduceBroadcast(ctx, w, &dWRaw);
  shapes_AddInPlace(ctx, dW, &dWReduced);

  if (dB != NULL) {
    shapes_Tensor dBRaw = shapes_Sum(ctx, gContig, 0);
    shapes_Tensor dBReduced = shapes_ReduceBroadcast(ctx, dB, &dBRaw);
    shapes_AddInPlace(ctx, dB, &dBReduced);
  }

  return OK;
}
