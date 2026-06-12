#include "shapes_internal.h"
#include "shapes.h"
#include <sched.h>
#include <stdlib.h>

static bool areBatchDimsBroadcastable(shapes_Tensor *a, shapes_Tensor *b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;

  for (int d = 2; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;

    shapes_dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    shapes_dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;

    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return false;
    }
  }

  return true;
}

shapes_Tensor shapes_MatMul(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b) {
  PANIC_IF(isInvalidTensor(a) || isInvalidTensor(b), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(a->shape.numOfDims < 2 || b->shape.numOfDims < 2, ERR_MATMUL_MIN_2D);
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(a->dtype != F32 && a->dtype != F64 && a->dtype != F16, ERR_DTYPE_MISMATCH);

  shapes_dim_t innerDimA = a->shape.dims[a->shape.numOfDims - 1];
  shapes_dim_t innerDimB = b->shape.dims[b->shape.numOfDims - 2];
  PANIC_IF(innerDimA != innerDimB, ERR_MATMUL_INNER_DIM_MISMATCH);
  PANIC_IF(!areBatchDimsBroadcastable(a, b), ERR_DIM_MISMATCH);

  shapes_TensorPair ops = {.a = *a, .b = *b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  shapes_Tensor *opA = materializeTensorOnContext(ctx, &ops.a);
  shapes_Tensor *opB = materializeTensorOnContext(ctx, &ops.b);

  shapes_dim_t m = opA->shape.dims[opA->shape.numOfDims - 2];
  shapes_dim_t k = opB->shape.dims[opB->shape.numOfDims - 2];
  shapes_dim_t n = opB->shape.dims[opB->shape.numOfDims - 1];

  shapes_tensor_size_t batchSizeA, batchSizeB;
  shapes_dim_t batchDimIdx = opA->shape.numOfDims - 2;
  Result r = calculateNumElementsBeforeDim(opA, batchDimIdx, &batchSizeA);
  PANIC_IF(r != OK, r);

  r = calculateNumElementsBeforeDim(opB, batchDimIdx, &batchSizeB);
  PANIC_IF(r != OK, r);

  shapes_tensor_size_t batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

  shapes_Tensor *sentinel = batchSizeA >= batchSizeB ? opA : opB;
  shapes_Dim newDim = (shapes_Dim){
      .dims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * sentinel->shape.numOfDims),
      .numOfDims = sentinel->shape.numOfDims};
  PANIC_IF(newDim.dims == NULL, ALLOCATION_FAILED);

  r = getDimsBefore(ctx, sentinel, batchDimIdx, &newDim);
  PANIC_IF(r != OK, r);

  newDim.dims[newDim.numOfDims - 2] = m;
  newDim.dims[newDim.numOfDims - 1] = n;

  shapes_Tensor result = t_Empty(ctx, newDim, opA->dtype);

  size_t elemSize = getBytesForDtype(opA->dtype);
  for (shapes_tensor_size_t i = 0; i < batchSize; i++) {
    shapes_tensor_size_t aIdx = i % batchSizeA;
    shapes_tensor_size_t bIdx = i % batchSizeB;
    void *A_batch = (char *)opA->values + aIdx * (m * k) * elemSize;
    void *B_batch = (char *)opB->values + bIdx * (k * n) * elemSize;
    void *C_batch = (char *)result.values + i * (m * n) * elemSize;

    runGemm(ctx, opA->dtype, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, A_batch, (int)k,
            B_batch, (int)n, false, C_batch, (int)n);
  }

  return result;
}

shapes_Tensor shapes_Dot(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b) {
  PANIC_IF(isInvalidTensor(a) || isInvalidTensor(b), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(a->shape.numOfDims != 1 || b->shape.numOfDims != 1, ERR_DIM_MISMATCH);
  PANIC_IF(a->size != b->size, ERR_DIM_MISMATCH);
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(a->dtype != F32 && a->dtype != F64 && a->dtype != F16, ERR_DTYPE_MISMATCH);

  shapes_Tensor opA = *materializeTensorOnContext(ctx, a);
  shapes_Tensor opB = *materializeTensorOnContext(ctx, b);

  shapes_Tensor result = t_Empty(ctx, SHAPE1D(1), a->dtype);

  switch (a->dtype) {
    case F64:
      *((double *)result.values) = cblas_ddot((int)a->size, (double *)opA.values, 1, (double *)opB.values, 1);
      break;
    case F32:
    case F16:
      *((float *)result.values) = cblas_sdot((int)a->size, (float *)opA.values, 1, (float *)opB.values, 1);
      break;
    default:
      break;
  }

  return result;
}
