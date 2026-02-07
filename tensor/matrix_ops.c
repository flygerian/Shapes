#include "tensor_internal.h"
#include "blas.h"
#include "../memory.h"
#include "value.h"

static bool areBatchDimsBroadcastable(Tensor *a, Tensor *b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;

  for (int d = 2; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;

    dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;

    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return false;
    }
  }

  return true;
}

Result MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor *result) {
  if (isInvalidTensor(a) || isInvalidTensor(b)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if ((a->shape.numOfDims < 2 || b->shape.numOfDims < 2)) {
    return ERR_MATMUL_MIN_2D;
  }

  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (a->dtype != F32 && a->dtype != F64 && a->dtype != F16) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t innerDimA = a->shape.dims[a->shape.numOfDims - 1];
  dim_t innerDimB = b->shape.dims[b->shape.numOfDims - 2];

  if (innerDimA != innerDimB) {
    return ERR_MATMUL_INNER_DIM_MISMATCH;
  }

  if (!areBatchDimsBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  Tensor *opA = ops.a;
  Tensor *opB = ops.b;

  if (!opA->isContigous) {
    opA = copyToContiguous(ctx, opA);
  }

  if (!opB->isContigous) {
    opB = copyToContiguous(ctx, opB);
  }

  dim_t m = opA->shape.dims[opA->shape.numOfDims - 2];
  dim_t k = opB->shape.dims[opB->shape.numOfDims - 2];
  dim_t n = opB->shape.dims[opB->shape.numOfDims - 1];

  tensor_size_t batchSizeA, batchSizeB;
  dim_t batchDimIdx = opA->shape.numOfDims - 2;
  Result r = calculateNumElementsBeforeDim(opA, batchDimIdx, &batchSizeA);
  if (r != OK)
    return r;
  r = calculateNumElementsBeforeDim(opB, batchDimIdx, &batchSizeB);
  if (r != OK)
    return r;

  tensor_size_t batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

  Tensor *sentinel = batchSizeA >= batchSizeB ? opA : opB;
  Dim newDim = (Dim){.dims = allocate(ctx->memory, sizeof(dim_t) * sentinel->shape.numOfDims),
                     .numOfDims = sentinel->shape.numOfDims};
  r = getDimsBefore(ctx, sentinel, batchDimIdx, &newDim);
  if (r != OK)
    return r;

  newDim.dims[newDim.numOfDims - 2] = m;
  newDim.dims[newDim.numOfDims - 1] = n;

  newDim.multipliers = allocate(ctx->memory, sizeof(multiplier_t) * newDim.numOfDims);
  tensor_size_t size = calculateNumValuesAndMultipliers(newDim, newDim.multipliers);

  *result = (Tensor){.dtype = a->dtype,
                     .isContigous = true,
                     .isView = false,
                     .shape = newDim,
                     .size = size,
                     .values = allocate(ctx->memory, getBytesForDtype(opA->dtype) * size)};

  size_t elemSize = getBytesForDtype(opA->dtype);
  for (tensor_size_t i = 0; i < batchSize; i++) {
    tensor_size_t aIdx = i % batchSizeA;
    tensor_size_t bIdx = i % batchSizeB;
    void *A_batch = (char *)opA->values + aIdx * (m * k) * elemSize;
    void *B_batch = (char *)opB->values + bIdx * (k * n) * elemSize;
    void *C_batch = (char *)result->values + i * (m * n) * elemSize;

    BLAS_GEMM(opA->dtype, A_batch, B_batch, C_batch, m, n, k);
  }

  return OK;
}

Result Dot(Context *ctx, Tensor *a, Tensor *b, Tensor *result) {
  if (isInvalidTensor(a) || isInvalidTensor(b)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (a->shape.numOfDims != 1 || b->shape.numOfDims != 1) {
    return ERR_DIM_MISMATCH;
  }

  if (a->size != b->size) {
    return ERR_DIM_MISMATCH;
  }

  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (a->dtype != F32 && a->dtype != F64 && a->dtype != F16) {
    return ERR_DTYPE_MISMATCH;
  }

  Tensor *opA = a;
  Tensor *opB = b;

  if (!opA->isContigous) {
    opA = copyToContiguous(ctx, opA);
  }
  if (!opB->isContigous) {
    opB = copyToContiguous(ctx, opB);
  }

  dim_t *resDims = allocate(ctx->memory, sizeof(dim_t));
  resDims[0] = 1;
  multiplier_t *resMult = allocate(ctx->memory, sizeof(multiplier_t));
  resMult[0] = 1;

  void *resVal = allocate(ctx->memory, getBytesForDtype(a->dtype));

  BLAS_DOT(a->dtype, (int)a->size, opA->values, opB->values, resVal);

  *result = (Tensor){.dtype = a->dtype,
                     .isContigous = true,
                     .isView = false,
                     .boundary = NULL,
                     .size = 1,
                     .values = resVal,
                     .shape = (Dim){.dims = resDims, .numOfDims = 1, .multipliers = resMult}};

  return OK;
}
