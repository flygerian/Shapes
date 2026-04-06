#include "common.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "blas.h"
#include <sched.h>
#include <stdlib.h>

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

  Tensor *opA = materializeTensorOnContext(ctx, ops.a);
  Tensor *opB = materializeTensorOnContext(ctx, ops.b);

  dim_t m = opA->shape.dims[opA->shape.numOfDims - 2];
  dim_t k = opB->shape.dims[opB->shape.numOfDims - 2];
  dim_t n = opB->shape.dims[opB->shape.numOfDims - 1];

  tensor_size_t batchSizeA, batchSizeB;
  dim_t batchDimIdx = opA->shape.numOfDims - 2;
  Result r = calculateNumElementsBeforeDim(opA, batchDimIdx, &batchSizeA);
  if (r != OK) {
    goto cleanup;
  }

  r = calculateNumElementsBeforeDim(opB, batchDimIdx, &batchSizeB);
  if (r != OK) {
    goto cleanup;
  }

  tensor_size_t batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

  Tensor *sentinel = batchSizeA >= batchSizeB ? opA : opB;
  Dim newDim = (Dim){.dims = allocate(ctx->memory, sizeof(dim_t) * sentinel->shape.numOfDims),
                     .numOfDims = sentinel->shape.numOfDims};

  r = ensureAllocated(newDim.dims);
  if (r != OK) {
    goto cleanup;
  }

  r = getDimsBefore(ctx, sentinel, batchDimIdx, &newDim);
  if (r != OK) {
    goto cleanup;
  }

  newDim.dims[newDim.numOfDims - 2] = m;
  newDim.dims[newDim.numOfDims - 1] = n;

  newDim.multipliers = allocate(ctx->memory, sizeof(multiplier_t) * newDim.numOfDims);
  r = ensureAllocated(newDim.multipliers);
  if (r != OK) {
    freeAlloc(ctx->memory, newDim.dims);
    goto cleanup;
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDim.dims, newDim.numOfDims);
  void *values = allocateOnCtx(ctx, getBytesForDtype(opA->dtype) * snm.size);
  PANIC_IF(values == NULL, ALLOCATION_FAILED);

  *result = (Tensor){.context = ctx,
                     .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                     .dtype = a->dtype,
                     .isContigous = true,
                     .isView = false,
                     .shape = newDim,
                     .size = snm.size,
                     .values = values};

  size_t elemSize = getBytesForDtype(opA->dtype);
  for (tensor_size_t i = 0; i < batchSize; i++) {
    tensor_size_t aIdx = i % batchSizeA;
    tensor_size_t bIdx = i % batchSizeB;
    void *A_batch = (char *)opA->values + aIdx * (m * k) * elemSize;
    void *B_batch = (char *)opB->values + bIdx * (k * n) * elemSize;
    void *C_batch = (char *)result->values + i * (m * n) * elemSize;

    runGemm(ctx, opA->dtype, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, A_batch, (int)k,
            B_batch, (int)n, false, C_batch, (int)n);
  }

cleanup:
  freeIfContingousCopy(ctx, opA);
  freeIfContingousCopy(ctx, opB);
  return r;
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

  Tensor *opA = materializeTensorOnContext(ctx, a);
  Tensor *opB = materializeTensorOnContext(ctx, b);

  dim_t *resDims = allocate(ctx->memory, sizeof(dim_t));
  PANIC_IF(resDims == NULL, ALLOCATION_FAILED);
  resDims[0] = 1;

  multiplier_t *resMult = allocate(ctx->memory, sizeof(multiplier_t));
  PANIC_IF(resMult == NULL, ALLOCATION_FAILED);
  resMult[0] = 1;

  void *resVal = allocateOnCtx(ctx, getBytesForDtype(a->dtype));
  PANIC_IF(resVal == NULL, ALLOCATION_FAILED);

  BLAS_DOT(a->dtype, (int)a->size, opA->values, opB->values, resVal);

  *result = (Tensor){.context = ctx,
                     .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                     .dtype = a->dtype,
                     .isContigous = true,
                     .isView = false,
                     .boundary = NULL,
                     .size = 1,
                     .values = resVal,
                     .shape = (Dim){.dims = resDims, .numOfDims = 1, .multipliers = resMult}};

  freeIfContingousCopy(ctx, opA);
  freeIfContingousCopy(ctx, opB);

  return OK;
}
