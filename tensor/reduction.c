#include <string.h>
#include "tensor_internal.h"
#include "value.h"
#include "../memory.h"

static Result straightSum(Context *ctx, Tensor *t, Tensor* dest) {
  Value sums[MAX_PARALLEL_SUMS];

  tensor_size_t x = 0;
  while (x < t->size) {
    u8 numComputations = 0;
    for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
      if (x + y >= t->size) {
        break;
      }

      Value val;
      VALUE_GET_FROM_ARR(t->values, x + y, &val, t->dtype);
      VALUE_BINOP(sums[y], sums[y], val, +);
      numComputations++;
      x += numComputations;
    }
  }

  Value sum = VALUE(t->dtype, 0);
  for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
    VALUE_BINOP(sum, sum, sums[y], +);
  }

  dim_t dims[] = {1};
  Tensor *result = t_Zeros(ctx, (Dim) {.dims=dims, .numOfDims=1}, t->dtype);
  VALUE_UNBOX(sum, result->values);
  *dest = *result;
  
  return OK;
}

Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }
 
  if (dim >= t->shape.numOfDims) {
    return ERR_SUM_DIM_OUT_OF_BOUNDS;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    return numBeforeResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest = (Tensor) {
    .dtype=workingTensor->dtype, 
    .isContigous = true, 
    .isView = false, 
    .size = (resultSize), 
    .values=allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * resultSize)
  };

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      Value acc = VALUE(workingTensor->dtype, 0);
      for (dim_t r=0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
      
        Value v;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &v, workingTensor->dtype);
        
        VALUE_BINOP(acc, acc, v, +);
      } 
      VALUE_UNBOX(acc, dest->values + (outer * numAfterDim + inner));
   }
 }

  dest->shape = (Dim) {.numOfDims = workingTensor->shape.numOfDims, .dims=allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims, sizeof(dim_t) * workingTensor->shape.numOfDims);
  dest->shape.dims[dim] = 1;

 if (!t->isContigous) {
   Result freeRes = FreeTensor(ctx, workingTensor);
   if (freeRes != OK) {
     return freeRes;
   }
 }

  return OK;
}
