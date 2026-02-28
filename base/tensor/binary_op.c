#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "value.h"
#include "unary.h"

static Result binaryOp(Context *ctx, Tensor *a, Tensor *b, Tensor *destination, OpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
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

  Dim outputShape;
  if (opA->size > opB->size) {
    outputShape = opA->shape;
  } else {
    outputShape = opB->shape;
  }

  Tensor *output = t_Zeros(ctx, outputShape, opA->dtype);

  dim_t currentCoord[output->shape.numOfDims];
  dim_t aCoords[output->shape.numOfDims];
  dim_t bCoords[output->shape.numOfDims];

  for (tensor_size_t x = 0; x < output->size; x++) {
    unravel_index(x, &outputShape, currentCoord);

    for (u8 d = 0; d < output->shape.numOfDims; d++) {
      aCoords[d] = currentCoord[d] % opA->shape.dims[d];
      bCoords[d] = currentCoord[d] % opB->shape.dims[d];
    }

    Value aVal;
    u64 idx = getContigousIdxFromCoord(opA, aCoords);
    VALUE_GET_FROM_ARR(opA->values, idx, &aVal, opA->dtype);

    Value bVal;
    idx = getContigousIdxFromCoord(opB, bCoords);
    VALUE_GET_FROM_ARR(opB->values, idx, &bVal, opB->dtype);

    Value result;
    switch (opType) {
      case OP_ADD: VALUE_BINOP(result, aVal, bVal, +); break;
      case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
      case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;

      default: return ERR_NOT_A_BINOP;
    }

    VALUE_SET(output->values, x, result);
  }

  *destination = *output;
  freeAlloc(ctx->memory, output);

  if (opA != ops.a)
    FreeTensor(ctx, opA);
  if (opB != ops.b)
    FreeTensor(ctx, opB);
  if (ops.a != a)
    FreeViewTensor(ctx, ops.a);
  if (ops.b != b)
    FreeViewTensor(ctx, ops.b);

  return OK;
}

Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_ADD);
}

Result AddInPlace(Context *ctx, Tensor *a, Tensor *b) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *opB = b;
  Tensor *paddedB = NULL;
  if (a->shape.numOfDims != b->shape.numOfDims) {
    TensorPair ops = padSmallerTensor(ctx, a, b);
    opB = ops.b;
    paddedB = ops.b;
  }

  Tensor *contiguousB = NULL;
  if (!opB->isContigous) {
    opB = copyToContiguous(ctx, opB);
    contiguousB = opB;
  }

  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];

  for (tensor_size_t x = 0; x < a->size; x++) {
    // Unravel using a's logical shape (ignoring boundary).
    unravel_index(x, &a->shape, currentCoord);

    // Compute storage index in a (accounts for per-dim boundary via getContigousIdxFromCoord).
    u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);

    for (u8 d = 0; d < a->shape.numOfDims; d++) {
      bCoords[d] = currentCoord[d] % opB->shape.dims[d];
    }

    Value aVal;
    VALUE_GET_FROM_ARR(a->values, aStorageIdx, &aVal, a->dtype);

    Value bVal;
    u64 bIdx = getContigousIdxFromCoord(opB, bCoords);
    VALUE_GET_FROM_ARR(opB->values, bIdx, &bVal, opB->dtype);

    Value result;
    VALUE_BINOP(result, aVal, bVal, +);

    VALUE_SET(a->values, aStorageIdx, result);
  }

  if (contiguousB != NULL)
    FreeTensor(ctx, contiguousB);
  if (paddedB != NULL)
    FreeViewTensor(ctx, paddedB);

  return OK;
}

Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_SUBTRACT);
}

Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_MULTIPLY);
}

Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination) {
  // Implement division as: numerator / denominator = numerator * (denominator^-1)
  // This automatically gets correct gradients through the computation graph!

  // Pow writes into dest by assigning a newly allocated tensor payload. Allocate only the
  // container struct here so we don't leak a preallocated payload on overwrite.
  Tensor *denom_inv = allocate(ctx->memory, sizeof(Tensor));
  Result res = Pow(ctx, denominator, -1.0f, denom_inv);
  if (res != OK) {
    freeAlloc(ctx->memory, denom_inv);
    return res;
  }

  // Compute numerator * denominator^-1
  res = Multiply(ctx, numerator, denom_inv, destination);
  FreeTensor(ctx, denom_inv);
  return res;
}
