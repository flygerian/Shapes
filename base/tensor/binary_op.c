#include "common.h"
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

      default:
        return ERR_NOT_A_BINOP;
    }

    VALUE_SET(output->values, x, result);
  }

  *destination = *output;

  return OK;
}

Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_ADD);
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

  // Allocate tensor for denominator^-1 using creation function (avoids stack-use-after-return)
  Tensor *denom_inv = t_Zeros(ctx, denominator->shape, denominator->dtype);
  Result res = Pow(ctx, denominator, -1.0f, denom_inv);
  if (res != OK) {
    return res;
  }

  // Compute numerator * denominator^-1
  return Multiply(ctx, numerator, denom_inv, destination);
}
