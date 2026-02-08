#include "grad.h"
#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor/tensor_internal.h"

static GraphNode *contructGraphNode(Context *ctx, OpType type, Tensor *a, Tensor *b,
                                    Tensor *result) {
  GraphNode *node = allocate(ctx->memory, sizeof(GraphNode));

  node->optype = type;
  node->grad = t_Zeros(ctx, result->shape, result->dtype);
  node->inputs = (Tensor **)allocate(ctx->memory, sizeof(Tensor *) * 2);
  node->numInputs = 2;

  node->inputs[0] = a;
  node->inputs[1] = b;

  return node;
}

// Reduces gradient to match input shape when broadcasting occurred
static Result reduceGradForInput(Context *ctx, Tensor *input, Tensor *outputGrad, Tensor **reducedGrad) {
  Context noGradCtx = NoGradContext(ctx);
  Tensor *current = outputGrad;

  // Handle dimension mismatch: sum along leading dimensions if output has more dims than input
  u8 dimDiff = outputGrad->shape.numOfDims - input->shape.numOfDims;
  for (u8 i = 0; i < dimDiff; i++) {
    Tensor *reduced = allocate(ctx->memory, sizeof(Tensor));
    Result res = Sum(&noGradCtx, current, reduced, 0);
    if (res != OK) {
      return res;
    }
    current = reduced;
  }

  // After reducing leading dimensions, we now have the same number of dimensions as input
  // (though some may still be size 1 from the Sum operation). Now handle size-1 broadcasting:
  // sum along dimensions where input has size 1 but current grad has size > 1

  // Calculate the offset due to leading dimension reductions
  // After dimDiff reductions, leading dims in current are size 1
  u8 offset = dimDiff;

  for (u8 input_d = 0; input_d < input->shape.numOfDims; input_d++) {
    u8 current_d = input_d + offset;
    if (current_d < current->shape.numOfDims &&
        input->shape.dims[input_d] == 1 && current->shape.dims[current_d] > 1) {
      Tensor *reduced = allocate(ctx->memory, sizeof(Tensor));
      Result res = Sum(&noGradCtx, current, reduced, current_d);
      if (res != OK) {
        return res;
      }
      current = reduced;
    }
  }

  *reducedGrad = current;
  return OK;
}

Result addBackward(Context *ctx, GraphNode *node) {
  Tensor *a = node->inputs[0];
  Tensor *b = node->inputs[1];

  Context noGradCtx = NoGradContext(ctx);

  // Reduce gradient for input a if broadcasting occurred
  Tensor *gradA;
  Result res = reduceGradForInput(ctx, a, node->grad, &gradA);
  if (res != OK) {
    return res;
  }

  // Accumulate gradient to input a
  res = Add(&noGradCtx, a->computation->grad, gradA, a->computation->grad);
  if (res != OK) {
    return res;
  }

  // Reduce gradient for input b if broadcasting occurred
  Tensor *gradB;
  res = reduceGradForInput(ctx, b, node->grad, &gradB);
  if (res != OK) {
    return res;
  }

  // Accumulate gradient to input b
  res = Add(&noGradCtx, b->computation->grad, gradB, b->computation->grad);
  if (res != OK) {
    return res;
  }

  return OK;
}

Result subtractBackward(Context *ctx, GraphNode *node) {
  return OK;
}

Result divideBackward(Context *ctx, GraphNode *node) {
  return OK;
}

Result multiplyBackward(Context *ctx, GraphNode *node) {
  return OK;
}

Result ConstructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result) {
  GraphNode *node = contructGraphNode(ctx, type, a, b, result);

  switch (type) {
    case OP_ADD: node->backward = addBackward; break;

    case OP_SUBTRACT: node->backward = subtractBackward; break;

    case OP_DIVIDE: node->backward = divideBackward; break;

    case OP_MULTIPLY: node->backward = multiplyBackward; break;
  }

  result->computation = node;

  return OK;
}
