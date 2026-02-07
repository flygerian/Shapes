#include "grad.h"
#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"

static GraphNode *contructGraphNode(Context *ctx, OpType type, Tensor *a, Tensor *b,
                                    Tensor *result) {
  GraphNode *node = allocate(ctx->memory, sizeof(GraphNode));

  node->optype = type, node->grad = t_Zeros(ctx, result->shape, result->dtype),
  node->inputs = allocate(ctx->memory, sizeof(Tensor) * 2), node->numInputs = 2;

  node->inputs[0] = a;
  node->inputs[1] = b;

  return node;
}

Result addBackward(Context *ctx, GraphNode *node) {
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
