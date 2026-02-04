#include "backwards.h"
#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"

GraphNode contructGraphNode(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result) {
  GraphNode node = {
    .optype = type,
    .grad = t_Zeros(ctx, result->shape, result->dtype),
    .inputs = allocate(ctx->memory, sizeof(Tensor) * 2),
    .numInputs = 2
  };

  node.inputs[0] = a;
  node.inputs[1] = b;

  return node;
}

Result add_backward(Context *ctx, GraphNode *node) {
  return OK;
}

Result constructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result) {
  GraphNode node = contructGraphNode(ctx, type, a, b, result);

  switch (type) {
    case OP_ADD:
      node.backward = add_backward;
      break;
  }


}

