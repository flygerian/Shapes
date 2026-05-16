#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "tensor/types.h"
#include "tensor/value.h"
#include "utils_lib/array.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include "nn_internal.h"

void ZeroGrad(Context *ctx, Array *graph) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(graph == NULL, ERR_NULL_PTR);

  for (size_t i = 0; i < graph->size; i++) {
    Tensor *p = Array_TensorIdx(graph, i);
    Tensor *g = p->grad;

    SetValues(g, VALUE(g->dtype, 0));
  }
}

Tensor *Forward(Context *ctx, FowardPassOp *fwdOp, Tensor *input) {
  PANIC_IF(fwdOp->ctx == NULL, NULL_CONTEXT);
  PANIC_IF(fwdOp == NULL, ERR_NULL_PTR);
  PANIC_IF(input == NULL, ERR_NULL_PTR);

  switch (fwdOp->type) {
    case OP_DENSE: return denseForward(ctx, (Layer *)fwdOp->op, input); break;
    case OP_EMBEDDING: return embeddingForward(ctx, (Layer *)fwdOp->op, input);
    case OP_BATCH_NORM: return batchNormForward(ctx, (Layer *)fwdOp->op, input);
    case OP_TANH: return tanhForward(ctx, (Layer *)fwdOp->op, input);
    case OP_RELU: return reluForward(ctx, (Layer *)fwdOp->op, input);
    case OP_MAXPOOL2D: return maxPool2dForward(ctx, (Layer *)fwdOp->op, input);
    case OP_ADAPTIVE_AVG_POOL2D: return adaptiveAvgPool2dForward(ctx, (Layer *)fwdOp->op, input);
    case OP_CONV2D: return conv2dForward(ctx, (Layer *)fwdOp->op, input);
    case OP_SEQUENTIAL: return sequentialModelForward(ctx, fwdOp, input);
    case OP_FLATTEN: return flattenFoward(ctx, fwdOp->op, input);
  }

  PANIC_IF(true, LAYER_OP_NOT_FOUND);
}

Array *Parameters(Context *ctx, FowardPassOp *op) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: return denseLayerParameters(ctx, (Layer *)op->op);
    case OP_EMBEDDING: return embeddingParameters(ctx, (Layer *)op->op);
    case OP_BATCH_NORM: return batchNormLayerParameters(ctx, (Layer *)op->op);
    case OP_TANH: return tanhLayerParameters(ctx, (Layer *)op->op);
    case OP_RELU: return reluLayerParameters(ctx, (Layer *)op->op);
    case OP_MAXPOOL2D: return maxPool2dLayerParameters(ctx, (Layer *)op->op);
    case OP_ADAPTIVE_AVG_POOL2D: return adaptiveAvgPool2dLayerParameters(ctx, (Layer *)op->op);
    case OP_CONV2D: return conv2dLayerParameters(ctx, (Layer *)op->op);
    case OP_SEQUENTIAL: return sequentialModelParameters(ctx, op);
    case OP_FLATTEN: return flattenParameters(ctx);
  }

  PANIC_IF(true, LAYER_OP_NOT_FOUND);
}

void OptimizerStep(Context *ctx, Optimizer *optimizer, Array *parameters) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(optimizer == NULL, ERR_NULL_PTR);

  switch (optimizer->opType) {
    case OP_SGD: sgdStep(ctx, optimizer, parameters); return;
    case OP_ADAM: adamStep(ctx, optimizer, parameters); return;
  }

  PANIC_IF(true, OPTIMIZER_OP_NOT_FOUND);
}
