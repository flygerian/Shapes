#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "value.h"
#include "array.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include "nn_internal.h"

void shapesnn_ZeroGrad(shapes_Context *ctx, olib_Array *graph) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(graph == NULL, ERR_NULL_PTR);

  for (size_t i = 0; i < graph->size; i++) {
    shapes_Tensor p = shapes_ArrayTensorIdx(graph, i);
    shapes_Tensor *g = p.grad;

    shapes_SetValues(g, VALUE(g->dtype, 0));
  }
}

shapes_Tensor shapesnn_Forward(shapes_Context *ctx, shapesnn_FowardPassOp *fwdOp, shapes_Tensor *input) {
  PANIC_IF(fwdOp->ctx == NULL, NULL_CONTEXT);
  PANIC_IF(fwdOp == NULL, ERR_NULL_PTR);
  PANIC_IF(input == NULL, ERR_NULL_PTR);

  switch (fwdOp->type) {
    case OP_DENSE: return denseForward(ctx, (shapesnn_layer *)fwdOp->op, input); break;
    case OP_EMBEDDING: return embeddingForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_BATCH_NORM: return batchNormForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_TANH: return tanhForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_RELU: return reluForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_MAXPOOL2D: return maxPool2dForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_ADAPTIVE_AVG_POOL2D: return adaptiveAvgPool2dForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_CONV2D: return conv2dForward(ctx, (shapesnn_layer *)fwdOp->op, input);
    case OP_SEQUENTIAL: return sequentialModelForward(ctx, fwdOp, input);
    case OP_FLATTEN: return flattenFoward(ctx, fwdOp->op, input);
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

olib_Array *shapesnn_Parameters(shapes_Context *ctx, shapesnn_FowardPassOp *op) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(op == NULL, ERR_NULL_PTR);

  switch (op->type) {
    case OP_DENSE: return denseLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_EMBEDDING: return embeddingParameters(ctx, (shapesnn_layer *)op->op);
    case OP_BATCH_NORM: return batchNormLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_TANH: return tanhLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_RELU: return reluLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_MAXPOOL2D: return maxPool2dLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_ADAPTIVE_AVG_POOL2D: return adaptiveAvgPool2dLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_CONV2D: return conv2dLayerParameters(ctx, (shapesnn_layer *)op->op);
    case OP_SEQUENTIAL: return sequentialModelParameters(ctx, op);
    case OP_FLATTEN: return flattenParameters(ctx);
    default: PANIC_WITH_CODE(LAYER_OP_NOT_FOUND);
  }
}

void shapesnn_OptimizerStep(shapes_Context *ctx, shapesnn_Optimizer *optimizer, olib_Array *parameters) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(optimizer == NULL, ERR_NULL_PTR);

  switch (optimizer->opType) {
    case OP_SGD: sgdStep(ctx, optimizer, parameters); return;
    case OP_ADAM: adamStep(ctx, optimizer, parameters); return;
    default: PANIC_WITH_CODE(OPTIMIZER_OP_NOT_FOUND);
  }
}
