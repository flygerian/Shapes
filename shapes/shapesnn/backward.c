#include "result.h"
#include "shapes.h"
#include "array.h"
#include "map.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "nn_internal.h"

olib_Array *buildGraph(Context *ctx, Tensor *tensor);
void topoSort(olib_Array *graph, PtrSet *visited, Tensor *tensor);

void backward(Context *ctx, Tensor *node) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(node == NULL, NULL_CONTEXT);

  switch (node->opType) {
    case OP_NONE: return;
    case OP_DENSE: denseBackward(ctx, node); return;
    case OP_EMBEDDING: embeddingBackward(ctx, node); return;
    case OP_RESHAPE: shapes_ReshapeBackward(ctx, node); return;
    case OP_MSE: mseBackward(ctx, node); return;
    case OP_BATCH_NORM: batchnormBackward(ctx, node); return;
    case OP_CROSS_ENTHROPY: crossEnthropyBackward(ctx, node); return;
    case OP_TANH: tanhBackward(ctx, node); return;
    case OP_RELU: reluBackward(ctx, node); return;
    case OP_MAXPOOL2D: maxPool2dBackward(ctx, node); return;
    case OP_ADAPTIVE_AVG_POOL2D: adaptiveAvgPool2dBackward(ctx, node); return;
    case OP_CONV2D: conv2dBackward(ctx, node); return;
    case OP_SQRT: shapes_SqrtBackward(ctx, node); return;
    default: PANIC_WITH_CODE(BACKWARD_TENSOR_OP_NOT_FOUND);
  }
}

olib_Array *shapesnn_Backward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *ones = olib_Allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(ones == NULL, ALLOCATION_FAILED);
  *ones = shapes_Make_FloatTensor(ctx, SHAPE1D(1), 1);
  shapes_AddInPlace(ctx, tensor->grad, ones);

  olib_Array *graph = buildGraph(ctx, tensor);

  for (size_t i = graph->size; i-- > 0;) {
    Tensor node = shapes_Array_TensorIdx(graph, i);
    backward(ctx, &node);
  }

  return graph;
}

olib_Array *buildGraph(Context *ctx, Tensor *tensor) {
  olib_Array *graph = shapes_Make_DynamicTensorArray(ctx->memory);
  PtrSet *visited = Make_PtrSet(ctx->memory);
  topoSort(graph, visited, tensor);
  return graph;
}

void topoSort(olib_Array *graph, PtrSet *visited, Tensor *tensor) {
  PANIC_IF(graph == NULL, ERR_NULL_PTR);
  PANIC_IF(visited == NULL, ERR_NULL_PTR);
  PANIC_IF(tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  void *key = (void *)(uintptr_t)tensor->nodeId;
  if (PtrSet_Contains(visited, key)) {
    return;
  }

  PtrSet_Put(visited, key);

  if (tensor->inputs == NULL) {
    shapes_Array_AppendTensor(graph, tensor);
    return;
  }

  for (size_t i = 0; i < tensor->inputs->size; i++) {
    Tensor t = shapes_Array_TensorIdx(tensor->inputs, i);
    topoSort(graph, visited, &t);
  }

  shapes_Array_AppendTensor(graph, tensor);
}
