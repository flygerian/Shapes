#include "../common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include "utils_lib/set.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include "nn_internal.h"

Array *buildGraph(Context *ctx, Tensor *tensor);
void topoSort(Array *graph, PtrSet *visited, Tensor *tensor);

void backward(Context *ctx, Tensor *node) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(node == NULL, NULL_CONTEXT);

  switch (node->opType) {
    case OP_DENSE:
      denseBackward(ctx, node);
      return;
    case OP_MSE:
      mseBackward(ctx, node);
      return;
    case OP_BATCH_NORM:
      batchnormBackward(ctx, node);
      return;
    case OP_CROSS_ENTHROPY:
      crossEnthropyBackward(ctx, node); 
      return;
  }

  PANIC_IF(node->opType != OP_NONE, BACKWARD_TENSOR_OP_NOT_FOUND);
}

Array *Backward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *ones = T_Float(ctx, SHAPE1D(1), 1);
  AddInPlace(ctx, tensor->grad, ones);

  Array *graph = buildGraph(ctx, tensor);

  for (size_t i = graph->size; i-- > 0;) {
    Tensor *node = Array_TensorIdx(graph, i);
    backward(ctx, node);
  }

  return graph;
}

Array *buildGraph(Context *ctx, Tensor *tensor) {
  Array *graph = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PtrSet *visited = MakePtrSet(ctx->memory);
  topoSort(graph, visited, tensor);
  return graph;
}

void topoSort(Array *graph, PtrSet *visited, Tensor *tensor) {
  PANIC_IF(graph == NULL, ERR_NULL_PTR);
  PANIC_IF(visited == NULL, ERR_NULL_PTR);
  PANIC_IF(tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  if (PtrSet_Contains(visited, tensor)) {
    return;
  }

  PtrSet_Put(visited, tensor);

  if (tensor->inputs == NULL) {
    Array_AppendTensor(graph, tensor);
    return;
  }

  for (size_t i = 0; i < tensor->inputs->size; i++) {
    Tensor *t = Array_TensorIdx(tensor->inputs, i);
    topoSort(graph, visited, t);
  }

  Array_AppendTensor(graph, tensor);
}
