#include "../common.h"
#include "result/result.h"
#include "shapes.h"
#include "utils_lib/array.h"
#include "utils_lib/set.h"
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

Array *buildGraph(Context *ctx, Tensor *tensor);
void topoSort(Array *graph, PtrSet *visited, Tensor *tensor);

Array* Backward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *ones = T_Float(ctx, SHAPE1D(1), 1);
  Result result = AddInPlace(ctx, tensor->grad, ones);
  PANIC_IF(result != OK, result);

  Array *graph = buildGraph(ctx, tensor);

  for (size_t i = graph->size; i-- > 0;) {
    Tensor *node = *(Tensor **)Array_Idx(graph, i);
    if (node->backward != NULL) {
      node->backward(ctx, node);
    }
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
    Array_Append(graph, &tensor);
    return;
  }

  for (size_t i = 0; i < tensor->inputs->size; i++) {
    Tensor *t = *(Tensor **)Array_Idx(tensor->inputs, i);
    topoSort(graph, visited, t);
  }

  Array_Append(graph, &tensor);
}
