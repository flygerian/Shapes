#include "common.h"
#include "grad/grad.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor/value.h"
#include <stdbool.h>
#include <stddef.h>
#include <time.h>

bool contains(ComputationGraph *computationGraph, GraphNode *search) {
  for (size_t i = 0; i < computationGraph->size; i++) {
    if (computationGraph->nodes[i] == search) {
      return true;
    }
  }

  return false;
}

void add(Context *ctx, ComputationGraph *computationGraph, GraphNode *node) {
  if ((computationGraph->size + 1) == computationGraph->capacity) {
    computationGraph->capacity = GROW_CAPACITY(computationGraph->capacity);
    computationGraph->nodes =
        GROW_ARRAY(ctx->memory, GraphNode *, computationGraph->nodes, computationGraph->capacity);
  }

  computationGraph->nodes[computationGraph->size] = node;
  computationGraph->size += 1;
}

void topo(Context *ctx, ComputationGraph *computationGraph, ComputationGraph *visited,
          GraphNode *currentNode) {
  if (contains(visited, currentNode)) {
    return;
  }

  add(ctx, visited, currentNode);

  for (size_t i = 0; i < currentNode->numInputs; i++) {
    topo(ctx, computationGraph, visited, currentNode->inputs[i]->computation);
  }

  add(ctx, computationGraph, currentNode);
}

ComputationGraph *InitComputationGraph(Context *ctx, Tensor *t) {
  ComputationGraph *graph = allocate(ctx->memory, sizeof(ComputationGraph));
  graph->size = 0;
  graph->capacity = 8;
  graph->nodes = (GraphNode **)allocate(ctx->memory, sizeof(GraphNode *) * graph->capacity);

  ComputationGraph visited = {.capacity = 8,
                              .size = 0,
                              .nodes =
                                  (GraphNode **)allocate(ctx->memory, sizeof(GraphNode *) * 8)};

  topo(ctx, graph, &visited, t->computation);

  freeAlloc(ctx->memory, (void *)visited.nodes);

  return graph;
}

Result Backward(Context *ctx, ComputationGraph *computationGraph) {
  // Set the grad of the origin tensor to one
  GraphNode *first = computationGraph->nodes[computationGraph->size - 1];
  SetValues(first->grad, VALUE(F32, 1.0));

  // Iterate backwards through the graph (reverse topological order)
  for (int i = (int)computationGraph->size - 1; i >= 0; i--) {
    Result result = computationGraph->nodes[i]->backward(ctx, computationGraph->nodes[i]);
    if (result != OK) {
      return result;
    }
  }

  return OK;
}
