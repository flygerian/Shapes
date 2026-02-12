#ifndef shapes_graph_h
#define shapes_graph_h

#include "common.h"
#include "result/result.h"


typedef struct ComputationGraph {
  GraphNode **nodes; // Dont want to copy node. we just want to know where they are
  size_t capacity;
  size_t size;
} ComputationGraph;

Result ConstructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result);
Result ConstructTanhBackwardpass(Context *ctx, Tensor *t, Tensor *result);
Result ConstructPowBackwardpass(Context *ctx, Tensor *t, f32 power, Tensor *result);

ComputationGraph *InitComputationGraph(Context *ctx, Tensor *t);

Result Backward(Context *ctx, ComputationGraph *graph);

#endif
