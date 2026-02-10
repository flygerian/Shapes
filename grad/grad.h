#ifndef shapes_graph_h
#define shapes_graph_h

#include "common.h"
#include "result/result.h"

Result ConstructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result);
Result ConstructTanhBackwardpass(Context *ctx, Tensor *t, Tensor *result);

Result Backward(Context *ctx, Tensor *t);

#endif
