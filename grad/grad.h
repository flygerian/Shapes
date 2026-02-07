#ifndef shapes_graph_h
#define shapes_graph_h

#include "common.h"

Result ConstructBinopBackwardpass(Context *ctx, OpType type, Tensor *a, Tensor *b, Tensor *result);

#endif
