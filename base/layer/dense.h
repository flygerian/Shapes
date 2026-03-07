#ifndef shapes_layer_dense_h
#define shapes_layer_dense_h

#include "common.h"
#include "tensor/tensor.h"

Result DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor *dest);

#endif
