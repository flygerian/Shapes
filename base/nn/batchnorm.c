#include "common.h"
#include "shapes.h"
#include <stddef.h>

Tensor nn_BatchNorm2d(Context *ctx, Tensor *x, size_t numFeatures) {
  Tensor *gamma = T_Float(ctx, SHAPE1D(numFeatures), 1.0);
  Tensor *beta = T_Zeros(ctx, SHAPE1D(numFeatures));

    
}
