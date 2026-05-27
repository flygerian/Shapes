#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "types.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"
#include <alloca.h>
#include <stddef.h>
#include <stdio.h>

Tensor *flattenFoward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(layer != NULL, ERR_NO_OP);
  PANIC_IF_NULL(tensor);

  Dim shape = tensor->shape;
  if (shape.numOfDims < 2) {
    return tensor;
  }

  size_t flattenedSize = 1;
  for (RANGE_FROM(1, shape.numOfDims, i)) {
    dim_t dim = shape.dims[i];
    flattenedSize *= dim;
  }

  Tensor *out = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(out == NULL, ALLOCATION_FAILED);
  *out = shapes_Reshape(ctx, tensor, SHAPE2D(shape.dims[0], flattenedSize));
  return out;
}

Array *flattenParameters(Context *ctx) {
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

Array *flattenLayerTensors(Context *ctx) {
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

void flattenLayerLoad(Context *ctx, Array *tensors) {
  (void)ctx;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

FowardPassOp *shapesnn_Flatten(Context *ctx, Dtype dtype) {
  PANIC_IF_NULL(ctx);

  FowardPassOp *flattenLayer = allocate(ctx->memory, sizeof(FowardPassOp));
  *flattenLayer = (FowardPassOp){.ctx = ctx, .type = OP_FLATTEN, .dtype = dtype};
  return flattenLayer;
}
