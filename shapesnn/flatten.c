#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include "memory.h"
#include <sched.h>
#include <stddef.h>
#include <stdio.h>

Tensor flattenFoward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(layer != NULL, ERR_NO_OP); // there should be no layer
  PANIC_IF_NULL(tensor);

  Dim shape = tensor->shape;
  PANIC_IF(shape.numOfDims < 2, ERR_NO_OP);

  size_t flattenedSize = 1;
  for (RANGE_FROM(1, shape.numOfDims, i)) {
    dim_t dim = shape.dims[i];
    flattenedSize *= dim;
  }

  return shapes_Reshape(ctx, tensor, SHAPE2D(shape.dims[0], flattenedSize));
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

FowardPassOp shapesnn_Flatten(Context *ctx, Dtype dtype) {
  PANIC_IF_NULL(ctx);

  FowardPassOp *flattenLayer = allocate(ctx->memory, sizeof(FowardPassOp));
  return (FowardPassOp){.ctx = ctx, .type = OP_FLATTEN, .dtype = dtype};
}
