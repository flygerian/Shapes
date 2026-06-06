#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include "memory.h"
#include <sched.h>
#include <stddef.h>
#include <stdio.h>

shapes_Tensor flattenFoward(shapes_Context *ctx, shapesnn_layer *layer, shapes_Tensor *tensor) {
  PANIC_IF(layer != NULL, ERR_NO_OP); // there should be no layer
  PANIC_IF_NULL(tensor);

  shapes_Dim shape = tensor->shape;
  PANIC_IF(shape.numOfDims < 2, ERR_NO_OP);

  size_t flattenedSize = 1;
  for (RANGE_FROM(1, shape.numOfDims, i)) {
    shapes_dim_t dim = shape.dims[i];
    flattenedSize *= dim;
  }

  return shapes_Reshape(ctx, tensor, SHAPE2D(shape.dims[0], flattenedSize));
}

olib_Array *flattenParameters(shapes_Context *ctx) {
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

olib_Array *flattenLayerTensors(shapes_Context *ctx) {
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

void flattenLayerLoad(shapes_Context *ctx, olib_Array *tensors) {
  (void)ctx;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

shapesnn_FowardPassOp shapesnn_Flatten(shapes_Context *ctx, shapes_Dtype dtype) {
  PANIC_IF_NULL(ctx);

  shapesnn_FowardPassOp *flattenLayer = olib_Allocate(ctx->memory, sizeof(shapesnn_FowardPassOp));
  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_FLATTEN, .dtype = dtype};
}
