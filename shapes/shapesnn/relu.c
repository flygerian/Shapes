#include "shapes.h"
#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include <stdio.h>

void reluBackward(shapes_Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor input = shapes_ArrayTensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor dInput = shapes_ReluBackward(ctx, tensor, tensor->grad);
  Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &dInput);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

Tensor reluForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor) {
  (void)layer;

  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor out = shapes_Relu(ctx, tensor);
  out.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(Tensor));

  Tensor *inputRef = tensor;
  shapes_ArrayAppendTensor(out.inputs, inputRef);

  out.opType = OP_RELU;
  return out;
}

olib_Array *reluLayerParameters(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(Tensor), 0);
}

olib_Array *reluLayerTensors(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(Tensor), 0);
}

void reluLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

shapesnn_FowardPassOp shapesnn_Relu(shapes_Context *ctx, shapes_Dtype dtype) {
  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  *layer = (shapesnn_layer) {.weights = {}, .bias = {}, .layerData = NULL};

  shapesnn_FowardPassOp op = (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_RELU, .dtype = dtype, .op = layer};
  return op;
}
