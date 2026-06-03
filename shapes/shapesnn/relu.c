#include "shapes.h"
#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include <stdio.h>

void reluBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor input = shapes_Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor dInput = shapes_ReluBackward(ctx, tensor, tensor->grad);
  Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &dInput);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

Tensor reluForward(Context *ctx, Layer *layer, Tensor *tensor) {
  (void)layer;

  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor out = shapes_Relu(ctx, tensor);
  out.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));

  Tensor *inputRef = tensor;
  shapes_Array_AppendTensor(out.inputs, inputRef);

  out.opType = OP_RELU;
  return out;
}

Array *reluLayerParameters(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

Array *reluLayerTensors(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

void reluLayerLoad(Context *ctx, Layer *state, Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

FowardPassOp shapesnn_Relu(Context *ctx, Dtype dtype) {
  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  *layer = (Layer) {.weights = {}, .bias = {}, .layerData = NULL};

  FowardPassOp op = (FowardPassOp){.ctx = ctx, .type = OP_RELU, .dtype = dtype, .op = layer};
  return op;
}
