#include "shapes.h"
#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"

void tanhBackward(shapes_Context *ctx, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor input = shapes_ArrayTensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor ones = shapes_MakeFloatTensor(ctx, SHAPE1D(1), 1.0f);
  shapes_Tensor tanhSquared = shapes_Multiply(ctx, tensor, tensor);
  shapes_Tensor oneMinusTanhSquared = shapes_Subtract(ctx, &ones, &tanhSquared);
  shapes_Tensor gradInput = shapes_Multiply(ctx, tensor->grad, &oneMinusTanhSquared);

  shapes_Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &gradInput);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

shapes_Tensor tanhForward(shapes_Context *ctx, shapesnn_layer *layer, shapes_Tensor *tensor) {
  (void)layer;
  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor out = shapes_Tanh(ctx, tensor);
  out.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(out.inputs == NULL, ALLOCATION_FAILED);

  shapes_Tensor *inputRef = tensor;
  shapes_ArrayAppendTensor(out.inputs, inputRef);
  out.opType = OP_TANH;

  return out;
}

olib_Array *tanhLayerParameters(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

olib_Array *tanhLayerTensors(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

void tanhLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

shapesnn_FowardPassOp shapesnn_Tanh(shapes_Context *ctx, shapes_Dtype dtype) {
  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  layer->weights = (shapes_Tensor){0};
  layer->bias = (shapes_Tensor){0};
  layer->layerData = NULL;

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_TANH, .dtype = dtype, .op = layer};
}
