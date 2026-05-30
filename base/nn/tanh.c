#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "types.h"
#include "utils_lib/array.h"

void tanhBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor input = shapes_Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor ones = shapes_Make_FloatTensor(ctx, SHAPE1D(1), 1.0f);
  Tensor tanhSquared = shapes_Multiply(ctx, tensor, tensor);
  Tensor oneMinusTanhSquared = shapes_Subtract(ctx, &ones, &tanhSquared);
  Tensor gradInput = shapes_Multiply(ctx, tensor->grad, &oneMinusTanhSquared);

  Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &gradInput);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

Tensor tanhForward(Context *ctx, Layer *layer, Tensor *tensor) {
  (void)layer;
  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor out = shapes_Tanh(ctx, tensor);
  out.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));
  PANIC_IF(out.inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  shapes_Array_AppendTensor(out.inputs, inputRef);
  out.opType = OP_TANH;

  return out;
}

Array *tanhLayerParameters(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

Array *tanhLayerTensors(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

void tanhLayerLoad(Context *ctx, Layer *state, Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

FowardPassOp shapesnn_Tanh(Context *ctx, Dtype dtype) {
  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  layer->weights = (Tensor){0};
  layer->bias = (Tensor){0};
  layer->layerData = NULL;

  return (FowardPassOp){.ctx = ctx, .type = OP_TANH, .dtype = dtype, .op = layer};
}
