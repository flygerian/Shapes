#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor/tensor_internal.h"
#include "utils_lib/array.h"

void tanhBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input == NULL || input->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *ones = T_Float(ctx, SHAPE1D(1), 1.0f);
  Tensor *tanhSquared = Multiply(ctx, tensor, tensor);
  Tensor *oneMinusTanhSquared = Subtract(ctx, ones, tanhSquared);
  Tensor *gradInput = Multiply(ctx, tensor->grad, oneMinusTanhSquared);

  Tensor *reducedGrad = ReduceBroadcast(ctx, input, gradInput);
  AddInPlace(ctx, input->grad, reducedGrad);
}

Tensor *tanhForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *out = Tanh(ctx, tensor);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PANIC_IF(out->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  Array_AppendTensor(out->inputs, inputRef);

  out->opType = OP_TANH;
  out->backward = tanhBackward;
  return out;
}

Array *tanhLayerParameters(Context *ctx, Layer *state) {
  return MakeArray(ctx->memory, sizeof(Tensor *), 0);
}

FowardPassOp layer_Tanh(Context *ctx) {
  Layer *layer = allocateOnCtx(ctx, sizeof(Layer));
  layer->weights = NULL;
  layer->bias = NULL;
  layer->layerData = NULL;

  return (FowardPassOp){.ctx = ctx, .type = OP_TANH, .op = layer};
}
