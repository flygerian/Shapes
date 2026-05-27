#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor_internal.h"
#include "utils_lib/array.h"

void reluBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = shapes_Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input == NULL || input->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *dInput = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(dInput == NULL, ALLOCATION_FAILED);
  *dInput = shapes_ReluBackward(ctx, tensor, tensor->grad);

  Tensor reducedGrad = shapes_ReduceBroadcast(ctx, input, dInput);
  shapes_AddInPlace(ctx, input->grad, &reducedGrad);
}

Tensor *reluForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *out = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(out == NULL, ALLOCATION_FAILED);
  *out = shapes_Relu(ctx, tensor);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));
  PANIC_IF(out->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  shapes_Array_AppendTensor(out->inputs, inputRef);

  out->opType = OP_RELU;
  return out;
}

Array *reluLayerParameters(Context *ctx, Layer *state) {
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

FowardPassOp *shapesnn_Relu(Context *ctx, Dtype dtype) {
  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  layer->weights = (Tensor){0};
  layer->bias = (Tensor){0};
  layer->layerData = NULL;

  FowardPassOp *op = allocate(ctx->memory, sizeof(FowardPassOp));
  *op = (FowardPassOp){.ctx = ctx, .type = OP_RELU, .dtype = dtype, .op = layer};
  return op;
}
