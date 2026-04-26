#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor/tensor_internal.h"
#include "utils_lib/array.h"

void reluBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input == NULL || input->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *dInput = ReluBackward(ctx, tensor, tensor->grad);

  Tensor *reducedGrad = ReduceBroadcast(ctx, input, dInput);
  AddInPlace(ctx, input->grad, reducedGrad);
}

Tensor *reluForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *out = Relu(ctx, tensor);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PANIC_IF(out->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  Array_AppendTensor(out->inputs, inputRef);

  out->opType = OP_RELU;
  return out;
}

Array *reluLayerParameters(Context *ctx, Layer *state) {
  return MakeArray(ctx->memory, sizeof(Tensor *), 0);
}

FowardPassOp *layer_Relu(Context *ctx, Dtype dtype) {
  Layer *layer = allocateOnCtx(ctx, sizeof(Layer));
  layer->weights = NULL;
  layer->bias = NULL;
  layer->layerData = NULL;

  FowardPassOp *op = allocateOnCtx(ctx, sizeof(FowardPassOp));
  *op = (FowardPassOp){.ctx = ctx, .type = OP_RELU, .dtype = dtype, .op = layer};
  return op;
}
