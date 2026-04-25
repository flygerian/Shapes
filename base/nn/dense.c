#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"
#include <sched.h>
#include <stdlib.h>
#include <time.h>

typedef struct denseLayerData {
  bool withBias;
} denseLayerData;

void denseBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->opType != OP_DENSE, NOT_A_DENSE_LAYER);

  denseLayerData *layerData = tensor->opMetadata;

  Tensor *input = Array_TensorIdx(tensor->inputs, 0);
  Tensor *weights = Array_TensorIdx(tensor->inputs, 1);
  Tensor *bias = NULL;

  if (layerData->withBias) {
    PANIC_IF(tensor != NULL, ERR_DIM_MISMATCH);
    bias = Array_TensorIdx(tensor->inputs, 2);
  }

  PANIC_IF(input == NULL || weights == NULL || input->grad == NULL || weights->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(bias != NULL && bias->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Result result = DenseBackward(ctx, input, weights, tensor->grad, input->grad, weights->grad,
                                bias != NULL ? bias->grad : NULL);
  PANIC_IF(result != OK, result);
}

Tensor *denseForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL || layer->weights == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  denseLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  Tensor *out = DenseLinear(ctx, tensor, layer->weights, layer->bias, layerData->withBias);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PANIC_IF(out->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  Tensor *weightRef = layer->weights;
  Array_AppendTensor(out->inputs, inputRef);
  Array_AppendTensor(out->inputs, weightRef);


  if (layerData->withBias) {
    Tensor *biasRef = layer->bias;
    Array_Append(out->inputs, (void *)&biasRef);
  }

  out->opMetadata = layerData;
  out->opType = OP_DENSE;
  return out;
}

Array *denseLayerParameters(Context *ctx, Layer *state) {
  denseLayerData *layerData = state->layerData;
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 2);
  Array_AppendTensor(params, (void *)state->weights);

  if (layerData->withBias) {
    Array_AppendTensor(params, (void *)state->bias);
  }

  return params;
}

FowardPassOp layer_Dense(Context *ctx, size_t inputSize, size_t outputSize, bool withBias) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inputSize, 0.5f);
  Tensor *w = MakeRandomTensor(ctx, SHAPE2D(outputSize, inputSize), -initVal, initVal, F32);

  denseLayerData *layerData = allocate(ctx->memory, sizeof(denseLayerData));
  *layerData = (denseLayerData){.withBias = withBias};

  Layer *layer = allocateOnCtx(ctx, sizeof(Layer));
  *layer = (Layer){.weights = w, .layerData = layerData};

  if (withBias) {
    Tensor *b = MakeRandomTensor(ctx, SHAPE1D(outputSize), -0.1, 0.1, F32);
    layer->bias = b;
  }

  FowardPassOp denseLayer = {.ctx = ctx, .type = OP_DENSE, .op = layer};
  return denseLayer;
}
