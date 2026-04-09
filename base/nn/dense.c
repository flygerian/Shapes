#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include <stdlib.h>
#include <time.h>

void denseBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = Array_TensorIdx(tensor->inputs, 0);
  Tensor *weights = *(Tensor **)Array_TensorIdx(tensor->inputs, 1);
  Tensor *bias = NULL;
  if (tensor->inputs->size > 2) {
    bias = Array_TensorIdx(tensor->inputs, 2);
  }

  PANIC_IF(input == NULL || weights == NULL || input->grad == NULL || weights->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(bias != NULL && bias->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Result result = DenseBackward(ctx, input, weights, tensor->grad, input->grad, weights->grad,
                                bias != NULL ? bias->grad : NULL);
  PANIC_IF(result != OK, result);
}

Tensor *denseForward(Context *ctx, LayerState *state, Tensor *tensor) {
  PANIC_IF(ctx == NULL || state == NULL || tensor == NULL || state->weights == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  Tensor *out = DenseLinear(ctx, tensor, state->weights, state->bias, state->bias != NULL);

  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PANIC_IF(out->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  Tensor *weightRef = state->weights;
  Array_AppendTensor(out->inputs, (void*) inputRef);
  Array_AppendTensor(out->inputs, (void*) weightRef);

  if (state->bias != NULL) {
    Tensor *biasRef = state->bias;
    Array_Append(out->inputs, (void*) &biasRef);
  }

  out->opType = OP_DENSE;
  out->backward = denseBackward;
  return out;
}

Array *layerParameters(Context *ctx, LayerState *state) {
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 2);
  Array_AppendTensor(params, (void*) state->weights);
  Array_AppendTensor(params, (void*) state->bias);

  return params;
}

Layer nn_Dense(Context *ctx, size_t inputSize, size_t outputSize) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inputSize, 0.5f);
  Tensor *w = MakeRandomTensor(ctx, SHAPE2D(outputSize, inputSize), -initVal, initVal, F32);
  Tensor *b = MakeRandomTensor(ctx, SHAPE1D(outputSize), -0.1, 0.1, F32);

  LayerState state = (LayerState){.weights = w, .bias = b};

  Layer denseLayer = {.state = state, .forward = denseForward, .parameters = layerParameters};
  return denseLayer;
}
