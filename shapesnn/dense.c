#include "shapes.h"
#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include "memory.h"
#include <sched.h>
#include <stdlib.h>
#include <time.h>

typedef struct denseLayerData {
  bool withBias;
} denseLayerData;

void denseBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->opType != OP_DENSE, NOT_A_DENSE_LAYER);

  denseLayerData *layerData = tensor->opMetadata;

  Tensor input = shapes_Array_TensorIdx(tensor->inputs, 0);
  Tensor weights = shapes_Array_TensorIdx(tensor->inputs, 1);
  Tensor bias = {};

  if (layerData->withBias) {
    PANIC_IF(tensor != NULL, ERR_DIM_MISMATCH);
    bias = shapes_Array_TensorIdx(tensor->inputs, 2);
    PANIC_IF(bias.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  }

  PANIC_IF(input.grad == NULL || weights.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Result result = shapes_layer_DenseBackward(ctx, &input, &weights, tensor->grad, input.grad, weights.grad, layerData->withBias ? bias.grad : NULL);
  PANIC_IF(result != OK, result);
}

Tensor denseForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL || layer->weights.values == NULL, ERR_NULL_TENSOR_PROVIDED);

  denseLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);
  Tensor out = shapes_layer_DenseLinear(ctx, tensor, &layer->weights, &layer->bias, layerData->withBias);
  out.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));

  shapes_Array_AppendTensor(out.inputs, tensor);
  shapes_Array_AppendTensor(out.inputs, &layer->weights);

  if (layerData->withBias) {
    shapes_Array_AppendTensor(out.inputs, &layer->bias);
  }

  out.opMetadata = layerData;
  out.opType = OP_DENSE;
  return out;
}

Array *denseLayerParameters(Context *ctx, Layer *layer) {
  denseLayerData *layerData = layer->layerData;
  Array *params = MakeArray(ctx->memory, sizeof(Tensor), 2);
  shapes_Array_AppendTensor(params, &layer->weights);

  if (layerData->withBias) {
    shapes_Array_AppendTensor(params, &layer->bias);
  }

  return params;
}

Array* denseLayerTensors(Context *ctx, Layer *layer) {
  return denseLayerParameters(ctx, layer);
}

void denseLayerLoad(Context *ctx, Layer *layer, Array *tensors) {
  denseLayerData *layerData = layer->layerData;
  size_t expected = layerData->withBias ? 2 : 1;
  PANIC_IF(tensors->size != expected, ERR_DIM_MISMATCH);

  Tensor temp = shapes_Array_TensorIdx(tensors, 0);
  loadIntoTensor(ctx, &layer->weights, &temp);
  if (layerData->withBias) {
    temp = shapes_Array_TensorIdx(tensors, 1);
    loadIntoTensor(ctx, &layer->bias, &temp);
  }
}

FowardPassOp shapesnn_Dense(Context *ctx, Dtype dtype, size_t inputSize, size_t outputSize, bool withBias) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inputSize, 0.5f);

  denseLayerData *layerData = allocate(ctx->memory, sizeof(denseLayerData));
  *layerData = (denseLayerData){.withBias = withBias};

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_Make_RandomTensor(ctx, SHAPE2D(outputSize, inputSize), -initVal, initVal, dtype);
  layer->weights.label = "weights";
  layer->layerData = layerData;

  if (withBias) {
    layer->bias = shapes_Make_RandomTensor(ctx, SHAPE1D(outputSize), -0.1, 0.1, dtype);
    layer->bias.label = "bias";
  } else {
    layer->bias = (Tensor){0};
  }

  return (FowardPassOp){.ctx = ctx, .type = OP_DENSE, .dtype = dtype, .op = layer};
}
