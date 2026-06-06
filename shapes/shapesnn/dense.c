#include "shapes.h"
#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include <sched.h>
#include <stdlib.h>
#include <time.h>

typedef struct denseLayerData {
  bool withBias;
} denseLayerData;

void denseBackward(shapes_Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->opType != OP_DENSE, NOT_A_DENSE_LAYER);

  denseLayerData *layerData = tensor->opMetadata;

  Tensor input = shapes_ArrayTensorIdx(tensor->inputs, 0);
  Tensor weights = shapes_ArrayTensorIdx(tensor->inputs, 1);
  Tensor bias = {};

  if (layerData->withBias) {
    PANIC_IF(tensor != NULL, ERR_DIM_MISMATCH);
    bias = shapes_ArrayTensorIdx(tensor->inputs, 2);
    PANIC_IF(bias.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  }

  PANIC_IF(input.grad == NULL || weights.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Result result = shapes_DenseBackward(ctx, &input, &weights, tensor->grad, input.grad, weights.grad, layerData->withBias ? bias.grad : NULL);
  PANIC_IF(result != OK, result);
}

Tensor denseForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL || layer->weights.values == NULL, ERR_NULL_TENSOR_PROVIDED);

  denseLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);
  Tensor out = shapes_DenseLinear(ctx, tensor, &layer->weights, &layer->bias, layerData->withBias);
  out.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(Tensor));

  shapes_ArrayAppendTensor(out.inputs, tensor);
  shapes_ArrayAppendTensor(out.inputs, &layer->weights);

  if (layerData->withBias) {
    shapes_ArrayAppendTensor(out.inputs, &layer->bias);
  }

  out.opMetadata = layerData;
  out.opType = OP_DENSE;
  return out;
}

olib_Array *denseLayerParameters(shapes_Context *ctx, shapesnn_layer *layer) {
  denseLayerData *layerData = layer->layerData;
  olib_Array *params = olib_MakeArray(ctx->memory, sizeof(Tensor), 2);
  shapes_ArrayAppendTensor(params, &layer->weights);

  if (layerData->withBias) {
    shapes_ArrayAppendTensor(params, &layer->bias);
  }

  return params;
}

olib_Array* denseLayerTensors(shapes_Context *ctx, shapesnn_layer *layer) {
  return denseLayerParameters(ctx, layer);
}

void denseLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors) {
  denseLayerData *layerData = layer->layerData;
  size_t expected = layerData->withBias ? 2 : 1;
  PANIC_IF(tensors->size != expected, ERR_DIM_MISMATCH);

  Tensor temp = shapes_ArrayTensorIdx(tensors, 0);
  loadIntoTensor(ctx, &layer->weights, &temp);
  if (layerData->withBias) {
    temp = shapes_ArrayTensorIdx(tensors, 1);
    loadIntoTensor(ctx, &layer->bias, &temp);
  }
}

shapesnn_FowardPassOp shapesnn_Dense(shapes_Context *ctx, shapes_Dtype dtype, size_t inputSize, size_t outputSize, bool withBias) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inputSize, 0.5f);

  denseLayerData *layerData = olib_Allocate(ctx->memory, sizeof(denseLayerData));
  *layerData = (denseLayerData){.withBias = withBias};

  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_MakeRandomTensor(ctx, SHAPE2D(outputSize, inputSize), -initVal, initVal, dtype);
  layer->weights.label = "weights";
  layer->layerData = layerData;

  if (withBias) {
    layer->bias = shapes_MakeRandomTensor(ctx, SHAPE1D(outputSize), -0.1, 0.1, dtype);
    layer->bias.label = "bias";
  } else {
    layer->bias = (Tensor){0};
  }

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_DENSE, .dtype = dtype, .op = layer};
}
