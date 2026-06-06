#include "shapes.h"
#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include "memory.h"
#include <stdio.h>
#include <math.h>

typedef struct conv2dLayerData {
  size_t inChannels;
  size_t outChannels;
  u8 stride;
  bool withBias;
  shapes_Tensor *colBuffer;
} conv2dLayerData;

void conv2dBackward(shapes_Context *ctx, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  conv2dLayerData *layerData = tensor->opMetadata;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  shapes_Tensor input = shapes_ArrayTensorIdx(tensor->inputs, 0);
  shapes_Tensor kernels = shapes_ArrayTensorIdx(tensor->inputs, 1);
  shapes_Tensor bias = {};

  if (layerData->withBias) {
    bias = shapes_ArrayTensorIdx(tensor->inputs, 2);
    PANIC_IF(bias.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  }

  PANIC_IF(input.grad == NULL || kernels.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(layerData->colBuffer == NULL, ERR_NULL_PTR);

  shapes_Tensor dInput = shapes_MakeFloatTensor(ctx, input.shape, input.dtype);
  shapes_Tensor dKernels = shapes_MakeFloatTensor(ctx, kernels.shape, kernels.dtype);

  shapes_Tensor dBias = {};
  if (layerData->withBias) {
    dBias = shapes_MakeFloatTensor(ctx, bias.shape, bias.dtype);
  }

  Result result = shapes_Conv2dBackward(ctx, &input, &dInput, &kernels, &dKernels, tensor->grad, layerData->colBuffer, &dBias, layerData->withBias, layerData->stride);
  PANIC_IF(result != OK, result);

  shapes_Tensor reducedInputGrad = shapes_ReduceBroadcast(ctx, &input, &dInput);
  shapes_AddInPlace(ctx, input.grad, &reducedInputGrad);

  shapes_Tensor reducedKernelGrad = shapes_ReduceBroadcast(ctx, &kernels, &dKernels);
  shapes_AddInPlace(ctx, kernels.grad, &reducedKernelGrad);

  if (layerData->withBias) {
    shapes_Tensor reducedBiasGrad = shapes_ReduceBroadcast(ctx, &bias, &dBias);
    shapes_AddInPlace(ctx, bias.grad, &reducedBiasGrad);
  }
}

shapes_Tensor conv2dForward(shapes_Context *ctx, shapesnn_layer *layer, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL || layer->weights.values == NULL, ERR_NULL_TENSOR_PROVIDED);

  conv2dLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  shapes_Tensor *kernels = &layer->weights;
  shapes_Tensor *bias = layerData->withBias ? &layer->bias : NULL;

  shapes_dim_t kH = kernels->shape.dims[2];
  shapes_dim_t kW = kernels->shape.dims[3];
  shapes_dim_t batch = tensor->shape.dims[0];
  shapes_dim_t h = tensor->shape.dims[1];
  shapes_dim_t w = tensor->shape.dims[2];
  shapes_dim_t outH = (h - kH) / layerData->stride + 1;
  shapes_dim_t outW = (w - kW) / layerData->stride + 1;

  shapes_Tensor dest = shapes_MakeFloatTensor(ctx, SHAPE4D(batch, outH, outW, layerData->outChannels), tensor->dtype);

  shapes_Tensor colBuffer;
  Result result = shapes_Conv2d(ctx, layerData->inChannels, layerData->outChannels, layerData->stride, kernels, bias, layerData->withBias, tensor, &dest, &colBuffer);
  PANIC_IF(result != OK, result);

  shapes_Tensor *colBufferPtr = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  *colBufferPtr = colBuffer;
  layerData->colBuffer = colBufferPtr;

  dest.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(dest.inputs == NULL, ALLOCATION_FAILED);

  shapes_ArrayAppendTensor(dest.inputs, tensor);
  shapes_ArrayAppendTensor(dest.inputs, kernels);

  if (layerData->withBias) {
    shapes_ArrayAppendTensor(dest.inputs, bias);
  }

  dest.opMetadata = layerData;
  dest.opType = OP_CONV2D;
  return dest;
}

olib_Array *conv2dLayerParameters(shapes_Context *ctx, shapesnn_layer *state) {
  conv2dLayerData *layerData = state->layerData;
  olib_Array *params = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 2);
  shapes_ArrayAppendTensor(params, &state->weights);

  if (layerData->withBias) {
    shapes_ArrayAppendTensor(params, &state->bias);
  }

  return params;
}

olib_Array *conv2dLayerTensors(shapes_Context *ctx, shapesnn_layer *state) {
  return conv2dLayerParameters(ctx, state);
}

void conv2dLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors) {
  conv2dLayerData *layerData = state->layerData;
  size_t expected = layerData->withBias ? 2 : 1;
  PANIC_IF(tensors->size != expected, ERR_DIM_MISMATCH);

  shapes_Tensor temp = shapes_ArrayTensorIdx(tensors, 0);
  loadIntoTensor(ctx, &state->weights, &temp);
  if (layerData->withBias) {
    temp = shapes_ArrayTensorIdx(tensors, 1);
    loadIntoTensor(ctx, &state->bias, &temp);
  }
}

shapesnn_FowardPassOp shapesnn_Conv2d(shapes_Context *ctx, shapes_Dtype dtype, size_t inChannels, size_t outChannels, shapes_dim_t kH, shapes_dim_t kW, u8 stride, bool withBias) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inChannels, 0.5f);

  conv2dLayerData *layerData = olib_Allocate(ctx->memory, sizeof(conv2dLayerData));
  *layerData = (conv2dLayerData){.inChannels = inChannels, .outChannels = outChannels, .stride = stride, .withBias = withBias, .colBuffer = NULL};

  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_MakeRandomTensor(ctx, SHAPE4D(outChannels, inChannels, kH, kW), -initVal, initVal, dtype);
  layer->weights.label = "weights";
  layer->layerData = layerData;

  if (withBias) {
    layer->bias = shapes_MakeFloatTensor(ctx, SHAPE4D(1, 1, 1, outChannels), dtype);
    layer->bias.label = "bias";
  } else {
    layer->bias = (shapes_Tensor){0};
  }

  shapesnn_FowardPassOp *op = olib_Allocate(ctx->memory, sizeof(shapesnn_FowardPassOp));
  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_CONV2D, .dtype = dtype, .op = layer};
}
