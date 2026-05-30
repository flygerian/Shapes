#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "nn_internal.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "types.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"
#include <stdio.h>
#include <math.h>

typedef struct conv2dLayerData {
  size_t inChannels;
  size_t outChannels;
  u8 stride;
  bool withBias;
  Tensor *colBuffer;
} conv2dLayerData;

void conv2dBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  conv2dLayerData *layerData = tensor->opMetadata;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  Tensor input = shapes_Array_TensorIdx(tensor->inputs, 0);
  Tensor kernels = shapes_Array_TensorIdx(tensor->inputs, 1);
  Tensor bias = {};

  if (layerData->withBias) {
    bias = shapes_Array_TensorIdx(tensor->inputs, 2);
    PANIC_IF(bias.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  }

  PANIC_IF(input.grad == NULL || kernels.grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(layerData->colBuffer == NULL, ERR_NULL_PTR);

  Tensor dInput = t_Zeros(ctx, input.shape, input.dtype);
  Tensor dKernels = t_Zeros(ctx, kernels.shape, kernels.dtype);

  Tensor dBias = {};
  if (layerData->withBias) {
    dBias = t_Zeros(ctx, bias.shape, bias.dtype);
  }

  Result result = shapes_layer_Conv2dBackward(ctx, &input, &dInput, &kernels, &dKernels, tensor->grad, layerData->colBuffer, &dBias, layerData->withBias, layerData->stride);
  PANIC_IF(result != OK, result);

  Tensor reducedInputGrad = shapes_ReduceBroadcast(ctx, &input, &dInput);
  shapes_AddInPlace(ctx, input.grad, &reducedInputGrad);

  Tensor reducedKernelGrad = shapes_ReduceBroadcast(ctx, &kernels, &dKernels);
  shapes_AddInPlace(ctx, kernels.grad, &reducedKernelGrad);

  if (layerData->withBias) {
    Tensor reducedBiasGrad = shapes_ReduceBroadcast(ctx, &bias, &dBias);
    shapes_AddInPlace(ctx, bias.grad, &reducedBiasGrad);
  }
}

Tensor conv2dForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL || layer->weights.values == NULL, ERR_NULL_TENSOR_PROVIDED);

  conv2dLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  Tensor *kernels = &layer->weights;
  Tensor *bias = layerData->withBias ? &layer->bias : NULL;

  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];
  dim_t batch = tensor->shape.dims[0];
  dim_t h = tensor->shape.dims[1];
  dim_t w = tensor->shape.dims[2];
  dim_t outH = (h - kH) / layerData->stride + 1;
  dim_t outW = (w - kW) / layerData->stride + 1;

  Tensor dest = t_Zeros(ctx, SHAPE4D(batch, outH, outW, layerData->outChannels), tensor->dtype);

  Tensor colBuffer;
  Result result = shapes_layer_Conv2d(ctx, layerData->inChannels, layerData->outChannels, layerData->stride, kernels, bias, layerData->withBias, tensor, &dest, &colBuffer);
  PANIC_IF(result != OK, result);

  Tensor *colBufferPtr = allocate(ctx->memory, sizeof(Tensor));
  *colBufferPtr = colBuffer;
  layerData->colBuffer = colBufferPtr;

  dest.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));
  PANIC_IF(dest.inputs == NULL, ALLOCATION_FAILED);

  shapes_Array_AppendTensor(dest.inputs, tensor);
  shapes_Array_AppendTensor(dest.inputs, kernels);

  if (layerData->withBias) {
    shapes_Array_AppendTensor(dest.inputs, bias);
  }

  dest.opMetadata = layerData;
  dest.opType = OP_CONV2D;
  return dest;
}

Array *conv2dLayerParameters(Context *ctx, Layer *state) {
  conv2dLayerData *layerData = state->layerData;
  Array *params = MakeArray(ctx->memory, sizeof(Tensor), 2);
  shapes_Array_AppendTensor(params, &state->weights);

  if (layerData->withBias) {
    shapes_Array_AppendTensor(params, &state->bias);
  }

  return params;
}

Array *conv2dLayerTensors(Context *ctx, Layer *state) {
  return conv2dLayerParameters(ctx, state);
}

void conv2dLayerLoad(Context *ctx, Layer *state, Array *tensors) {
  conv2dLayerData *layerData = state->layerData;
  size_t expected = layerData->withBias ? 2 : 1;
  PANIC_IF(tensors->size != expected, ERR_DIM_MISMATCH);

  Tensor temp = shapes_Array_TensorIdx(tensors, 0);
  loadIntoTensor(ctx, &state->weights, &temp);
  if (layerData->withBias) {
    temp = shapes_Array_TensorIdx(tensors, 1);
    loadIntoTensor(ctx, &state->bias, &temp);
  }
}

FowardPassOp shapesnn_Conv2d(Context *ctx, Dtype dtype, size_t inChannels, size_t outChannels, dim_t kH, dim_t kW, u8 stride, bool withBias) {
  f32 initVal = (5.0f / 3.0f) / powf((f32)inChannels, 0.5f);

  conv2dLayerData *layerData = allocate(ctx->memory, sizeof(conv2dLayerData));
  *layerData = (conv2dLayerData){.inChannels = inChannels, .outChannels = outChannels, .stride = stride, .withBias = withBias, .colBuffer = NULL};

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_Make_RandomTensor(ctx, SHAPE4D(outChannels, inChannels, kH, kW), -initVal, initVal, dtype);
  layer->weights.label = "weights";
  layer->layerData = layerData;

  if (withBias) {
    layer->bias = t_Zeros(ctx, SHAPE4D(1, 1, 1, outChannels), dtype);
    layer->bias.label = "bias";
  } else {
    layer->bias = (Tensor){0};
  }

  FowardPassOp *op = allocate(ctx->memory, sizeof(FowardPassOp));
  return (FowardPassOp){.ctx = ctx, .type = OP_CONV2D, .dtype = dtype, .op = layer};
}
