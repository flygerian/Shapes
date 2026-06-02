#include "shapes.h"
#include "nn.h"
#include "result.h"
#include "tensor_internal.h"
#include "types.h"
#include "array.h"
#include "memory.h"

typedef struct maxPool2dLayerData {
  Dim kernel;
  u8 stride;
} maxPool2dLayerData;

void maxPool2dBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor input = shapes_Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  maxPool2dLayerData *layerData = tensor->opMetadata;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  Tensor dX;
  Result result = shapes_layer_MaxPool2dBackward(ctx, &input, tensor->grad, layerData->kernel, layerData->stride, &dX);
  PANIC_IF(result != OK, result);

  Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &dX);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

Tensor maxPool2dForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  maxPool2dLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  dim_t kH = layerData->kernel.dims[0];
  dim_t kW = layerData->kernel.dims[1];
  dim_t h = tensor->shape.dims[1];
  dim_t w = tensor->shape.dims[2];
  dim_t channels = tensor->shape.dims[3];
  dim_t batch = tensor->shape.dims[0];
  dim_t outH = (h - kH) / layerData->stride + 1;
  dim_t outW = (w - kW) / layerData->stride + 1;

  Tensor dest = t_Zeros(ctx, SHAPE4D(batch, outH, outW, channels), tensor->dtype);

  Result result = shapes_layer_MaxPool2d(ctx, tensor, layerData->kernel, layerData->stride, &dest);
  PANIC_IF(result != OK, result);

  dest.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));

  Tensor *inputRef = tensor;
  shapes_Array_AppendTensor(dest.inputs, inputRef);

  dest.opMetadata = layerData;
  dest.opType = OP_MAXPOOL2D;
  return dest;
}

Array *maxPool2dLayerParameters(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

Array *maxPool2dLayerTensors(Context *ctx, Layer *state) {
  (void)state;
  return MakeArray(ctx->memory, sizeof(Tensor), 0);
}

void maxPool2dLayerLoad(Context *ctx, Layer *state, Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

FowardPassOp shapesnn_MaxPool2d(Context *ctx, Dtype dtype, dim_t kernelH, dim_t kW, u8 stride) {
  maxPool2dLayerData *layerData = allocate(ctx->memory, sizeof(maxPool2dLayerData));
  dim_t *kernelDims = allocate(ctx->memory, sizeof(dim_t) * 2);
  kernelDims[0] = kernelH;
  kernelDims[1] = kW;
  *layerData = (maxPool2dLayerData){
      .kernel = {.dims = kernelDims, .numOfDims = 2, .multipliers = NULL},
      .stride = stride,
  };

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  layer->weights = (Tensor){0};
  layer->bias = (Tensor){0};
  layer->layerData = layerData;

  return (FowardPassOp){.ctx = ctx, .type = OP_MAXPOOL2D, .dtype = dtype, .op = layer};
}
