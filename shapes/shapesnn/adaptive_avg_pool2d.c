#include "shapes.h"
#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"
#include "memory.h"

typedef struct adaptiveAvgPool2dLayerData {
  shapes_dim_t outH;
  shapes_dim_t outW;
} adaptiveAvgPool2dLayerData;

void adaptiveAvgPool2dBackward(shapes_Context *ctx, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor input = shapes_ArrayTensorIdx(tensor->inputs, 0);
  PANIC_IF(input.grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  adaptiveAvgPool2dLayerData *layerData = tensor->opMetadata;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  shapes_Tensor dX;
  Result result = shapes_AdaptiveAvgPool2dBackward(ctx, &input, tensor->grad, layerData->outH, layerData->outW, &dX);
  PANIC_IF(result != OK, result);

  shapes_Tensor reducedGrad = shapes_ReduceBroadcast(ctx, &input, &dX);
  shapes_AddInPlace(ctx, input.grad, &reducedGrad);
}

shapes_Tensor adaptiveAvgPool2dForward(shapes_Context *ctx, shapesnn_layer *layer, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  adaptiveAvgPool2dLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  shapes_dim_t batch = tensor->shape.dims[0];
  shapes_dim_t channels = tensor->shape.dims[3];

  shapes_Tensor dest = shapes_MakeFloatTensor(ctx, SHAPE4D(batch, layerData->outH, layerData->outW, channels), tensor->dtype);
  Result result = shapes_AdaptiveAvgPool2d(ctx, tensor, layerData->outH, layerData->outW, &dest);
  PANIC_IF(result != OK, result);

  dest.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(shapes_Tensor));

  shapes_Tensor *inputRef = tensor;
  shapes_ArrayAppendTensor(dest.inputs, inputRef);

  dest.opMetadata = layerData;
  dest.opType = OP_ADAPTIVE_AVG_POOL2D;
  return dest;
}

olib_Array *adaptiveAvgPool2dLayerParameters(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

olib_Array *adaptiveAvgPool2dLayerTensors(shapes_Context *ctx, shapesnn_layer *state) {
  (void)state;
  return olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 0);
}

void adaptiveAvgPool2dLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors) {
  (void)ctx;
  (void)state;
  PANIC_IF(tensors->size != 0, ERR_DIM_MISMATCH);
}

shapesnn_FowardPassOp shapesnn_AdaptiveAvgPool2d(shapes_Context *ctx, shapes_Dtype dtype, shapes_dim_t outH, shapes_dim_t outW) {
  adaptiveAvgPool2dLayerData *layerData = olib_Allocate(ctx->memory, sizeof(adaptiveAvgPool2dLayerData));
  *layerData = (adaptiveAvgPool2dLayerData){.outH = outH, .outW = outW};

  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  layer->weights = (shapes_Tensor){0};
  layer->bias = (shapes_Tensor){0};
  layer->layerData = layerData;

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_ADAPTIVE_AVG_POOL2D, .dtype = dtype, .op = layer};
}
