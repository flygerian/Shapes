#include "../common.h"
#include "../shapes.h"
#include "nn.h"
#include "result/result.h"
#include "../tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"

typedef struct adaptiveAvgPool2dLayerData {
  dim_t outH;
  dim_t outW;
} adaptiveAvgPool2dLayerData;

void adaptiveAvgPool2dBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = Array_TensorIdx(tensor->inputs, 0);
  PANIC_IF(input == NULL || input->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  adaptiveAvgPool2dLayerData *layerData = tensor->opMetadata;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  Tensor dX;
  Result result =
      AdaptiveAvgPool2dBackward(ctx, input, tensor->grad, layerData->outH, layerData->outW, &dX);
  PANIC_IF(result != OK, result);

  Tensor *reducedGrad = ReduceBroadcast(ctx, input, &dX);
  AddInPlace(ctx, input->grad, reducedGrad);
}

Tensor *adaptiveAvgPool2dForward(Context *ctx, Layer *layer, Tensor *tensor) {
  PANIC_IF(ctx == NULL || layer == NULL || tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

  adaptiveAvgPool2dLayerData *layerData = layer->layerData;
  PANIC_IF(layerData == NULL, ERR_NULL_PTR);

  dim_t batch = tensor->shape.dims[0];
  dim_t channels = tensor->shape.dims[3];

  Tensor *dest =
      t_Zeros(ctx, SHAPE4D(batch, layerData->outH, layerData->outW, channels), tensor->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  Result result = AdaptiveAvgPool2d(ctx, tensor, layerData->outH, layerData->outW, dest);
  PANIC_IF(result != OK, result);

  dest->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  PANIC_IF(dest->inputs == NULL, ALLOCATION_FAILED);

  Tensor *inputRef = tensor;
  Array_AppendTensor(dest->inputs, inputRef);

  dest->opMetadata = layerData;
  dest->opType = OP_ADAPTIVE_AVG_POOL2D;
  return dest;
}

Array *adaptiveAvgPool2dLayerParameters(Context *ctx, Layer *state) {
  return MakeArray(ctx->memory, sizeof(Tensor *), 0);
}

FowardPassOp *layer_AdaptiveAvgPool2d(Context *ctx, Dtype dtype, dim_t outH, dim_t outW) {
  adaptiveAvgPool2dLayerData *layerData = allocate(ctx->memory, sizeof(adaptiveAvgPool2dLayerData));
  *layerData = (adaptiveAvgPool2dLayerData){.outH = outH, .outW = outW};

  Layer *layer = allocateOnCtx(ctx, sizeof(Layer));
  layer->weights = NULL;
  layer->bias = NULL;
  layer->layerData = layerData;

  FowardPassOp *op = allocateOnCtx(ctx, sizeof(FowardPassOp));
  *op = (FowardPassOp){.ctx = ctx, .type = OP_ADAPTIVE_AVG_POOL2D, .dtype = dtype, .op = layer};
  return op;
}
