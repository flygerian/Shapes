#include "common.h"
#include "nn.h"
#include "nn_internal.h"
#include "result/result.h"
#include "shapes.h"
#include "types.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdlib.h>

void embeddingBackward(Context *ctx, Tensor *out) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor input = shapes_Array_TensorIdx(out->inputs, 0);
  Tensor *indices = (Tensor *)out->opMetadata;

  shapes_IndexAccumulate1d(ctx, input.grad, indices, out->grad);
}

Tensor embeddingForward(Context *ctx, Layer *layer, Tensor *indices) {
  PANIC_IF(layer == NULL, ERR_NULL_PTR);
  PANIC_IF(layer->weights.values == NULL, ERR_NULL_PTR);
  PANIC_IF(indices == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  PANIC_IF(layer->weights.shape.numOfDims != 2, ERR_DIM_MISMATCH);

  Tensor out = shapes_IndexWithTensor(ctx, &layer->weights, indices);
  out.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));
  shapes_Array_AppendTensor(out.inputs, &layer->weights);

  out.opMetadata = indices;
  out.opType = OP_EMBEDDING;
  Tensor *gradPtr = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
  *gradPtr = shapes_Make_ZerosTensor(ctx, out.shape);
  out.grad = gradPtr;
  return out;
}

Array *embeddingParameters(Context *ctx, Layer *layer) {
  Array *params = MakeArray(ctx->memory, sizeof(Tensor), 1);
  shapes_Array_AppendTensor(params, &layer->weights);

  return params;
}

Array *embeddingLayerTensors(Context *ctx, Layer *layer) {
  return embeddingParameters(ctx, layer);
}

void embeddingLayerLoad(Context *ctx, Layer *layer, Array *tensors) {
  PANIC_IF(tensors->size != 1, ERR_DIM_MISMATCH);
  Tensor temp = shapes_Array_TensorIdx(tensors, 0);
  loadIntoTensor(ctx, &layer->weights, &temp);
}

FowardPassOp shapesnn_Embedding(Context *ctx, Dtype dtype, size_t vocabSize, dim_t embeddingDim) {
  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_Make_RandomTensor(ctx, SHAPE2D(vocabSize, embeddingDim), -0.1f, 0.1f, dtype);
  layer->weights.label = "weights";
  layer->bias = (Tensor){0};
  layer->layerData = NULL;

  return (FowardPassOp){.ctx = ctx, .type = OP_EMBEDDING, .dtype = dtype, .op = layer};
}
