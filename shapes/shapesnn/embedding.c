#include "memory.h"
#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include <stddef.h>
#include <stdlib.h>

void embeddingBackward(shapes_Context *ctx, Tensor *out) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor input = shapes_ArrayTensorIdx(out->inputs, 0);
  Tensor *indices = (Tensor *)out->opMetadata;

  shapes_IndexAccumulate1d(ctx, input.grad, indices, out->grad);
}

Tensor embeddingForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *indices) {
  PANIC_IF(layer == NULL, ERR_NULL_PTR);
  PANIC_IF(layer->weights.values == NULL, ERR_NULL_PTR);
  PANIC_IF(indices == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  PANIC_IF(layer->weights.shape.numOfDims != 2, ERR_DIM_MISMATCH);

  Tensor out = shapes_IndexWithTensor(ctx, &layer->weights, indices);
  out.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(Tensor));
  shapes_ArrayAppendTensor(out.inputs, &layer->weights);

  out.opMetadata = olib_Allocate(ctx->memory, sizeof(Tensor));
  *((Tensor*) out.opMetadata ) = *indices;
  out.opType = OP_EMBEDDING;
  Tensor *gradPtr = olib_Allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
  *gradPtr = shapes_MakeZerosTensor(ctx, out.shape);
  out.grad = gradPtr;
  return out;
}

olib_Array *embeddingParameters(shapes_Context *ctx, shapesnn_layer *layer) {
  olib_Array *params = olib_MakeArray(ctx->memory, sizeof(Tensor), 1);
  shapes_ArrayAppendTensor(params, &layer->weights);

  return params;
}

olib_Array *embeddingLayerTensors(shapes_Context *ctx, shapesnn_layer *layer) {
  return embeddingParameters(ctx, layer);
}

void embeddingLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors) {
  PANIC_IF(tensors->size != 1, ERR_DIM_MISMATCH);
  Tensor temp = shapes_ArrayTensorIdx(tensors, 0);
  loadIntoTensor(ctx, &layer->weights, &temp);
}

shapesnn_FowardPassOp shapesnn_Embedding(shapes_Context *ctx, shapes_Dtype dtype, size_t vocabSize, shapes_dim_t embeddingDim) {
  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  PANIC_IF(layer == NULL, ALLOCATION_FAILED);
  layer->weights = shapes_MakeRandomTensor(ctx, SHAPE2D(vocabSize, embeddingDim), -0.1f, 0.1f, dtype);
  layer->weights.label = "weights";
  layer->bias = (Tensor){0};
  layer->layerData = NULL;

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_EMBEDDING, .dtype = dtype, .op = layer};
}
