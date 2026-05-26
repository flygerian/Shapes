#include "common.h"
#include "nn.h"
#include "nn_internal.h"
#include "result/result.h"
#include "shapes.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdlib.h>

void embeddingBackward(Context *ctx, Tensor *out) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *input = shapes_Array_TensorIdx(out->inputs, 0);
  Tensor *indices = (Tensor *)out->opMetadata;

  shapes_IndexAccumulate1d(ctx, input->grad, indices, out->grad);
}

Tensor *embeddingForward(Context *ctx, Layer *layer, Tensor *indices) {
  PANIC_IF(layer == NULL, ERR_NULL_PTR);
  PANIC_IF(layer->weights == NULL, ERR_NULL_PTR);
  PANIC_IF(indices == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *embedding = layer->weights;

  PANIC_IF(embedding->shape.numOfDims != 2, ERR_DIM_MISMATCH);

  Tensor *out = shapes_IndexWithTensor(ctx, embedding, indices);
  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  shapes_Array_AppendTensor(out->inputs, embedding);

  out->opMetadata = indices;
  out->opType = OP_EMBEDDING;
  out->grad = shapes_Make_ZerosTensor(ctx, out->shape);
  return out;
}

Array *embeddingParameters(Context *ctx, Layer *layer) {
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 1);
  shapes_Array_AppendTensor(params, (void *)layer->weights);

  return params;
}

Array *embeddingLayerTensors(Context *ctx, Layer *layer) {
  return embeddingParameters(ctx, layer);
}

void embeddingLayerLoad(Context *ctx, Layer *layer, Array *tensors) {
  PANIC_IF(tensors->size != 1, ERR_DIM_MISMATCH);
  loadIntoTensor(ctx, layer->weights, shapes_Array_TensorIdx(tensors, 0));
}

FowardPassOp *shapesnn_Embedding(Context *ctx, Dtype dtype, size_t vocabSize, dim_t embeddingDim) {
  Tensor *embedding = shapes_Make_RandomTensor(ctx, SHAPE2D(vocabSize, embeddingDim), -0.1f, 0.1f, dtype);
  embedding->label = "weights";

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  layer->weights = embedding;
  layer->bias = NULL;

  FowardPassOp *op = allocate(ctx->memory, sizeof(FowardPassOp));
  *op = (FowardPassOp){.ctx = ctx, .type = OP_EMBEDDING, .dtype = dtype, .op = layer};
  return op;
}
