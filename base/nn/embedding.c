#include "common.h"
#include "nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdlib.h>

void embeddingBackward(Context *ctx, Tensor *out) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *input = Array_TensorIdx(out->inputs, 0);
  Tensor *indices = (Tensor *)out->opMetadata;

  IndexAccumulate1d(ctx, input->grad, indices, out->grad);
}

Tensor *embeddingForward(Context *ctx, Layer *layer, Tensor *indices) {
  PANIC_IF(layer == NULL, ERR_NULL_PTR);
  PANIC_IF(layer->weights == NULL, ERR_NULL_PTR);
  PANIC_IF(indices == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  Tensor *embedding = layer->weights;

  PANIC_IF(embedding->shape.numOfDims != 2, ERR_DIM_MISMATCH);

  Tensor *out = IndexWithTensor(ctx, embedding, indices);
  out->inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor *));
  Array_AppendTensor(out->inputs, embedding);

  out->opMetadata = indices;
  out->opType = OP_EMBEDDING;
  out->grad = T_Zeros(ctx, out->shape);
  return out;
}

Array *embeddingParameters(Context *ctx, Layer *layer) {
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 1);
  Array_AppendTensor(params, (void *)layer->weights);

  return params;
}

FowardPassOp *layer_Embedding(Context *ctx, Dtype dtype, size_t vocabSize, dim_t embeddingDim) {
  Tensor *embedding = MakeRandomTensor(ctx, SHAPE2D(vocabSize, embeddingDim), -0.1f, 0.1f, dtype);

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  layer->weights = embedding;
  layer->bias = NULL;

  FowardPassOp *op = allocate(ctx->memory, sizeof(FowardPassOp));
  *op = (FowardPassOp){.ctx = ctx, .type = OP_EMBEDDING, .dtype = dtype, .op = layer};
  return op;
}
