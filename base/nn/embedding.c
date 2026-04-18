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
  Tensor *indices = (Tensor*) out->opMetadata;

  IndexAccumulate1d(ctx, input, indices, out->grad);
}

Tensor* embeddingForward(Context *ctx, LayerState *layerState, Tensor *indices) {
  PANIC_IF(layerState == NULL, ERR_NULL_PTR);
  PANIC_IF(layerState->additionalData == NULL, ERR_NULL_PTR);
  PANIC_IF(indices == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(indices->shape.numOfDims != 1, ERR_DIM_MISMATCH);

  Tensor* embedding = (Tensor*) layerState->additionalData;

  PANIC_IF(embedding->shape.numOfDims != 2, ERR_DIM_MISMATCH);

  Tensor* out = IndexWithTensor(ctx, embedding, indices);
  out->inputs = MakeDynamicArray(ctx->memory, 1);
  Array_AppendTensor(out->inputs, out);

  out->opMetadata = indices;
  out->backward = embeddingBackward;
  return out;
}

Layer nn_Embedding(Context *ctx, size_t vocabSize, dim_t embeddingDim) {
  Tensor *embedding = MakeRandomTensor(ctx, SHAPE2D(vocabSize, embeddingDim), -1.0, 1.0, F32);

  return (Layer) {.state = {.additionalData = embedding, }};
}
