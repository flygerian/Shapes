#include "result/result.h"
#include "shapes.h"
#include "utils_lib/array.h"

void crossEnthropyBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  Tensor *yGround = shapes_Array_TensorIdx(tensor->inputs, 0);
  Tensor *logits = shapes_Array_TensorIdx(tensor->inputs, 1);
  Tensor *probs = tensor->opMetadata;

  Tensor *dLogits = shapes_loss_CrossEntropyBackward(ctx, yGround, probs, tensor->grad);
  Tensor *reducedLogits = shapes_ReduceBroadcast(ctx, logits, dLogits);
  shapes_AddInPlace(ctx, logits->grad, reducedLogits);
}

Tensor shapesnn_CrossEnthropy(Context *ctx, Tensor *yGround, Tensor *logits) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(yGround == NULL, ERR_NULL_PTR);
  PANIC_IF(logits == NULL, ERR_NULL_PTR);

  TensorPair crossEnthropyResult = shapes_loss_CrossEntropyForward(ctx, yGround, logits);

  Tensor *loss = crossEnthropyResult.a;
  Tensor *probs = crossEnthropyResult.b;

  loss->inputs = MakeArray(ctx->memory, sizeof(Tensor *), 2);
  shapes_Array_AppendTensor(loss->inputs, yGround);
  shapes_Array_AppendTensor(loss->inputs, logits);

  loss->opType = OP_CROSS_ENTHROPY;
  loss->opMetadata = probs;

  return *loss;
}
