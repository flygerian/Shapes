#include "result.h"
#include "shapes.h"
#include "types.h"
#include "array.h"
#include "memory.h"

void crossEnthropyBackward(shapes_Context *ctx, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor yGround = shapes_ArrayTensorIdx(tensor->inputs, 0);
  shapes_Tensor logits = shapes_ArrayTensorIdx(tensor->inputs, 1);
  shapes_Tensor *probs = tensor->opMetadata;

  shapes_Tensor dLogits = shapes_loss_CrossEntropyBackward(ctx, &yGround, probs, tensor->grad);
  shapes_Tensor reducedLogits = shapes_ReduceBroadcast(ctx, &logits, &dLogits);
  shapes_AddInPlace(ctx, logits.grad, &reducedLogits);
}

shapes_Tensor shapesnn_CrossEnthropy(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *logits) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(yGround == NULL, ERR_NULL_PTR);
  PANIC_IF(logits == NULL, ERR_NULL_PTR);

  shapes_TensorPair crossEnthropyResult = shapes_loss_CrossEntropyForward(ctx, yGround, logits);

  shapes_Tensor loss = crossEnthropyResult.a;
  shapes_Tensor probs = crossEnthropyResult.b;

  loss.inputs = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 2);
  shapes_ArrayAppendTensor(loss.inputs, yGround);
  shapes_ArrayAppendTensor(loss.inputs, logits);

  loss.opType = OP_CROSS_ENTHROPY;
  shapes_Tensor *opMetadata = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  *opMetadata = probs;
  loss.opMetadata = opMetadata;

  return loss;
}
