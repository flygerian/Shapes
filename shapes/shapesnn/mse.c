#include "memory.h"
#include "result.h"
#include "shapes.h"
#include <stddef.h>
#include "array.h"

void mseBackward(shapes_Context *ctx, shapes_Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor yGround = shapes_ArrayTensorIdx(tensor->inputs, 0);
  shapes_Tensor yPred = shapes_ArrayTensorIdx(tensor->inputs, 1);

  // ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
  shapes_Tensor diff = shapes_Subtract(ctx, &yPred, &yGround);
  shapes_Tensor two = shapes_MakeFloatTensor(ctx, diff.shape, 2.0);
  shapes_Tensor localGrad = shapes_Multiply(ctx, &two, &diff);
  shapes_Tensor gradYPred = shapes_Multiply(ctx, tensor->grad, &localGrad);
  shapes_Tensor reducedGradYPred = shapes_ReduceBroadcast(ctx, &yPred, &gradYPred);
  shapes_AddInPlace(ctx, yPred.grad, &reducedGradYPred);

  // ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
  shapes_Tensor negLocalGrad = shapes_Negate(ctx, &localGrad);
  shapes_Tensor gradYGround = shapes_Multiply(ctx, tensor->grad, &negLocalGrad);
  shapes_Tensor reducedGradYGround = shapes_ReduceBroadcast(ctx, &yGround, &gradYGround);
  shapes_AddInPlace(ctx, yGround.grad, &reducedGradYGround);
}

shapes_Tensor shapesnn_Mse(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *yPred) {
  shapes_Tensor diffVal = shapes_Subtract(ctx, yPred, yGround);

  shapes_Tensor *loss = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(loss == NULL, ALLOCATION_FAILED);
  *loss = shapes_Pow(ctx, &diffVal, 2);

  for (shapes_dim_t i = 0; i < loss->shape.numOfDims; i++) {
    shapes_dim_t dim = loss->shape.dims[i];
    if (dim > 1) {
      shapes_Tensor *newLoss = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
      PANIC_IF(newLoss == NULL, ALLOCATION_FAILED);
      *newLoss = shapes_Sum(ctx, loss, i);
      loss = newLoss;
    }
  }

  loss->inputs = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 2);
  shapes_ArrayAppendTensor(loss->inputs, yGround);
  shapes_ArrayAppendTensor(loss->inputs, yPred);

  loss->opType = OP_MSE;
  return *loss;
}
