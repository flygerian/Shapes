#include "result/result.h"
#include "shapes.h"
#include <stddef.h>
#include "tensor_internal.h"
#include "utils_lib/array.h"

void mseBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  Tensor *yGround = shapes_Array_TensorIdx(tensor->inputs, 0);
  Tensor *yPred = shapes_Array_TensorIdx(tensor->inputs, 1);

  PANIC_IF(yGround == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yPred == NULL, ERR_NULL_TENSOR_PROVIDED);

  // ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
  Tensor diff = shapes_Subtract(ctx, yPred, yGround);
  Tensor two = shapes_Make_FloatTensor(ctx, diff.shape, 2.0);
  Tensor localGrad = shapes_Multiply(ctx, &two, &diff);
  Tensor gradYPred = shapes_Multiply(ctx, tensor->grad, &localGrad);
  Tensor reducedGradYPred = shapes_ReduceBroadcast(ctx, yPred, &gradYPred);
  shapes_AddInPlace(ctx, yPred->grad, &reducedGradYPred);

  // ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
  Tensor negLocalGrad = shapes_Negate(ctx, &localGrad);
  Tensor gradYGround = shapes_Multiply(ctx, tensor->grad, &negLocalGrad);
  Tensor reducedGradYGround = shapes_ReduceBroadcast(ctx, yGround, &gradYGround);
  shapes_AddInPlace(ctx, yGround->grad, &reducedGradYGround);
}

Tensor shapesnn_Mse(Context *ctx, Tensor *yGround, Tensor *yPred) {
  Tensor diffVal = shapes_Subtract(ctx, yPred, yGround);

  Tensor *loss = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(loss == NULL, ALLOCATION_FAILED);
  *loss = shapes_Pow(ctx, &diffVal, 2);

  for (dim_t i = 0; i < loss->shape.numOfDims; i++) {
    dim_t dim = loss->shape.dims[i];
    if (dim > 1) {
      Tensor *newLoss = allocate(ctx->memory, sizeof(Tensor));
      PANIC_IF(newLoss == NULL, ALLOCATION_FAILED);
      *newLoss = shapes_Sum(ctx, loss, i);
      loss = newLoss;
    }
  }

  loss->inputs = MakeArray(ctx->memory, sizeof(Tensor), 2);
  shapes_Array_AppendTensor(loss->inputs, yGround);
  shapes_Array_AppendTensor(loss->inputs, yPred);

  loss->opType = OP_MSE;
  return *loss;
}
