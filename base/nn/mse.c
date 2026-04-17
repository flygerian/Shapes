#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include <stddef.h>
#include "tensor/tensor_internal.h"
#include "utils_lib/array.h"

void mseBackward(Context *ctx, Tensor *tensor) {
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  Tensor *yGround = Array_TensorIdx(tensor->inputs, 0);
  Tensor *yPred = Array_TensorIdx(tensor->inputs, 1);

  PANIC_IF(yGround == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yPred == NULL, ERR_NULL_TENSOR_PROVIDED);

  // ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
  Tensor *diff = Subtract(ctx, yPred, yGround);
  Tensor *two = T_Float(ctx, diff->shape, 2.0);
  Tensor *localGrad = Multiply(ctx, two, diff);
  Tensor *gradYPred = Multiply(ctx, tensor->grad, localGrad);
  Tensor *reducedGradYPred = ReduceBroadcast(ctx, yPred, gradYPred);
  AddInPlace(ctx, yPred->grad, reducedGradYPred);

  // ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
  Tensor *negLocalGrad = Negate(ctx, localGrad);
  Tensor *gradYGround = Multiply(ctx, tensor->grad, negLocalGrad);
  Tensor *reducedGradYGround = ReduceBroadcast(ctx, yGround, gradYGround);
  AddInPlace(ctx, yGround->grad, reducedGradYGround);
}

Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred) {
  Tensor *diff = Subtract(ctx, yPred, yGround);

  Tensor *loss = Pow(ctx, diff, 2);

  for (dim_t i = 0; i < loss->shape.numOfDims; i++) {
    dim_t dim = loss->shape.dims[i];
    if (dim > 1) {
      loss = Sum(ctx, loss, i);
    }
  }

  loss->inputs = MakeArray(ctx->memory, sizeof(Tensor *), 2);
  Array_AppendTensor(loss->inputs, yGround);
  Array_AppendTensor(loss->inputs, yPred);

  loss->opType = OP_MSE;
  loss->backward = mseBackward;
  return *loss;
}
