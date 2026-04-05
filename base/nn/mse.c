#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include <stddef.h>
#include "stdlib.h"
#include "utils_lib/array.h"

void mseBackward(Context *ctx, Tensor* tensor) {
  
  PANIC_IF(ctx == NULL || tensor == NULL || tensor->inputs == NULL || tensor->grad == NULL,
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(tensor->inputs->size < 2, ERR_NULL_TENSOR_PROVIDED);

  Tensor *yGround = *(Tensor **)Array_Idx(tensor->inputs, 0);
  Tensor *yPred = *(Tensor **)Array_Idx(tensor->inputs, 1);

  PANIC_IF(yGround == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yPred == NULL, ERR_NULL_TENSOR_PROVIDED);


	// ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
  Tensor diff;
  Subtract(ctx, yPred, yGround, &diff);
  Tensor *two = T_Float(ctx, diff.shape, 2.0);
  Tensor localGrad;
  Multiply(ctx, two, &diff, &localGrad);
  Tensor gradYPred;
  Multiply(ctx, tensor->grad, &localGrad, &gradYPred);
  Tensor reducedGradYPred;
  ReduceBroadcast(ctx, yPred, &gradYPred, &reducedGradYPred);
  AddInPlace(ctx, yPred->grad, &reducedGradYPred);

	// ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
  Tensor negLocalGrad;
  Negate(ctx, &localGrad, &negLocalGrad);
  Tensor gradYGround;
  Multiply(ctx, tensor->grad, &negLocalGrad, &gradYGround);
  Tensor reducedGradYGround;
  ReduceBroadcast(ctx, yGround, &gradYGround, &reducedGradYGround);
  AddInPlace(ctx, yGround->grad, &reducedGradYGround);
}

Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred) {
  Tensor diff;
  Result result = Subtract(ctx, yPred, yGround, &diff);
  PANIC_IF(result != OK, result);

  Tensor se;
  result = Pow(ctx, &diff, 2, &se);
  PANIC_IF(result != OK, result);

  Tensor loss = se;

  for (dim_t i = 0; i < loss.shape.numOfDims; i++) {
    dim_t dim = loss.shape.dims[i];
    if (dim > 1) {
      Sum(ctx, &loss, &loss, dim);
    }
  }

  loss.inputs = MakeArray(ctx->memory, sizeof(Tensor*), 2);
  Array_Append(loss.inputs, &yGround);
  Array_Append(loss.inputs, &yPred);

  loss.backward = mseBackward;
  return loss;
}
