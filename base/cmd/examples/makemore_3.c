
#include "../../shapes.h"
#include "common.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <termios.h>
#include "nn/nn.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include "utils_lib/array.h"

void makemore_3() {
  Memory *mem = initializeArena((size_t)1024 * 1024 * 4096, 1);
  Context ctx = {.memory = mem};

  f32 xData[12] = {2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0f};
  f32 yData[4] = {1.0, -1.0, -1.0, 1.0};

  Tensor *xs = MakeFromContigousArray(&ctx, SHAPE2D(4, 3), &xData, F32);
  Tensor *ys = MakeFromContigousArray(&ctx, SHAPE1D(4), &yData, F32);

  FowardPassOp *dense = layer_Dense(&ctx, F32, 3, 10, false);
  FowardPassOp *bn1 = layer_BatchNorm(&ctx, F32, 10);
  FowardPassOp *dense2 = layer_Dense(&ctx, F32, 10, 1, true);

  Optimizer *sgd = optimizer_Adam(&ctx, 1e-2);

  ctx.isTraining = true;
  for (u8 epoch = 1; epoch <= 50; epoch++) {
    Tensor *out = Forward(&ctx, dense, xs);
    out = Forward(&ctx, bn1, out);
    Tensor *logits = Forward(&ctx, dense2, out);

    Tensor *logitsSqueezed = Squeeze(&ctx, logits);
    Tensor loss = loss_Mse(&ctx, ys, logitsSqueezed);

    Value lossValue;
    VALUE_GET_FROM_ARR(loss.values, 0, &lossValue, loss.dtype);

    fprintf(stdout, "Loss: %f \n", lossValue.as.f32);

    Backward(&ctx, &loss);

    Array *parameters = MakeArray(ctx.memory, sizeof(Tensor *), 6);

    Array_AppendTensorArray(parameters, Parameters(&ctx, dense));
    Array_AppendTensorArray(parameters, Parameters(&ctx, dense2));
    Array_AppendTensorArray(parameters, Parameters(&ctx, bn1));

    OptimizerStep(&ctx, sgd, parameters);
    ZeroGrad(&ctx, parameters);
  }

  DestroyContext(&ctx);
}
