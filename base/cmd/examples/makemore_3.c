
#include "../../shapes.h"
#include "common.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <termios.h>
#include "nn/nn.h"
#include "value.h"
#include "utils_lib/array.h"

void makemore_3() {
  Memory *mem = initializeArena((size_t)1024 * 1024 * 4096, 1);
  Context ctx = {.memory = mem};

  f32 xData[12] = {2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0f};
  f32 yData[4] = {1.0, -1.0, -1.0, 1.0};

  Tensor *xs = shapes_Make_FromContigousArray(&ctx, SHAPE2D(4, 3), &xData, F32);
  Tensor *ys = shapes_Make_FromContigousArray(&ctx, SHAPE1D(4), &yData, F32);

  FowardPassOp *dense = shapesnn_Dense(&ctx, F32, 3, 10, false);
  FowardPassOp *bn1 = shapesnn_BatchNorm(&ctx, F32, 10);
  FowardPassOp *dense2 = shapesnn_Dense(&ctx, F32, 10, 1, true);

  Optimizer *sgd = shapesnn_Adam(&ctx, 1e-2);

  ctx.isTraining = true;
  for (u8 epoch = 1; epoch <= 50; epoch++) {
    Tensor *out = shapesnn_Forward(&ctx, dense, xs);
    out = shapesnn_Forward(&ctx, bn1, out);
    Tensor *logits = shapesnn_Forward(&ctx, dense2, out);

    Tensor *logitsSqueezed = shapes_Squeeze(&ctx, logits);
    Tensor loss = shapesnn_Mse(&ctx, ys, logitsSqueezed);

    Value lossValue;
    VALUE_GET_FROM_ARR(loss.values, 0, &lossValue, loss.dtype);

    fprintf(stdout, "Loss: %f \n", lossValue.as.f32);

    shapesnn_Backward(&ctx, &loss);

    Array *parameters = MakeArray(ctx.memory, sizeof(Tensor *), 6);

    shapes_Array_AppendTensorArray(parameters, shapesnn_Parameters(&ctx, dense));
    shapes_Array_AppendTensorArray(parameters, shapesnn_Parameters(&ctx, dense2));
    shapes_Array_AppendTensorArray(parameters, shapesnn_Parameters(&ctx, bn1));

    shapesnn_OptimizerStep(&ctx, sgd, parameters);
    shapesnn_ZeroGrad(&ctx, parameters);
  }

  shapes_DestroyContext(&ctx);
}
