#include "../../shapes.h"
#include "cblas.h"
#include "common.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <termios.h>
#include "memory.h"
#include "nn/nn.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include "utils_lib/array.h"


int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Memory *mem = initializeArena((size_t)1024 * 1024 * 4096, 1);
  Context ctx = {.memory = mem};

  f32 xData[12] = {2.0, 3.0, -1.0, 3.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, -1.0f};

  f32 yData[4] = {1.0, -1.0, -1.0, 1.0};

  Tensor *xs = MakeFromContigousArray(&ctx, SHAPE2D(4, 3), &xData, 12, F32);
  Tensor *ys = MakeFromContigousArray(&ctx, SHAPE1D(4), &yData, 4, F32);

  Layer dense = nn_Dense(&ctx, 3, 10);
  Layer dense2 = nn_Dense(&ctx, 10, 1);

  Optimzer sgd = nn_SGD(1e-4);

  for (u8 epoch = 1; epoch <= 10; epoch++) {
    Tensor *out = dense.forward(&ctx, &dense.state, xs);
    Tensor *logits = dense2.forward(&ctx, &dense2.state, out);

    Tensor logitsSqueezed;
    Squeeze(&ctx, logits, &logitsSqueezed);
    Tensor loss = loss_Mse(&ctx, ys, &logitsSqueezed);
    
    Value lossValue;
    VALUE_GET_FROM_ARR(loss.values, 0, &lossValue, lossValue.dtype);

    fprintf(stdout, "Loss: %f \n", lossValue.as.f32);

    Array *graph = Backward(&ctx, &loss);

    Array *parameters = MakeArray(ctx.memory, sizeof(Tensor*), 4);

    Array_Append(parameters, &dense.state.weights);
    Array_Append(parameters, &dense.state.bias);
    Array_Append(parameters, &dense2.state.weights);
    Array_Append(parameters, &dense2.state.bias);

    sgd.step(&ctx, sgd.opts, parameters);
    ZeroGrad(&ctx, graph);
  }

  DestroyContext(&ctx);

  return 0;
}
