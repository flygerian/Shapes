#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"

void sgdStep(Context *ctx, Optimizer *opts, Array *parameters) {
  Result res = Sgd(ctx, parameters, opts->learningRate);
  PANIC_IF(res != OK, res);
}

Optimizer nn_SGD(f32 learningRate) {
  return (Optimizer){.learningRate = learningRate, .opType = OP_SGD};
}
