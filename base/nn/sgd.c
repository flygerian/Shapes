#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"

void sgdStep(Context *ctx, Optimizer *opts, Array *parameters) {
  Result res = shapes_optimizer_Sgd(ctx, parameters, opts->learningRate);
  PANIC_IF(res != OK, res);
}

Optimizer shapesnn_SGD(Context *ctx, f32 learningRate) {
  return (Optimizer) {.learningRate = learningRate, .opType = OP_SGD};
}
