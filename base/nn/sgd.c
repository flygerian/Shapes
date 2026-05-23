#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"

void sgdStep(Context *ctx, Optimizer *opts, Array *parameters) {
  Result res = shapes_optimizer_Sgd(ctx, parameters, opts->learningRate);
  PANIC_IF(res != OK, res);
}

Optimizer *optimizer_SGD(Context *ctx, f32 learningRate) {
  Optimizer *opt = allocate(ctx->memory, sizeof(Optimizer));
  *opt = (Optimizer){.learningRate = learningRate, .opType = OP_SGD};
  return opt;
}
