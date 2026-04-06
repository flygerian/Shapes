#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"


void sgdStep(Context *ctx, OptimizerOpts opts, Array *parameters) {
  Result res = Sgd(ctx, parameters, opts.learningRate);
  PANIC_IF(res != OK, res);
}

Optimzer nn_SGD(f32 learningRate) {
  return (Optimzer){.step = sgdStep, .opts = (OptimizerOpts){.learningRate = learningRate}};
}
