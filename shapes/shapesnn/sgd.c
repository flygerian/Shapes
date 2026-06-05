#include "nn.h"
#include "result.h"
#include "shapes.h"

void sgdStep(shapes_Context *ctx, shapesnn_Optimizer *opts, olib_Array *parameters) {
  Result res = shapes_optimizer_Sgd(ctx, parameters, opts->learningRate);
  PANIC_IF(res != OK, res);
}

shapesnn_Optimizer shapesnn_SGD(shapes_Context *ctx, f32 learningRate) {
  return (shapesnn_Optimizer) {.learningRate = learningRate, .opType = OP_SGD};
}
