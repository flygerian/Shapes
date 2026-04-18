
#include "common.h"
#include "nn/nn.h"
#include "result/result.h"

Array* denseLayerParameters(Context *ctx, Layer *state);
Array* batchNormLayerParameters(Context *ctx, Layer *layer);

Tensor *denseForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor* batchNormForward(Context *ctx, Layer *layer, Tensor *inputs);

Result sgdStep(Context *ctx, Optimizer *opts, Array *parameters);
void adamStep(Context *ctx, Optimizer *opts, Array *parameters);

void denseBackward(Context *ctx, Tensor *tensor);
void mseBackward(Context *ctx, Tensor *tensor);
void batchnormBackward(Context *ctx, Tensor *output);
void crossEnthropyBackward(Context *ctx, Tensor *tensor);
