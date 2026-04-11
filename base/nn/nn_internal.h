
#include "common.h"
#include "nn/nn.h"
#include "result/result.h"

Array *denseLayerParameters(Context *ctx, Layer *state);

Tensor *denseForward(Context *ctx, Layer *layer, Tensor *tensor);
Result sgdStep(Context *ctx, Optimizer *opts, Array *parameters);

void denseBackward(Context *ctx, Tensor *tensor);
void mseBackward(Context *ctx, Tensor *tensor);
