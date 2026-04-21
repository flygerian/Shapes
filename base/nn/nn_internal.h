
#include "common.h"
#include "nn/nn.h"
#include "result/result.h"

Array *denseLayerParameters(Context *ctx, Layer *state);
Array *embeddingParameters(Context *ctx, Layer *layer);
Array *batchNormLayerParameters(Context *ctx, Layer *layer);
Array *tanhLayerParameters(Context *ctx, Layer *state);

Tensor *denseForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *embeddingForward(Context *ctx, Layer *layer, Tensor *indices);
Tensor *batchNormForward(Context *ctx, Layer *layer, Tensor *inputs);
Tensor *tanhForward(Context *ctx, Layer *layer, Tensor *tensor);

Result sgdStep(Context *ctx, Optimizer *opts, Array *parameters);
void adamStep(Context *ctx, Optimizer *opts, Array *parameters);

void denseBackward(Context *ctx, Tensor *tensor);
void embeddingBackward(Context *ctx, Tensor *tensor);
void mseBackward(Context *ctx, Tensor *tensor);
void batchnormBackward(Context *ctx, Tensor *output);
void crossEnthropyBackward(Context *ctx, Tensor *tensor);
void tanhBackward(Context *ctx, Tensor *tensor);
