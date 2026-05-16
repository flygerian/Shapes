#include "nn/nn.h"
#include "result/result.h"
#include "tensor/types.h"

static OpType LayerOps[] = {OP_DENSE,  OP_EMBEDDING, OP_BATCH_NORM, OP_TANH, OP_RELU, OP_MAXPOOL2D, OP_ADAPTIVE_AVG_POOL2D,
                            OP_CONV2D, OP_FLATTEN,   OP_SEQUENTIAL, OP_NONE};

FowardPassOp *array_FowardPassOpIdx(Array *array, size_t idx);
void array_AppendFowardPassOp(Array *array, FowardPassOp *op);

Array *denseLayerParameters(Context *ctx, Layer *state);
Array *embeddingParameters(Context *ctx, Layer *layer);
Array *batchNormLayerParameters(Context *ctx, Layer *layer);
Array *tanhLayerParameters(Context *ctx, Layer *state);

Tensor *denseForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *embeddingForward(Context *ctx, Layer *layer, Tensor *indices);
Tensor *batchNormForward(Context *ctx, Layer *layer, Tensor *inputs);
Tensor *tanhForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *sequentialModelForward(Context *ctx, FowardPassOp *modelOp, Tensor *input);
Array *sequentialModelParameters(Context *ctx, FowardPassOp *modelOp);

Result sgdStep(Context *ctx, Optimizer *opts, Array *parameters);
void adamStep(Context *ctx, Optimizer *opts, Array *parameters);

void denseBackward(Context *ctx, Tensor *tensor);
void embeddingBackward(Context *ctx, Tensor *tensor);
void mseBackward(Context *ctx, Tensor *tensor);
void batchnormBackward(Context *ctx, Tensor *output);
void crossEnthropyBackward(Context *ctx, Tensor *tensor);
void tanhBackward(Context *ctx, Tensor *tensor);
void reluBackward(Context *ctx, Tensor *tensor);
void maxPool2dBackward(Context *ctx, Tensor *tensor);
void adaptiveAvgPool2dBackward(Context *ctx, Tensor *tensor);
void conv2dBackward(Context *ctx, Tensor *tensor);

Tensor *reluForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *maxPool2dForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *adaptiveAvgPool2dForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor *conv2dForward(Context *ctx, Layer *layer, Tensor *tensor);
Array *reluLayerParameters(Context *ctx, Layer *state);
Array *maxPool2dLayerParameters(Context *ctx, Layer *state);
Array *adaptiveAvgPool2dLayerParameters(Context *ctx, Layer *state);
Array *conv2dLayerParameters(Context *ctx, Layer *state);

Tensor *flattenFoward(Context *ctx, Layer *layer, Tensor *tensor);
Array *flattenParameters(Context *ctx);
