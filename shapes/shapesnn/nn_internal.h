#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"

static shapes_OpType LayerOps[] = {OP_DENSE, OP_EMBEDDING, OP_BATCH_NORM, OP_TANH, OP_RELU, OP_MAXPOOL2D, OP_ADAPTIVE_AVG_POOL2D, OP_CONV2D, OP_FLATTEN, OP_SEQUENTIAL, OP_NONE};

typedef struct sequentialModelData {
  olib_Array *layers;
  olib_Array *parameters;
} sequentialModel;

FowardPassOp array_FowardPassOpIdx(olib_Array *array, size_t idx);
void array_AppendFowardPassOp(olib_Array *array, FowardPassOp *op);

olib_Array *denseLayerParameters(Context *ctx, Layer *layer);
olib_Array *embeddingParameters(Context *ctx, Layer *layer);
olib_Array *batchNormLayerParameters(Context *ctx, Layer *layer);
olib_Array *tanhLayerParameters(Context *ctx, Layer *state);

olib_Array *denseLayerTensors(Context *ctx, Layer *layer);
olib_Array *embeddingLayerTensors(Context *ctx, Layer *layer);
olib_Array *batchNormLayerTensors(Context *ctx, Layer *layer);
olib_Array *tanhLayerTensors(Context *ctx, Layer *state);
olib_Array *reluLayerTensors(Context *ctx, Layer *state);
olib_Array *maxPool2dLayerTensors(Context *ctx, Layer *state);
olib_Array *adaptiveAvgPool2dLayerTensors(Context *ctx, Layer *state);
olib_Array *conv2dLayerTensors(Context *ctx, Layer *state);
olib_Array *flattenLayerTensors(Context *ctx);
olib_Array *sequentialModelTensors(Context *ctx, FowardPassOp *modelOp);

Tensor denseForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor embeddingForward(Context *ctx, Layer *layer, Tensor *indices);
Tensor batchNormForward(Context *ctx, Layer *layer, Tensor *inputs);
Tensor tanhForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor sequentialModelForward(Context *ctx, FowardPassOp *modelOp, Tensor *input);
olib_Array *sequentialModelParameters(Context *ctx, FowardPassOp *modelOp);

Result sgdStep(Context *ctx, Optimizer *opts, olib_Array *parameters);
void adamStep(Context *ctx, Optimizer *opts, olib_Array *parameters);

void denseBackward(Context *ctx, Tensor *tensor);
void embeddingBackward(Context *ctx, Tensor *out);
void mseBackward(Context *ctx, Tensor *tensor);
void batchnormBackward(Context *ctx, Tensor *output);
void crossEnthropyBackward(Context *ctx, Tensor *tensor);
void tanhBackward(Context *ctx, Tensor *tensor);
void reluBackward(Context *ctx, Tensor *tensor);
void maxPool2dBackward(Context *ctx, Tensor *tensor);
void adaptiveAvgPool2dBackward(Context *ctx, Tensor *tensor);
void conv2dBackward(Context *ctx, Tensor *tensor);

Tensor reluForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor maxPool2dForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor adaptiveAvgPool2dForward(Context *ctx, Layer *layer, Tensor *tensor);
Tensor conv2dForward(Context *ctx, Layer *layer, Tensor *tensor);
olib_Array *reluLayerParameters(Context *ctx, Layer *state);
olib_Array *maxPool2dLayerParameters(Context *ctx, Layer *state);
olib_Array *adaptiveAvgPool2dLayerParameters(Context *ctx, Layer *state);
olib_Array *conv2dLayerParameters(Context *ctx, Layer *state);

Tensor flattenFoward(Context *ctx, Layer *layer, Tensor *tensor);
olib_Array *flattenParameters(Context *ctx);

void loadIntoTensor(Context *ctx, Tensor *dst, Tensor *src);

void denseLayerLoad(Context *ctx, Layer *layer, olib_Array *tensors);
void embeddingLayerLoad(Context *ctx, Layer *layer, olib_Array *tensors);
void batchNormLayerLoad(Context *ctx, Layer *layer, olib_Array *tensors);
void tanhLayerLoad(Context *ctx, Layer *state, olib_Array *tensors);
void reluLayerLoad(Context *ctx, Layer *state, olib_Array *tensors);
void maxPool2dLayerLoad(Context *ctx, Layer *state, olib_Array *tensors);
void adaptiveAvgPool2dLayerLoad(Context *ctx, Layer *state, olib_Array *tensors);
void conv2dLayerLoad(Context *ctx, Layer *state, olib_Array *tensors);
void flattenLayerLoad(Context *ctx, olib_Array *tensors);
void sequentialModelLoad(Context *ctx, FowardPassOp *modelOp, olib_Array *tensors);
void shapesnn_Load(Context *ctx, FowardPassOp *op, olib_Array *tensors);
