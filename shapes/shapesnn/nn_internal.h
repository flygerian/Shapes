#include "nn.h"
#include "result.h"
#include "types.h"
#include "array.h"

static shapes_OpType LayerOps[] = {OP_DENSE, OP_EMBEDDING, OP_BATCH_NORM, OP_TANH, OP_RELU, OP_MAXPOOL2D, OP_ADAPTIVE_AVG_POOL2D, OP_CONV2D, OP_FLATTEN, OP_SEQUENTIAL, OP_NONE};

typedef struct sequentialModelData {
  olib_Array *layers;
  olib_Array *parameters;
} sequentialModel;

shapesnn_FowardPassOp array_FowardPassOpIdx(olib_Array *array, size_t idx);
void array_AppendFowardPassOp(olib_Array *array, shapesnn_FowardPassOp *op);

olib_Array *denseLayerParameters(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *embeddingParameters(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *batchNormLayerParameters(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *tanhLayerParameters(shapes_Context *ctx, shapesnn_layer *state);

olib_Array *denseLayerTensors(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *embeddingLayerTensors(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *batchNormLayerTensors(shapes_Context *ctx, shapesnn_layer *layer);
olib_Array *tanhLayerTensors(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *reluLayerTensors(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *maxPool2dLayerTensors(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *adaptiveAvgPool2dLayerTensors(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *conv2dLayerTensors(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *flattenLayerTensors(shapes_Context *ctx);
olib_Array *sequentialModelTensors(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp);

Tensor denseForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
Tensor embeddingForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *indices);
Tensor batchNormForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *inputs);
Tensor tanhForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
Tensor sequentialModelForward(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp, Tensor *input);
olib_Array *sequentialModelParameters(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp);

Result sgdStep(shapes_Context *ctx, shapesnn_Optimizer *opts, olib_Array *parameters);
void adamStep(shapes_Context *ctx, shapesnn_Optimizer *opts, olib_Array *parameters);

void denseBackward(shapes_Context *ctx, Tensor *tensor);
void embeddingBackward(shapes_Context *ctx, Tensor *out);
void mseBackward(shapes_Context *ctx, Tensor *tensor);
void batchnormBackward(shapes_Context *ctx, Tensor *output);
void crossEnthropyBackward(shapes_Context *ctx, Tensor *tensor);
void tanhBackward(shapes_Context *ctx, Tensor *tensor);
void reluBackward(shapes_Context *ctx, Tensor *tensor);
void maxPool2dBackward(shapes_Context *ctx, Tensor *tensor);
void adaptiveAvgPool2dBackward(shapes_Context *ctx, Tensor *tensor);
void conv2dBackward(shapes_Context *ctx, Tensor *tensor);

Tensor reluForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
Tensor maxPool2dForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
Tensor adaptiveAvgPool2dForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
Tensor conv2dForward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
olib_Array *reluLayerParameters(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *maxPool2dLayerParameters(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *adaptiveAvgPool2dLayerParameters(shapes_Context *ctx, shapesnn_layer *state);
olib_Array *conv2dLayerParameters(shapes_Context *ctx, shapesnn_layer *state);

Tensor flattenFoward(shapes_Context *ctx, shapesnn_layer *layer, Tensor *tensor);
olib_Array *flattenParameters(shapes_Context *ctx);

void loadIntoTensor(shapes_Context *ctx, Tensor *dst, Tensor *src);

void denseLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors);
void embeddingLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors);
void batchNormLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors);
void tanhLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors);
void reluLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors);
void maxPool2dLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors);
void adaptiveAvgPool2dLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors);
void conv2dLayerLoad(shapes_Context *ctx, shapesnn_layer *state, olib_Array *tensors);
void flattenLayerLoad(shapes_Context *ctx, olib_Array *tensors);
void sequentialModelLoad(shapes_Context *ctx, shapesnn_FowardPassOp *modelOp, olib_Array *tensors);
void shapesnn_Load(shapes_Context *ctx, shapesnn_FowardPassOp *op, olib_Array *tensors);
