#ifndef shapes_nn_h
#define shapes_nn_h

#include "types.h"
#include "utils_lib/array.h"
#include "utils_lib/map.h"

typedef struct Layer {
  Tensor *weights;
  Tensor *bias;
  void *layerData;
} Layer;

typedef struct FowardPassOp {
  Context *ctx;
  OpType type;
  Dtype dtype;
  void *op;
} FowardPassOp;

typedef struct Optimzer {
  f32 learningRate;
  void *state;
  OpType opType;
} Optimizer;

typedef PtrMap TensorPtrMap;

TensorPtrMap *Make_TensorPtrMap(Memory *memory);
void TensorPtrMap_Put(TensorPtrMap *map, void *key, Tensor *t);
Tensor *TensorPtrMap_Get(TensorPtrMap *map, void *key);

Tensor *Forward(Context *ctx, FowardPassOp *op, Tensor *input);
Array *Parameters(Context *ctx, FowardPassOp *op);

FowardPassOp *layer_Dense(Context *ctx, Dtype dtype, size_t inputSize, size_t outputSize, bool withBias);
FowardPassOp *layer_Embedding(Context *ctx, Dtype dtype, size_t vocabSize, dim_t embeddingDim);
Optimizer *optimizer_SGD(Context *ctx, f32 learningRate);
Optimizer *optimizer_Adam(Context *ctx, f32 learningRate);
Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred);
Tensor loss_CrossEnthropy(Context *ctx, Tensor *yGround, Tensor *logits);
FowardPassOp *layer_BatchNorm(Context *ctx, Dtype dtype, size_t numFeatures);
FowardPassOp *layer_BatchNorm2d(Context *ctx, Dtype dtype, size_t numFeatures);
FowardPassOp *layer_Tanh(Context *ctx, Dtype dtype);
FowardPassOp *layer_Relu(Context *ctx, Dtype dtype);
FowardPassOp *layer_MaxPool2d(Context *ctx, Dtype dtype, dim_t kernelH, dim_t kernelW, u8 stride);
FowardPassOp *layer_AdaptiveAvgPool2d(Context *ctx, Dtype dtype, dim_t outH, dim_t outW);
FowardPassOp *layer_Conv2d(Context *ctx, Dtype dtype, size_t inChannels, size_t outChannels, dim_t kH, dim_t kW, u8 stride, bool withBias);
FowardPassOp *layer_Flatten(Context *ctx, Dtype type);

FowardPassOp *Make_Sequential(Context *ctx, FowardPassOp **layerOps, size_t numLayers, Dtype dtype);

Array *Backward(Context *ctx, Tensor *tensor);
void ZeroGrad(Context *ctx, Array *graph);

void OptimizerStep(Context *ctx, Optimizer *optimizer, Array *parameters);

void Array_AppendLayer(Array *array, Layer *layer);
Layer *Array_LayerIdx(Array *array, size_t idx);

Tensor *nn_Softmax(Context *ctx, Tensor *logits);
#endif
