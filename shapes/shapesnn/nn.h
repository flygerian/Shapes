#ifndef shapes_nn_h
#define shapes_nn_h

#include "types.h"
#include "array.h"
#include "map.h"
#include "olib.h"

typedef struct Layer {
  Tensor weights;
  Tensor bias;
  void *layerData;
} Layer;

typedef struct FowardPassOp {
  Context *ctx;
  shapes_OpType type;
  shapes_Dtype dtype;
  string label;
  void *op;
} FowardPassOp;

typedef struct NamedTensor {
  String name;
  Tensor tensor;
} NamedTensor;

typedef struct Optimzer {
  f32 learningRate;
  void *state;
  shapes_OpType opType;
} Optimizer;

typedef PtrMap TensorPtrMap;
typedef Array* Array_NamedTensor;
static inline NamedTensor shapes_Array_NamedTensorIdx(Array *array, size_t idx) {
  return *((NamedTensor *)Array_Idx(array, idx));
}

void shapesnn_Array_AppendLayer(Array *array, Layer *layer);
Layer *shapesnn_Array_LayerIdx(Array *array, size_t idx);

TensorPtrMap *Make_TensorPtrMap(Memory *memory);
void TensorPtrMap_Put(TensorPtrMap *map, void *key, Tensor *t);
Tensor *TensorPtrMap_Get(TensorPtrMap *map, void *key);

Tensor shapesnn_Forward(Context *ctx, FowardPassOp *op, Tensor *input);
Array *shapesnn_Parameters(Context *ctx, FowardPassOp *op);
Array_NamedTensor shapesnn_Tensors(Context *ctx, FowardPassOp *op);
void shapesnn_SaveAsSafeTensors(Context *ctx, FowardPassOp *model, string path);
void shapesnn_LoadFromSafeTensors(Context *ctx, FowardPassOp *model, string path);

void shapesnn_SafeTensors_Save(Context *ctx, Array *named, string path);
Array *shapesnn_SafeTensors_Load(Context *ctx, string path);

FowardPassOp shapesnn_Dense(Context *ctx, shapes_Dtype dtype, size_t inputSize, size_t outputSize, bool withBias);
FowardPassOp shapesnn_Embedding(Context *ctx, shapes_Dtype dtype, size_t vocabSize, dim_t embeddingDim);
Optimizer shapesnn_SGD(Context *ctx, f32 learningRate);
Optimizer shapesnn_Adam(Context *ctx, f32 learningRate);
Tensor shapesnn_Mse(Context *ctx, Tensor *yGround, Tensor *yPred);
Tensor shapesnn_CrossEnthropy(Context *ctx, Tensor *yGround, Tensor *logits);
FowardPassOp shapesnn_BatchNorm(Context *ctx, shapes_Dtype dtype, size_t numFeatures);
FowardPassOp shapesnn_BatchNorm2d(Context *ctx, shapes_Dtype dtype, size_t numFeatures);
FowardPassOp shapesnn_Tanh(Context *ctx, shapes_Dtype dtype);
FowardPassOp shapesnn_Relu(Context *ctx, shapes_Dtype dtype);
FowardPassOp shapesnn_MaxPool2d(Context *ctx, shapes_Dtype dtype, dim_t kernelH, dim_t kW, u8 stride);
FowardPassOp shapesnn_AdaptiveAvgPool2d(Context *ctx, shapes_Dtype dtype, dim_t outH, dim_t outW);
FowardPassOp shapesnn_Conv2d(Context *ctx, shapes_Dtype dtype, size_t inChannels, size_t outChannels, dim_t kH, dim_t kW, u8 stride, bool withBias);
FowardPassOp shapesnn_Flatten(Context *ctx, shapes_Dtype type);

FowardPassOp shapesnn_Sequential(Context *ctx, FowardPassOp *layerOps, size_t numLayers, shapes_Dtype dtype);

Array *shapesnn_Backward(Context *ctx, Tensor *tensor);
void shapesnn_ZeroGrad(Context *ctx, Array *graph);

void shapesnn_OptimizerStep(Context *ctx, Optimizer *optimizer, Array *parameters);

void shapesnn_Array_AppendLayer(Array *array, Layer *layer);
Layer *shapesnn_Array_LayerIdx(Array *array, size_t idx);

Tensor shapesnn_Softmax(Context *ctx, Tensor *logits);
#endif
