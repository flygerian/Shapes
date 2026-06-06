#ifndef shapes_nn_h
#define shapes_nn_h

#include "types.h"
#include "array.h"
#include "map.h"
#include "olib.h"

typedef struct {
  shapes_Tensor weights;
  shapes_Tensor bias;
  void *layerData;
} shapesnn_layer;

typedef struct shapesnn_FowardPassOp {
  shapes_Context *ctx;
  shapes_OpType type;
  shapes_Dtype dtype;
  string label;
  void *op;
} shapesnn_FowardPassOp;

typedef struct shapes_NamedTensor {
  olib_String name;
  shapes_Tensor tensor;
} shapesnn_NamedTensor;

typedef struct shapes_Optimzer {
  f32 learningRate;
  void *state;
  shapes_OpType opType;
} shapesnn_Optimizer;

typedef PtrMap shapesnn_TensorPtrMap;
typedef olib_Array* shapes_ArrayNamedTensor;
static inline shapesnn_NamedTensor shapes_Array_NamedTensorIdx(olib_Array *array, size_t idx) {
  return *((shapesnn_NamedTensor *)olib_ArrayIdx(array, idx));
}

shapesnn_TensorPtrMap *shapesnn_MakeTensorPtrMap(olib_Memory *memory);
void shapes_TensorPtrMapPut(shapesnn_TensorPtrMap *map, void *key, shapes_Tensor *t);
shapes_Tensor *shapes_TensorPtrMapGet(shapesnn_TensorPtrMap *map, void *key);
shapes_Tensor shapesnn_Forward(shapes_Context *ctx, shapesnn_FowardPassOp *op, shapes_Tensor *input);
olib_Array *shapesnn_Parameters(shapes_Context *ctx, shapesnn_FowardPassOp *op);
shapes_ArrayNamedTensor shapesnn_Tensors(shapes_Context *ctx, shapesnn_FowardPassOp *op);
void shapesnn_SaveAsSafeTensors(shapes_Context *ctx, shapesnn_FowardPassOp *model, string path);
void shapesnn_LoadFromSafeTensors(shapes_Context *ctx, shapesnn_FowardPassOp *model, string path);
void shapesnn_SafeTensors_Save(shapes_Context *ctx, olib_Array *named, string path);
olib_Array *shapesnn_SafeTensors_Load(shapes_Context *ctx, string path);
shapesnn_FowardPassOp shapesnn_Dense(shapes_Context *ctx, shapes_Dtype dtype, size_t inputSize, size_t outputSize, bool withBias);
shapesnn_FowardPassOp shapesnn_Embedding(shapes_Context *ctx, shapes_Dtype dtype, size_t vocabSize, shapes_dim_t embeddingDim);
shapesnn_Optimizer shapesnn_SGD(shapes_Context *ctx, f32 learningRate);
shapesnn_Optimizer shapesnn_Adam(shapes_Context *ctx, f32 learningRate);
shapes_Tensor shapesnn_Mse(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *yPred);
shapes_Tensor shapesnn_CrossEnthropy(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *logits);
shapesnn_FowardPassOp shapesnn_BatchNorm(shapes_Context *ctx, shapes_Dtype dtype, size_t numFeatures);
shapesnn_FowardPassOp shapesnn_BatchNorm2d(shapes_Context *ctx, shapes_Dtype dtype, size_t numFeatures);
shapesnn_FowardPassOp shapesnn_Tanh(shapes_Context *ctx, shapes_Dtype dtype);
shapesnn_FowardPassOp shapesnn_Relu(shapes_Context *ctx, shapes_Dtype dtype);
shapesnn_FowardPassOp shapesnn_MaxPool2d(shapes_Context *ctx, shapes_Dtype dtype, shapes_dim_t kernelH, shapes_dim_t kW, u8 stride);
shapesnn_FowardPassOp shapesnn_AdaptiveAvgPool2d(shapes_Context *ctx, shapes_Dtype dtype, shapes_dim_t outH, shapes_dim_t outW);
shapesnn_FowardPassOp shapesnn_Conv2d(shapes_Context *ctx, shapes_Dtype dtype, size_t inChannels, size_t outChannels, shapes_dim_t kH, shapes_dim_t kW, u8 stride, bool withBias);
shapesnn_FowardPassOp shapesnn_Flatten(shapes_Context *ctx, shapes_Dtype type);
shapesnn_FowardPassOp shapesnn_Sequential(shapes_Context *ctx, shapesnn_FowardPassOp *layerOps, size_t numLayers, shapes_Dtype dtype);
olib_Array *shapesnn_Backward(shapes_Context *ctx, shapes_Tensor *tensor);
void shapesnn_ZeroGrad(shapes_Context *ctx, olib_Array *graph);
void shapesnn_OptimizerStep(shapes_Context *ctx, shapesnn_Optimizer *optimizer, olib_Array *parameters);
shapes_Tensor shapesnn_Softmax(shapes_Context *ctx, shapes_Tensor *logits);
#endif
