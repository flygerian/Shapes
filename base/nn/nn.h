#ifndef shapes_nn_h
#define shapes_nn_h

#include "../common.h"

typedef struct Layer {
  Tensor *weights;
  Tensor *bias;
  void *layerData;
} Layer;

typedef struct FowardPassOp {
  Context *ctx;
  OpType type;
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

FowardPassOp layer_Dense(Context *ctx, size_t inputSize, size_t outputSize, bool withBias);
FowardPassOp layer_Embedding(Context *ctx, size_t vocabSize, dim_t embeddingDim);
Optimizer optimizer_SGD(f32 learningRate);
Optimizer optimizer_Adam(Context *ctx, f32 learningRate);
Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred);
Tensor loss_CrossEnthropy(Context *ctx, Tensor *yGround, Tensor *logits);
FowardPassOp layer_BatchNorm(Context *ctx, size_t numFeatures);
FowardPassOp layer_Tanh(Context *ctx);

Array *Backward(Context *ctx, Tensor *tensor);
void ZeroGrad(Context *ctx, Array *graph);

void OptimizerStep(Context *ctx, Optimizer *optimizer, Array *parameters);

#endif
