#ifndef shapes_nn_h
#define shapes_nn_h

#include "../common.h"

typedef struct Layer {
  Tensor *weights;
  Tensor *bias;
  void *layerData;
} Layer;

typedef struct FowardPassOp {
  OpType type;
  void* op;
} FowardPassOp;


typedef struct Optimzer {
  f32 learningRate;
  void *state;
  OpType opType;
} Optimizer;


Tensor* Forward(Context *ctx, FowardPassOp *op, Tensor *input);
Array* Parameters(Context *ctx, FowardPassOp *op);

FowardPassOp nn_Dense(Context *ctx, size_t inputSize, size_t outputSize);
Optimizer nn_SGD(f32 learningRate);
Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred);

Array *Backward(Context *ctx, Tensor *tensor);
void ZeroGrad(Context *ctx, Array *graph);

void OptimizerStep(Context *ctx, Optimizer *optimizer, Array *parameters);

#endif
