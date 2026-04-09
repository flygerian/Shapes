#ifndef shapes_nn_h
#define shapes_nn_h

#include "../common.h"

typedef struct LayerState {
  Tensor *weights;
  Tensor *bias;
  void *additionalData;
} LayerState;

typedef Tensor *(*LayerForwardFn)(Context *ctx, LayerState *state, Tensor *tensor);
typedef Array *(*LayerParametersFn)(Context *ctx, LayerState *state);

typedef struct Layer {
  LayerState state;
  LayerForwardFn forward;
  LayerParametersFn parameters;
} Layer;


typedef struct OptimizerOpts {
  f32 learningRate;
  void *state;
} OptimizerOpts;

typedef void (*OptimizerStepFn)(Context *ctx, OptimizerOpts opts, Array *parameters);

typedef struct Optimzer {
  OptimizerOpts opts;
  OptimizerStepFn step;
} Optimzer;

Layer nn_Dense(Context *ctx, size_t inputSize, size_t outputSize);
Optimzer nn_SGD(f32 learningRate);
Tensor loss_Mse(Context *ctx, Tensor *yGround, Tensor *yPred);
Array *Backward(Context *ctx, Tensor *tensor);
void ZeroGrad(Context *ctx, Array *graph);

#endif
