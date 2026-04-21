#include "common.h"
#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include "utils_lib/array.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

typedef struct batchNormLayerData {
  Tensor *gamma;
  Tensor *beta;
  dim_t numFeatures;
  f32 epsilon;
  f32 momentum;
  u8 dims;
  bool runningStatsInitialised;
  Tensor *runningMean;
  Tensor *runningVar;
} batchNormLayerData;

typedef struct reshapedFeatures {
  Tensor *reshaped;
  Dim originalShape;
} reshapedFeatures;

reshapedFeatures reshapeToBatchFeature2D(Context *ctx, Tensor *tensor, dim_t numFeatures) {
  u8 numDims = tensor->shape.numOfDims;

  dim_t lastDimSize = tensor->shape.dims[numDims - 1];
  PANIC_IF(lastDimSize != numFeatures, ERR_DIM_MISMATCH);

  if (numDims == 1) {
    return (reshapedFeatures){.reshaped = UnSqueeze(ctx, tensor, 0),
                              .originalShape = tensor->shape};
  }

  u8 lastDim = tensor->shape.numOfDims - 1;
  tensor_size_t numElementsBeforeDim = 0;
  calculateNumElementsBeforeDim(tensor, lastDim, &numElementsBeforeDim);
  reshapedFeatures rf = {.originalShape = tensor->shape};
  rf.reshaped = Reshape(ctx, tensor, SHAPE2D(numElementsBeforeDim, numFeatures));
  return rf;
}

void batchnormBackward(Context *ctx, Tensor *output) {
  Tensor *x = Array_TensorIdx(output->inputs, 0);
  batchNormLayerData *layerData = output->opMetadata;

  Tensor *x2d;
  Tensor *grad2d;
  Dim originalShape;

  switch (layerData->dims) {
    case 1: {
      reshapedFeatures rf = reshapeToBatchFeature2D(ctx, x, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      rf = reshapeToBatchFeature2D(ctx, output->grad, layerData->numFeatures);
      grad2d = rf.reshaped;
    } break;
    default: PANIC_IF(true, ERR_NO_OP);
  }

  BatchNormBackwardResult backwardResult =
      BatchNormBackward(ctx, x2d, grad2d, layerData->gamma, layerData->epsilon);

  AddInPlace(ctx, layerData->beta->grad, backwardResult.dBeta);
  AddInPlace(ctx, layerData->gamma->grad, backwardResult.dGamma);

  Tensor *dX;

  if (originalShape.numOfDims == 1) {
    dX = Squeeze(ctx, backwardResult.dx2d);
  } else {
    dX = Reshape(ctx, backwardResult.dx2d, originalShape);
  }

  PANIC_IF(dX == NULL, ERR_NO_OP);

  Tensor *gradX = ReduceBroadcast(ctx, x, dX);
  AddInPlace(ctx, x->grad, gradX);
}

Array *batchNormLayerParameters(Context *ctx, Layer *layer) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(layer == NULL, ERR_NULL_PTR);

  batchNormLayerData *layerData = layer->layerData;
  Array *params = MakeArray(ctx->memory, sizeof(Tensor *), 2);
  Array_AppendTensor(params, layerData->gamma);
  Array_AppendTensor(params, layerData->beta);

  return params;
}


void updateRunningStats(Context *ctx, batchNormLayerData *layerData, Tensor *mean,
                        Tensor *variance) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(mean == NULL, ERR_NULL_PTR);
  PANIC_IF(variance == NULL, ERR_NULL_PTR);

  PANIC_IF(mean->size != layerData->runningMean->size, ERR_DIM_MISMATCH);
  PANIC_IF(variance->size != layerData->runningVar->size, ERR_DIM_MISMATCH);

  Tensor *keep = T_Float(ctx, SHAPE1D(1), 1.0 - layerData->momentum);
  Tensor *tMomentum = T_Float(ctx, SHAPE1D(1), layerData->momentum);

  Tensor *newRunningMean =
      Add(ctx, Multiply(ctx, layerData->runningMean, keep), Multiply(ctx, mean, tMomentum));
  Copy(ctx, newRunningMean, layerData->runningMean);

  Tensor *newRunningVar =
      Add(ctx, Multiply(ctx, layerData->runningVar, keep), Multiply(ctx, variance, tMomentum));
  Copy(ctx, newRunningVar, layerData->runningVar);
}

Tensor *batchNormForward(Context *ctx, Layer *layer, Tensor *input) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(input == NULL, ERR_NULL_PTR);

  Tensor *x2d;
  Dim originalShape;
  batchNormLayerData *layerData = layer->layerData;

  switch (layerData->dims) {
    case 1: {
      reshapedFeatures rf = reshapeToBatchFeature2D(ctx, input, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      break;
    }
    default: PANIC_IF(true, ERR_DIM_MISMATCH);
  }

  Tensor *bnOut2d;
  if (ctx->isTraining) {
    BatchNormFowardResult bnResult =
        BatchNormForwardTraining(ctx, x2d, layerData->gamma, layerData->beta, layerData->epsilon);
    bnOut2d = bnResult.out;
    if (layerData->runningStatsInitialised) {
      updateRunningStats(ctx, layerData, bnResult.mean, bnResult.variance);
    }
  } else {
    Tensor *mean = layerData->runningMean;
    Tensor *variance = layerData->runningVar;
    Tensor *centered = Subtract(ctx, x2d, mean);
    Tensor *eps = T_Float(ctx, SHAPE1D(1), layerData->epsilon);
    Tensor *invStd = Pow(ctx, Add(ctx, variance, eps), -0.5);
    Tensor *xHat = Multiply(ctx, centered, invStd);
    Tensor *addRes = Add(ctx, Multiply(ctx, xHat, layerData->gamma), layerData->beta);
    bnOut2d = addRes;
  }

  PANIC_IF(bnOut2d == NULL, ERR_NULL_PTR);
  if (originalShape.numOfDims == 1) {
    return Squeeze(ctx, bnOut2d);
  }

  bnOut2d->inputs = MakeArray(ctx->memory, sizeof(Tensor *), 1);
  Array_AppendTensor(bnOut2d->inputs, input);
  bnOut2d->opType = OP_BATCH_NORM;
  bnOut2d->opMetadata = layerData;

  return bnOut2d;
}

FowardPassOp layer_BatchNorm(Context *ctx, size_t numFeatures) {
  Tensor *gamma = T_Float(ctx, SHAPE1D(numFeatures), 1.0);
  Tensor *beta = T_Zeros(ctx, SHAPE1D(numFeatures));

  batchNormLayerData *data = allocateOnCtx(ctx, sizeof(batchNormLayerData));
  data->beta = beta;
  data->gamma = gamma;
  data->numFeatures = numFeatures;
  data->epsilon = 1e-5;
  data->momentum = 0.1;
  data->runningMean = T_Zeros(ctx, SHAPE1D(numFeatures));
  data->runningVar = T_Float(ctx, SHAPE1D(numFeatures), 1);
  data->runningStatsInitialised = true;
  data->dims = 1;

  Layer *layer = allocateOnCtx(ctx, sizeof(Layer));
  layer->layerData = data;

  return (FowardPassOp){.ctx = ctx, .type = OP_BATCH_NORM, .op = layer};
}
