#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "types.h"
#include "value.h"
#include "array.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

typedef struct batchNormLayerData {
  Tensor gamma;
  Tensor beta;
  dim_t numFeatures;
  f32 epsilon;
  f32 momentum;
  u8 dims;
  bool runningStatsInitialised;
  Tensor runningMean;
  Tensor runningVar;
} batchNormLayerData;

typedef struct reshapedFeatures {
  Tensor reshaped;
  Dim originalShape;
} reshapedFeatures;

reshapedFeatures reshapeToBatchFeature2D(Context *ctx, Tensor *tensor, dim_t numFeatures) {
  u8 numDims = tensor->shape.numOfDims;

  dim_t lastDimSize = tensor->shape.dims[numDims - 1];
  PANIC_IF(lastDimSize != numFeatures, ERR_DIM_MISMATCH);

  if (numDims == 1) {
    Tensor unsqueezed = shapes_UnSqueeze(ctx, tensor, 0);
    return (reshapedFeatures){.reshaped = unsqueezed, .originalShape = tensor->shape};
  }

  u8 lastDim = tensor->shape.numOfDims - 1;
  tensor_size_t numElementsBeforeDim = 0;
  calculateNumElementsBeforeDim(tensor, lastDim, &numElementsBeforeDim);
  reshapedFeatures rf = {.originalShape = tensor->shape};
  Tensor reshaped = shapes_Reshape(ctx, tensor, SHAPE2D(numElementsBeforeDim, numFeatures));
  rf.reshaped = reshaped;
  return rf;
}

reshapedFeatures reshapeNHWCToBatchFeature2D(Context *ctx, Tensor *tensor, dim_t numFeatures) {
  PANIC_IF(tensor->shape.numOfDims != 4, ERR_DIM_MISMATCH);
  dim_t channelDim = tensor->shape.dims[3];
  PANIC_IF(channelDim != numFeatures, ERR_DIM_MISMATCH);

  reshapedFeatures rf = reshapeToBatchFeature2D(ctx, tensor, numFeatures);
  reshapedFeatures result = {.reshaped = rf.reshaped, .originalShape = tensor->shape};
  return result;
}

Tensor restoreBatchNorm2DOutput(Context *ctx, Tensor *x2d, Dim originalShape) {
  PANIC_IF(originalShape.numOfDims != 4, ERR_DIM_MISMATCH);
  dim_t n = originalShape.dims[0];
  dim_t h = originalShape.dims[1];
  dim_t w = originalShape.dims[2];
  dim_t c = originalShape.dims[3];
  return shapes_Reshape(ctx, x2d, SHAPE4D(n, h, w, c));
}

void batchnormBackward(Context *ctx, Tensor *output) {
  Tensor x = shapes_Array_TensorIdx(output->inputs, 0);
  batchNormLayerData *layerData = output->opMetadata;

  Tensor x2d;
  Tensor grad2d;
  Dim originalShape;

  switch (layerData->dims) {
    case 1: {
      reshapedFeatures rf = reshapeToBatchFeature2D(ctx, &x, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      rf = reshapeToBatchFeature2D(ctx, output->grad, layerData->numFeatures);
      grad2d = rf.reshaped;
    } break;
    case 2: {
      reshapedFeatures rf = reshapeNHWCToBatchFeature2D(ctx, &x, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      rf = reshapeNHWCToBatchFeature2D(ctx, output->grad, layerData->numFeatures);
      grad2d = rf.reshaped;
    } break;
    default: PANIC_IF(true, ERR_NO_OP);
  }

  BatchNormBackwardResult backwardResult = shapes_layer_BatchNormBackward(ctx, &x2d, &grad2d, &layerData->gamma, layerData->epsilon);

  shapes_AddInPlace(ctx, layerData->beta.grad, &backwardResult.dBeta);
  shapes_AddInPlace(ctx, layerData->gamma.grad, &backwardResult.dGamma);

  Tensor dX;
  Tensor dXVal;

  if (layerData->dims == 2) {
    dX = restoreBatchNorm2DOutput(ctx, &backwardResult.dx2d, originalShape);
  } else if (originalShape.numOfDims == 1) {
    dX = shapes_Squeeze(ctx, &backwardResult.dx2d);
  } else {
    dX = shapes_Reshape(ctx, &backwardResult.dx2d, originalShape);
  }

  Tensor gradX = shapes_ReduceBroadcast(ctx, &x, &dX);
  shapes_AddInPlace(ctx, x.grad, &gradX);
}

Array *batchNormLayerParameters(Context *ctx, Layer *layer) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(layer == NULL, ERR_NULL_PTR);

  batchNormLayerData *layerData = layer->layerData;
  Array *params = MakeArray(ctx->memory, sizeof(Tensor), 2);
  shapes_Array_AppendTensor(params, &layerData->gamma);
  shapes_Array_AppendTensor(params, &layerData->beta);

  return params;
}

Array *batchNormLayerTensors(Context *ctx, Layer *layer) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(layer == NULL, ERR_NULL_PTR);

  batchNormLayerData *layerData = layer->layerData;
  Array *tensors = MakeArray(ctx->memory, sizeof(Tensor), 4);
  shapes_Array_AppendTensor(tensors, &layerData->gamma);
  shapes_Array_AppendTensor(tensors, &layerData->beta);
  shapes_Array_AppendTensor(tensors, &layerData->runningMean);
  shapes_Array_AppendTensor(tensors, &layerData->runningVar);

  return tensors;
}

void batchNormLayerLoad(Context *ctx, Layer *layer, Array *tensors) {
  PANIC_IF(tensors->size != 4, ERR_DIM_MISMATCH);
  batchNormLayerData *layerData = layer->layerData;

  Tensor gamma = shapes_Array_TensorIdx(tensors, 0);
  Tensor beta = shapes_Array_TensorIdx(tensors, 1);
  Tensor runningMean = shapes_Array_TensorIdx(tensors, 2);
  Tensor runningVar = shapes_Array_TensorIdx(tensors, 3);

  loadIntoTensor(ctx, &layerData->gamma, &gamma);
  loadIntoTensor(ctx, &layerData->beta, &beta);
  loadIntoTensor(ctx, &layerData->runningMean, &runningMean);
  loadIntoTensor(ctx, &layerData->runningVar, &runningVar);
}

void updateRunningStats(Context *ctx, batchNormLayerData *layerData, Tensor *mean, Tensor *variance) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(mean == NULL, ERR_NULL_PTR);
  PANIC_IF(variance == NULL, ERR_NULL_PTR);

  PANIC_IF(mean->size != layerData->runningMean.size, ERR_DIM_MISMATCH);
  PANIC_IF(variance->size != layerData->runningVar.size, ERR_DIM_MISMATCH);

  Tensor *keep = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(keep == NULL, ALLOCATION_FAILED);
  *keep = shapes_Make_FloatTensor(ctx, SHAPE1D(1), 1.0 - layerData->momentum);
  Tensor *tMomentum = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(tMomentum == NULL, ALLOCATION_FAILED);
  *tMomentum = shapes_Make_FloatTensor(ctx, SHAPE1D(1), layerData->momentum);

  Tensor meanWeighted = shapes_Multiply(ctx, &layerData->runningMean, keep);
  Tensor meanDelta = shapes_Multiply(ctx, mean, tMomentum);
  Tensor newRunningMean = shapes_Add(ctx, &meanWeighted, &meanDelta);
  shapes_Copy(ctx, &newRunningMean, &layerData->runningMean);

  Tensor varWeighted = shapes_Multiply(ctx, &layerData->runningVar, keep);
  Tensor varDelta = shapes_Multiply(ctx, variance, tMomentum);
  Tensor newRunningVar = shapes_Add(ctx, &varWeighted, &varDelta);
  shapes_Copy(ctx, &newRunningVar, &layerData->runningVar);
}

Tensor batchNormForward(Context *ctx, Layer *layer, Tensor *input) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(input == NULL, ERR_NULL_PTR);

  Tensor x2d;
  Dim originalShape;
  batchNormLayerData *layerData = layer->layerData;

  switch (layerData->dims) {
    case 1: {
      reshapedFeatures rf = reshapeToBatchFeature2D(ctx, input, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      break;
    }
    case 2: {
      reshapedFeatures rf = reshapeNHWCToBatchFeature2D(ctx, input, layerData->numFeatures);
      x2d = rf.reshaped;
      originalShape = rf.originalShape;
      break;
    }
    default: PANIC_IF(true, ERR_DIM_MISMATCH);
  }

  Tensor bnOut2d;
  if (ctx->isTraining) {
    BatchNormFowardResult bnResult = shapes_layer_BatchNormForwardTraining(ctx, &x2d, &layerData->gamma, &layerData->beta, layerData->epsilon);
    bnOut2d = bnResult.out;
    if (layerData->runningStatsInitialised) {
      updateRunningStats(ctx, layerData, &bnResult.mean, &bnResult.variance);
    }
  } else {
    Tensor mean = layerData->runningMean;
    Tensor variance = layerData->runningVar;
    Tensor centered = shapes_Subtract(ctx, &x2d, &mean);
    Tensor eps = shapes_Make_FloatTensor(ctx, SHAPE1D(1), layerData->epsilon);
    Tensor varPlusEps = shapes_Add(ctx, &variance, &eps);
    Tensor invStd = shapes_Pow(ctx, &varPlusEps, -0.5);
    Tensor xHat = shapes_Multiply(ctx, &centered, &invStd);
    Tensor scaled = shapes_Multiply(ctx, &xHat, &layerData->gamma);
    bnOut2d = shapes_Add(ctx, &scaled, &layerData->beta);
  }

  Tensor out;
  if (layerData->dims == 2) {
    out = restoreBatchNorm2DOutput(ctx, &bnOut2d, originalShape);
  } else if (originalShape.numOfDims == 1) {
    out = shapes_Squeeze(ctx, &bnOut2d);
  } else {
    out = shapes_Reshape(ctx, &bnOut2d, originalShape);
  }

  out.inputs = MakeArray(ctx->memory, sizeof(Tensor), 1);
  shapes_Array_AppendTensor(out.inputs, input);
  out.opType = OP_BATCH_NORM;
  out.opMetadata = layerData;

  return out;
}

FowardPassOp shapesnn_BatchNorm(Context *ctx, Dtype dtype, size_t numFeatures) {
  Tensor gamma = shapes_Make_FloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  gamma.label = "gamma";
  Tensor beta = t_Zeros(ctx, SHAPE1D(numFeatures), dtype);
  beta.label = "beta";

  batchNormLayerData *data = allocate(ctx->memory, sizeof(batchNormLayerData));
  data->beta = beta;
  data->gamma = gamma;
  data->numFeatures = numFeatures;
  data->epsilon = 1e-5;
  data->momentum = 0.1;
  data->runningMean = t_Zeros(ctx, SHAPE1D(numFeatures), dtype);
  data->runningMean.label = "running_mean";
  data->runningVar = t_Zeros(ctx, SHAPE1D(numFeatures), dtype);
  data->runningVar.label = "running_var";
  shapes_SetValues(&data->runningVar, VALUE(dtype, 1));
  data->runningStatsInitialised = true;
  data->dims = 1;

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  *layer = (Layer){0};
  layer->layerData = data;

  return (FowardPassOp){.ctx = ctx, .type = OP_BATCH_NORM, .dtype = dtype, .op = layer};
}

FowardPassOp shapesnn_BatchNorm2d(Context *ctx, Dtype dtype, size_t numFeatures) {
  Tensor gamma = shapes_Make_FloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  gamma.label = "gamma";
  Tensor beta = t_Zeros(ctx, SHAPE1D(numFeatures), dtype);
  beta.label = "beta";

  batchNormLayerData *data = allocate(ctx->memory, sizeof(batchNormLayerData));
  data->beta = beta;
  data->gamma = gamma;
  data->numFeatures = numFeatures;
  data->epsilon = 1e-5;
  data->momentum = 0.1;
  data->runningMean = t_Zeros(ctx, SHAPE1D(numFeatures), dtype);
  data->runningMean.label = "running_mean";
  data->runningVar = shapes_Make_FloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  data->runningVar.label = "running_var";
  data->runningStatsInitialised = true;
  data->dims = 2;

  Layer *layer = allocate(ctx->memory, sizeof(Layer));
  *layer = (Layer){0};
  layer->layerData = data;

  return (FowardPassOp){.ctx = ctx, .type = OP_BATCH_NORM, .dtype = dtype, .op = layer};
}
