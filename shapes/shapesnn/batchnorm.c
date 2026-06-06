#include "nn.h"
#include "nn_internal.h"
#include "result.h"
#include "shapes.h"
#include "shapes_internal.h"
#include "types.h"
#include "value.h"
#include "array.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

typedef struct batchNormLayerData {
  shapes_Tensor gamma;
  shapes_Tensor beta;
  shapes_dim_t numFeatures;
  f32 epsilon;
  f32 momentum;
  u8 dims;
  bool runningStatsInitialised;
  shapes_Tensor runningMean;
  shapes_Tensor runningVar;
} batchNormLayerData;

typedef struct reshapedFeatures {
  shapes_Tensor reshaped;
  shapes_Dim originalShape;
} reshapedFeatures;

reshapedFeatures reshapeToBatchFeature2D(shapes_Context *ctx, shapes_Tensor *tensor, shapes_dim_t numFeatures) {
  u8 numDims = tensor->shape.numOfDims;

  shapes_dim_t lastDimSize = tensor->shape.dims[numDims - 1];
  PANIC_IF(lastDimSize != numFeatures, ERR_DIM_MISMATCH);

  if (numDims == 1) {
    shapes_Tensor unsqueezed = shapes_UnSqueeze(ctx, tensor, 0);
    return (reshapedFeatures){.reshaped = unsqueezed, .originalShape = tensor->shape};
  }

  u8 lastDim = tensor->shape.numOfDims - 1;
  shapes_tensor_size_t numElementsBeforeDim = 0;
  calculateNumElementsBeforeDim(tensor, lastDim, &numElementsBeforeDim);
  reshapedFeatures rf = {.originalShape = tensor->shape};
  shapes_Tensor reshaped = shapes_Reshape(ctx, tensor, SHAPE2D(numElementsBeforeDim, numFeatures));
  rf.reshaped = reshaped;
  return rf;
}

reshapedFeatures reshapeNHWCToBatchFeature2D(shapes_Context *ctx, shapes_Tensor *tensor, shapes_dim_t numFeatures) {
  PANIC_IF(tensor->shape.numOfDims != 4, ERR_DIM_MISMATCH);
  shapes_dim_t channelDim = tensor->shape.dims[3];
  PANIC_IF(channelDim != numFeatures, ERR_DIM_MISMATCH);

  reshapedFeatures rf = reshapeToBatchFeature2D(ctx, tensor, numFeatures);
  reshapedFeatures result = {.reshaped = rf.reshaped, .originalShape = tensor->shape};
  return result;
}

shapes_Tensor restoreBatchNorm2DOutput(shapes_Context *ctx, shapes_Tensor *x2d, shapes_Dim originalShape) {
  PANIC_IF(originalShape.numOfDims != 4, ERR_DIM_MISMATCH);
  shapes_dim_t n = originalShape.dims[0];
  shapes_dim_t h = originalShape.dims[1];
  shapes_dim_t w = originalShape.dims[2];
  shapes_dim_t c = originalShape.dims[3];
  return shapes_Reshape(ctx, x2d, SHAPE4D(n, h, w, c));
}

void batchnormBackward(shapes_Context *ctx, shapes_Tensor *output) {
  shapes_Tensor x = shapes_ArrayTensorIdx(output->inputs, 0);
  batchNormLayerData *layerData = output->opMetadata;

  shapes_Tensor x2d;
  shapes_Tensor grad2d;
  shapes_Dim originalShape;

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

  shapes_BatchNormBackwardResult backwardResult = shapes_BatchNormBackward(ctx, &x2d, &grad2d, &layerData->gamma, layerData->epsilon);

  shapes_AddInPlace(ctx, layerData->beta.grad, &backwardResult.dBeta);
  shapes_AddInPlace(ctx, layerData->gamma.grad, &backwardResult.dGamma);

  shapes_Tensor dX;
  shapes_Tensor dXVal;

  if (layerData->dims == 2) {
    dX = restoreBatchNorm2DOutput(ctx, &backwardResult.dx2d, originalShape);
  } else if (originalShape.numOfDims == 1) {
    dX = shapes_Squeeze(ctx, &backwardResult.dx2d);
  } else {
    dX = shapes_Reshape(ctx, &backwardResult.dx2d, originalShape);
  }

  shapes_Tensor gradX = shapes_ReduceBroadcast(ctx, &x, &dX);
  shapes_AddInPlace(ctx, x.grad, &gradX);
}

olib_Array *batchNormLayerParameters(shapes_Context *ctx, shapesnn_layer *layer) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(layer == NULL, ERR_NULL_PTR);

  batchNormLayerData *layerData = layer->layerData;
  olib_Array *params = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 2);
  shapes_ArrayAppendTensor(params, &layerData->gamma);
  shapes_ArrayAppendTensor(params, &layerData->beta);

  return params;
}

olib_Array *batchNormLayerTensors(shapes_Context *ctx, shapesnn_layer *layer) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(layer == NULL, ERR_NULL_PTR);

  batchNormLayerData *layerData = layer->layerData;
  olib_Array *tensors = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 4);
  shapes_ArrayAppendTensor(tensors, &layerData->gamma);
  shapes_ArrayAppendTensor(tensors, &layerData->beta);
  shapes_ArrayAppendTensor(tensors, &layerData->runningMean);
  shapes_ArrayAppendTensor(tensors, &layerData->runningVar);

  return tensors;
}

void batchNormLayerLoad(shapes_Context *ctx, shapesnn_layer *layer, olib_Array *tensors) {
  PANIC_IF(tensors->size != 4, ERR_DIM_MISMATCH);
  batchNormLayerData *layerData = layer->layerData;

  shapes_Tensor gamma = shapes_ArrayTensorIdx(tensors, 0);
  shapes_Tensor beta = shapes_ArrayTensorIdx(tensors, 1);
  shapes_Tensor runningMean = shapes_ArrayTensorIdx(tensors, 2);
  shapes_Tensor runningVar = shapes_ArrayTensorIdx(tensors, 3);

  loadIntoTensor(ctx, &layerData->gamma, &gamma);
  loadIntoTensor(ctx, &layerData->beta, &beta);
  loadIntoTensor(ctx, &layerData->runningMean, &runningMean);
  loadIntoTensor(ctx, &layerData->runningVar, &runningVar);
}

void updateRunningStats(shapes_Context *ctx, batchNormLayerData *layerData, shapes_Tensor *mean, shapes_Tensor *variance) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(mean == NULL, ERR_NULL_PTR);
  PANIC_IF(variance == NULL, ERR_NULL_PTR);

  PANIC_IF(mean->size != layerData->runningMean.size, ERR_DIM_MISMATCH);
  PANIC_IF(variance->size != layerData->runningVar.size, ERR_DIM_MISMATCH);

  shapes_Tensor *keep = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(keep == NULL, ALLOCATION_FAILED);
  *keep = shapes_MakeFloatTensor(ctx, SHAPE1D(1), 1.0 - layerData->momentum);
  shapes_Tensor *tMomentum = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(tMomentum == NULL, ALLOCATION_FAILED);
  *tMomentum = shapes_MakeFloatTensor(ctx, SHAPE1D(1), layerData->momentum);

  shapes_Tensor meanWeighted = shapes_Multiply(ctx, &layerData->runningMean, keep);
  shapes_Tensor meanDelta = shapes_Multiply(ctx, mean, tMomentum);
  shapes_Tensor newRunningMean = shapes_Add(ctx, &meanWeighted, &meanDelta);
  shapes_Copy(ctx, &newRunningMean, &layerData->runningMean);

  shapes_Tensor varWeighted = shapes_Multiply(ctx, &layerData->runningVar, keep);
  shapes_Tensor varDelta = shapes_Multiply(ctx, variance, tMomentum);
  shapes_Tensor newRunningVar = shapes_Add(ctx, &varWeighted, &varDelta);
  shapes_Copy(ctx, &newRunningVar, &layerData->runningVar);
}

shapes_Tensor batchNormForward(shapes_Context *ctx, shapesnn_layer *layer, shapes_Tensor *input) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(input == NULL, ERR_NULL_PTR);

  shapes_Tensor x2d;
  shapes_Dim originalShape;
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

  shapes_Tensor bnOut2d;
  if (ctx->isTraining) {
    shapes_BatchNormFowardResult bnResult = shapes_BatchNormForwardTraining(ctx, &x2d, &layerData->gamma, &layerData->beta, layerData->epsilon);
    bnOut2d = bnResult.out;
    if (layerData->runningStatsInitialised) {
      updateRunningStats(ctx, layerData, &bnResult.mean, &bnResult.variance);
    }
  } else {
    shapes_Tensor mean = layerData->runningMean;
    shapes_Tensor variance = layerData->runningVar;
    shapes_Tensor centered = shapes_Subtract(ctx, &x2d, &mean);
    shapes_Tensor eps = shapes_MakeFloatTensor(ctx, SHAPE1D(1), layerData->epsilon);
    shapes_Tensor varPlusEps = shapes_Add(ctx, &variance, &eps);
    shapes_Tensor invStd = shapes_Pow(ctx, &varPlusEps, -0.5);
    shapes_Tensor xHat = shapes_Multiply(ctx, &centered, &invStd);
    shapes_Tensor scaled = shapes_Multiply(ctx, &xHat, &layerData->gamma);
    bnOut2d = shapes_Add(ctx, &scaled, &layerData->beta);
  }

  shapes_Tensor out;
  if (layerData->dims == 2) {
    out = restoreBatchNorm2DOutput(ctx, &bnOut2d, originalShape);
  } else if (originalShape.numOfDims == 1) {
    out = shapes_Squeeze(ctx, &bnOut2d);
  } else {
    out = shapes_Reshape(ctx, &bnOut2d, originalShape);
  }

  out.inputs = olib_MakeArray(ctx->memory, sizeof(shapes_Tensor), 1);
  shapes_ArrayAppendTensor(out.inputs, input);
  out.opType = OP_BATCH_NORM;
  out.opMetadata = layerData;

  return out;
}

shapesnn_FowardPassOp shapesnn_BatchNorm(shapes_Context *ctx, shapes_Dtype dtype, size_t numFeatures) {
  shapes_Tensor gamma = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  gamma.label = "gamma";
  shapes_Tensor beta = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), dtype);
  beta.label = "beta";

  batchNormLayerData *data = olib_Allocate(ctx->memory, sizeof(batchNormLayerData));
  data->beta = beta;
  data->gamma = gamma;
  data->numFeatures = numFeatures;
  data->epsilon = 1e-5;
  data->momentum = 0.1;
  data->runningMean = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), dtype);
  data->runningMean.label = "running_mean";
  data->runningVar = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), dtype);
  data->runningVar.label = "running_var";
  shapes_SetValues(&data->runningVar, VALUE(dtype, 1));
  data->runningStatsInitialised = true;
  data->dims = 1;

  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  *layer = (shapesnn_layer){0};
  layer->layerData = data;

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_BATCH_NORM, .dtype = dtype, .op = layer};
}

shapesnn_FowardPassOp shapesnn_BatchNorm2d(shapes_Context *ctx, shapes_Dtype dtype, size_t numFeatures) {
  shapes_Tensor gamma = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  gamma.label = "gamma";
  shapes_Tensor beta = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), dtype);
  beta.label = "beta";

  batchNormLayerData *data = olib_Allocate(ctx->memory, sizeof(batchNormLayerData));
  data->beta = beta;
  data->gamma = gamma;
  data->numFeatures = numFeatures;
  data->epsilon = 1e-5;
  data->momentum = 0.1;
  data->runningMean = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), dtype);
  data->runningMean.label = "running_mean";
  data->runningVar = shapes_MakeFloatTensor(ctx, SHAPE1D(numFeatures), 1.0);
  data->runningVar.label = "running_var";
  data->runningStatsInitialised = true;
  data->dims = 2;

  shapesnn_layer *layer = olib_Allocate(ctx->memory, sizeof(shapesnn_layer));
  *layer = (shapesnn_layer){0};
  layer->layerData = data;

  return (shapesnn_FowardPassOp){.ctx = ctx, .type = OP_BATCH_NORM, .dtype = dtype, .op = layer};
}
