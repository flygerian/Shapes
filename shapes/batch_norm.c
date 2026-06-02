#include "result.h"
#include "shapes.h"

#include "tensor_internal.h"
#include "types.h"
#include <string.h>

BatchNormFowardResult shapes_layer_BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon) {
  PANIC_IF(isInvalidTensor(x2d) || isInvalidTensor(gamma) || isInvalidTensor(beta), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(x2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1 || beta->shape.numOfDims != 1, ERR_DIM_MISMATCH);

  PANIC_IF(x2d->dtype != gamma->dtype || x2d->dtype != beta->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64, ERR_DTYPE_MISMATCH);

  dim_t batchSize = x2d->shape.dims[0];
  dim_t numFeatures = x2d->shape.dims[1];
  PANIC_IF(gamma->shape.dims[0] != numFeatures || beta->shape.dims[0] != numFeatures, ERR_DIM_MISMATCH);

  Tensor *xContig = materializeTensorOnContext(ctx, x2d);
  Tensor *gammaContig = materializeTensorOnContext(ctx, gamma);
  Tensor *betaContig = materializeTensorOnContext(ctx, beta);

  Tensor out = t_Zeros(ctx, SHAPE2D(batchSize, numFeatures), x2d->dtype);
  Tensor mean = t_Zeros(ctx, SHAPE1D(numFeatures), x2d->dtype);
  Tensor variance = t_Zeros(ctx, SHAPE1D(numFeatures), x2d->dtype);

  if (x2d->dtype == F64) {
    f64 *xVals = xContig->values;
    f64 *gammaVals = gammaContig->values;
    f64 *betaVals = betaContig->values;
    f64 *outVals = out.values;

    f64 *meanAcrossBatch = mean.values;
    f64 *varianceAcrossBatch = variance.values;

    f64 *invStd = allocate(ctx->memory, sizeof(f64) * numFeatures);
    PANIC_IF(invStd == NULL, ALLOCATION_FAILED);

    memset(meanAcrossBatch, 0, sizeof(f64) * numFeatures);

    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t col = 0; col < numFeatures; col++) {
        meanAcrossBatch[col] += xVals[row + col];
      }
    }

    f64 mAsDouble = (f64)batchSize;
    for (tensor_size_t j = 0; j < numFeatures; j++) {
      meanAcrossBatch[j] /= mAsDouble;
    }

    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t col = 0; col < numFeatures; col++) {
        f64 centered = xVals[row + col] - meanAcrossBatch[col];
        varianceAcrossBatch[col] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < numFeatures; j++) {
      varianceAcrossBatch[j] /= mAsDouble;
      invStd[j] = 1.0 / sqrt(varianceAcrossBatch[j] + (f64)epsilon);
    }

    for (tensor_size_t i = 0; i < batchSize; i++) {
      tensor_size_t row = i * numFeatures;
      for (tensor_size_t col = 0; col < numFeatures; col++) {
        f64 xHat = (xVals[row + col] - meanAcrossBatch[col]) * invStd[col];
        outVals[row + col] = xHat * gammaVals[col] + betaVals[col];
      }
    }
  } else {
    f32 *xVals = xContig->values;
    f32 *gammaVals = gammaContig->values;
    f32 *betaVals = betaContig->values;
    f32 *outVals = out.values;
    f32 *meanAcrossBatch = mean.values;
    f32 *varianceAcrossBatch = variance.values;
    f32 *invStd = allocate(ctx->memory, sizeof(f32) * numFeatures);
    PANIC_IF(invStd == NULL, ERR_OUT_OF_MEMORY);

    memset(meanAcrossBatch, 0, sizeof(f32) * numFeatures);

    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
        meanAcrossBatch[feature] += xVals[row + feature];
      }
    }

    f32 batchAsFloat = (f32)batchSize;
    for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
      meanAcrossBatch[feature] /= batchAsFloat;
    }

    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
        f32 centered = xVals[row + feature] - meanAcrossBatch[feature];
        varianceAcrossBatch[feature] += centered * centered;
      }
    }

    for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
      varianceAcrossBatch[feature] /= batchAsFloat;
      invStd[feature] = 1.0f / sqrtf(varianceAcrossBatch[feature] + epsilon);
    }

    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
        f32 xHat = (xVals[row + feature] - meanAcrossBatch[feature]) * invStd[feature];
        outVals[row + feature] = xHat * gammaVals[feature] + betaVals[feature];
      }
    }
  }

  return (BatchNormFowardResult){.out = out, .mean = mean, .variance = variance};
}

BatchNormBackwardResult shapes_layer_BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon) {
  PANIC_IF(isInvalidTensor(x2d) || isInvalidTensor(grad2d) || isInvalidTensor(gamma), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(x2d->shape.numOfDims != 2 || grad2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1, ERR_DIM_MISMATCH);
  PANIC_IF(x2d->dtype != grad2d->dtype || x2d->dtype != gamma->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64, ERR_DTYPE_MISMATCH);

  dim_t m = x2d->shape.dims[0];
  dim_t n = x2d->shape.dims[1];

  PANIC_IF(grad2d->shape.dims[0] != m || grad2d->shape.dims[1] != n || gamma->shape.dims[0] != n, ERR_DIM_MISMATCH);

  Tensor *xContig = materializeTensorOnContext(ctx, x2d);
  Tensor *gradContig = materializeTensorOnContext(ctx, grad2d);
  Tensor *gammaContig = materializeTensorOnContext(ctx, gamma);

  Tensor dX = t_Zeros(ctx, SHAPE2D(m, n), x2d->dtype);
  Tensor dGamma = t_Zeros(ctx, SHAPE1D(n), x2d->dtype);
  Tensor dBeta = t_Zeros(ctx, SHAPE1D(n), x2d->dtype);

  if (x2d->dtype == F64) {
    f64 *xVals = xContig->values;
    f64 *dyVals = gradContig->values;
    f64 *gammaVals = gammaContig->values;

    f64 *dxVals = dX.values;
    f64 *dGammaVals = dGamma.values;
    f64 *dBetaVals = dBeta.values;

    f64 *mean = allocate(ctx->memory, sizeof(f64) * n);
    f64 *var = allocate(ctx->memory, sizeof(f64) * n);
    f64 *invStd = allocate(ctx->memory, sizeof(f64) * n);
    f64 *sumDXHat = allocate(ctx->memory, sizeof(f64) * n);
    f64 *sumDXHatXHat = allocate(ctx->memory, sizeof(f64) * n);
    PANIC_IF(mean == NULL || var == NULL || invStd == NULL || sumDXHat == NULL || sumDXHatXHat == NULL, ERR_OUT_OF_MEMORY);

    memset(mean, 0, sizeof(f64) * n);
    memset(var, 0, sizeof(f64) * n);
    memset(sumDXHat, 0, sizeof(f64) * n);
    memset(sumDXHatXHat, 0, sizeof(f64) * n);

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        mean[j] += xVals[row + j];
      }
    }

    f64 mAsDouble = (f64)m;
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] /= mAsDouble;
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f64 centered = xVals[row + j] - mean[j];
        var[j] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < n; j++) {
      var[j] /= mAsDouble;
      invStd[j] = 1.0 / sqrt(var[j] + (f64)epsilon);
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f64 centered = xVals[row + j] - mean[j];
        f64 xHat = centered * invStd[j];
        f64 dy = dyVals[row + j];
        f64 dxHat = dy * gammaVals[j];

        dBetaVals[j] += dy;
        dGammaVals[j] += dy * xHat;
        sumDXHat[j] += dxHat;
        sumDXHatXHat[j] += dxHat * xHat;
      }
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f64 centered = xVals[row + j] - mean[j];
        f64 xHat = centered * invStd[j];
        f64 dy = dyVals[row + j];
        f64 dxHat = dy * gammaVals[j];

        f64 numerator = mAsDouble * dxHat - sumDXHat[j] - xHat * sumDXHatXHat[j];
        dxVals[row + j] = (invStd[j] * numerator) / mAsDouble;
      }
    }
  } else {
    f32 *xVals = xContig->values;
    f32 *dyVals = gradContig->values;
    f32 *gammaVals = gammaContig->values;

    f32 *dxVals = dX.values;
    f32 *dGammaVals = dGamma.values;
    f32 *dBetaVals = dBeta.values;

    f32 *mean = allocate(ctx->memory, sizeof(f32) * n);
    f32 *var = allocate(ctx->memory, sizeof(f32) * n);
    f32 *invStd = allocate(ctx->memory, sizeof(f32) * n);
    f32 *sumDXHat = allocate(ctx->memory, sizeof(f32) * n);
    f32 *sumDXHatXHat = allocate(ctx->memory, sizeof(f32) * n);
    PANIC_IF(mean == NULL || var == NULL || invStd == NULL || sumDXHat == NULL || sumDXHatXHat == NULL, ERR_OUT_OF_MEMORY);

    memset(mean, 0, sizeof(f32) * n);
    memset(var, 0, sizeof(f32) * n);
    memset(sumDXHat, 0, sizeof(f32) * n);
    memset(sumDXHatXHat, 0, sizeof(f32) * n);

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        mean[j] += xVals[row + j];
      }
    }

    f32 mAsFloat = (f32)m;
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] /= mAsFloat;
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f32 centered = xVals[row + j] - mean[j];
        var[j] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < n; j++) {
      var[j] /= mAsFloat;
      invStd[j] = 1.0f / sqrtf(var[j] + epsilon);
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f32 centered = xVals[row + j] - mean[j];
        f32 xHat = centered * invStd[j];
        f32 dy = dyVals[row + j];
        f32 dxHat = dy * gammaVals[j];

        dBetaVals[j] += dy;
        dGammaVals[j] += dy * xHat;
        sumDXHat[j] += dxHat;
        sumDXHatXHat[j] += dxHat * xHat;
      }
    }

    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        f32 centered = xVals[row + j] - mean[j];
        f32 xHat = centered * invStd[j];
        f32 dy = dyVals[row + j];
        f32 dxHat = dy * gammaVals[j];

        f32 numerator = mAsFloat * dxHat - sumDXHat[j] - xHat * sumDXHatXHat[j];
        dxVals[row + j] = (invStd[j] * numerator) / mAsFloat;
      }
    }
  }

  return (BatchNormBackwardResult){.dBeta = dBeta, .dGamma = dGamma, .dx2d = dX};
}
