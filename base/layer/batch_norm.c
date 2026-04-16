#include "result/result.h"
#include "shapes.h"

#include "tensor/tensor_internal.h"


BatchNormFowardResult BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon) {
  // Training forward expects flattened activations:
  // x2d: [m, n] where m=batch/items and n=features/channels.
  // gamma, beta: [n] affine parameters.
  // Outputs:
  // out: [m, n], mean: [n], variance: [n].
  PANIC_IF(isInvalidTensor(x2d) || isInvalidTensor(gamma) || isInvalidTensor(beta), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(x2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1 || beta->shape.numOfDims != 1, ERR_DIM_MISMATCH);

  PANIC_IF (x2d->dtype != gamma->dtype || x2d->dtype != beta->dtype, ERR_DTYPE_MISMATCH); 
  PANIC_IF (x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64, ERR_DTYPE_MISMATCH); 

  dim_t batchSize = x2d->shape.dims[0];
  dim_t numFeatures = x2d->shape.dims[1];
  PANIC_IF (gamma->shape.dims[0] != numFeatures || beta->shape.dims[0] != numFeatures, ERR_DIM_MISMATCH); 

  Tensor *xContig = materializeTensorOnContext(ctx, x2d);
  Tensor *gammaContig = materializeTensorOnContext(ctx, gamma);
  Tensor *betaContig = materializeTensorOnContext(ctx, beta);

  Result res = OK;
  Tensor *out = t_Zeros(ctx, SHAPE2D(batchSize, numFeatures), x2d->dtype);
  Tensor *mean = t_Zeros(ctx, SHAPE1D(numFeatures), x2d->dtype);
  Tensor *variance = t_Zeros(ctx, SHAPE1D(numFeatures), x2d->dtype);

  if (x2d->dtype == F64) {
    f64 *xVals = xContig->values;
    f64 *gammaVals = gammaContig->values;
    f64 *betaVals = betaContig->values;
    f64 *outVals = out->values;

    // Mean and variance across batch (dim 0), one value per feature/column.
    f64 *meanAcrossBatch = mean->values;
    f64 *varianceAcrossBatch = variance->values;

    f64 *invStd = allocate(ctx->memory, sizeof(f64) * numFeatures);
    PANIC_IF (invStd == NULL, ALLOCATION_FAILED); 

    // Initialize per-feature accumulators.
    for (tensor_size_t j = 0; j < numFeatures; j++) {
      meanAcrossBatch[j] = 0.0;
      varianceAcrossBatch[j] = 0.0;
    }

    // 1) mean over batch axis for each feature.
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

    // 2) variance over batch axis for each feature.
    for (tensor_size_t i = 0; i < batchSize; i++) {
      tensor_size_t row = i * numFeatures;
      for (tensor_size_t col = 0; col < numFeatures; col++) {
        f64 centered = xVals[row + col] - meanAcrossBatch[col];
        varianceAcrossBatch[col] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < numFeatures; j++) {
      varianceAcrossBatch[j] /= mAsDouble;
      invStd[j] = 1.0 / sqrt(varianceAcrossBatch[j] + (f64)epsilon);
    }

    // 3) normalize, then apply learned scale (gamma) and shift (beta).
    for (tensor_size_t i = 0; i < batchSize; i++) {
      tensor_size_t row = i * numFeatures;
      for (tensor_size_t col = 0; col < numFeatures; col++) {
        f64 xHat = (xVals[row + col] - meanAcrossBatch[col]) * invStd[col];
        outVals[row + col] = xHat * gammaVals[col] + betaVals[col];
      }
    }

    freeAlloc(ctx->memory, invStd);
  } else {
    f32 *xVals = xContig->values;
    f32 *gammaVals = gammaContig->values;
    f32 *betaVals = betaContig->values;
    f32 *outVals = out->values;
    f32 *meanAcrossBatch = mean->values;
    f32 *varianceAcrossBatch = variance->values;
    f32 *invStd = allocate(ctx->memory, sizeof(f32) * numFeatures);
    PANIC_IF (invStd == NULL, ERR_OUT_OF_MEMORY); 

    // Initialize per-feature accumulators.
    for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
      meanAcrossBatch[feature] = 0.0f;
      varianceAcrossBatch[feature] = 0.0f;
    }

    // 1) mean over batch axis for each feature.
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

    // 2) variance over batch axis for each feature.
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

    // 3) normalize, then apply learned scale (gamma) and shift (beta).
    for (tensor_size_t currentBatch = 0; currentBatch < batchSize; currentBatch++) {
      tensor_size_t row = currentBatch * numFeatures;
      for (tensor_size_t feature = 0; feature < numFeatures; feature++) {
        f32 xHat = (xVals[row + feature] - meanAcrossBatch[feature]) * invStd[feature];
        outVals[row + feature] = xHat * gammaVals[feature] + betaVals[feature];
      }
    }

    freeAlloc(ctx->memory, invStd);
  }

cleanup:
  freeIfContingousCopy(ctx, xContig);
  freeIfContingousCopy(ctx, gammaContig);
  freeIfContingousCopy(ctx, betaContig);


  return (BatchNormFowardResult) {.out = out, .mean = mean, .variance = variance};
}

BatchNormBackwardResult BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon) {
  // Backward uses the same flattened [m, n] layout:
  // grad2d is dL/dy, gamma is [n], outputs are dX [m, n], dGamma [n], dBeta [n].
  PANIC_IF(isInvalidTensor(x2d) || isInvalidTensor(grad2d) || isInvalidTensor(gamma),  ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF (x2d->shape.numOfDims != 2 || grad2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1,  ERR_DIM_MISMATCH); 
  PANIC_IF(x2d->dtype != grad2d->dtype || x2d->dtype != gamma->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64, ERR_DTYPE_MISMATCH); 

  dim_t m = x2d->shape.dims[0];
  dim_t n = x2d->shape.dims[1];

  PANIC_IF(grad2d->shape.dims[0] != m || grad2d->shape.dims[1] != n || gamma->shape.dims[0] != n, ERR_DIM_MISMATCH); 
  // Keep backward math on packed memory for predictable stride-1 access.
  Tensor *xContig = materializeTensorOnContext(ctx, x2d);
  Tensor *gradContig = materializeTensorOnContext(ctx, grad2d);
  ;
  Tensor *gammaContig = materializeTensorOnContext(ctx, gamma);


  Result res = OK;
  Tensor *dX = t_Zeros(ctx, SHAPE2D(m, n), x2d->dtype);
  Tensor *dGamma = t_Zeros(ctx, SHAPE1D(n), x2d->dtype);
  Tensor *dBeta = t_Zeros(ctx, SHAPE1D(n), x2d->dtype);

  if (x2d->dtype == F64) {
    f64 *xVals = xContig->values;
    f64 *dyVals = gradContig->values;
    f64 *gammaVals = gammaContig->values;

    f64 *dxVals = dX->values;
    f64 *dGammaVals = dGamma->values;
    f64 *dBetaVals = dBeta->values;

    f64 *mean = allocate(ctx->memory, sizeof(f64) * n);
    f64 *var = allocate(ctx->memory, sizeof(f64) * n);
    f64 *invStd = allocate(ctx->memory, sizeof(f64) * n);
    f64 *sumDXHat = allocate(ctx->memory, sizeof(f64) * n);
    f64 *sumDXHatXHat = allocate(ctx->memory, sizeof(f64) * n);
    if (mean == NULL || var == NULL || invStd == NULL || sumDXHat == NULL || sumDXHatXHat == NULL) {
      if (sumDXHatXHat != NULL) {
        freeAlloc(ctx->memory, sumDXHatXHat);
      }
      if (sumDXHat != NULL) {
        freeAlloc(ctx->memory, sumDXHat);
      }
      if (invStd != NULL) {
        freeAlloc(ctx->memory, invStd);
      }
      if (var != NULL) {
        freeAlloc(ctx->memory, var);
      }
      if (mean != NULL) {
        freeAlloc(ctx->memory, mean);
      }
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    // Recompute batch stats used by batch norm (matching forward training path).
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] = 0.0;
      var[j] = 0.0;
      invStd[j] = 0.0;
      dGammaVals[j] = 0.0;
      dBetaVals[j] = 0.0;
      sumDXHat[j] = 0.0;
      sumDXHatXHat[j] = 0.0;
    }

    // 1) Recompute mean per feature.
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

    // 2) Recompute variance and inverse std per feature.
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

    // 3) Accumulate parameter grads and helper sums for dX.
    // dBeta = sum(dy), dGamma = sum(dy * xHat).
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

    // 4) Closed-form dX:
    // dX = invStd/m * (m*dxHat - sum(dxHat) - xHat*sum(dxHat*xHat)).
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

    freeAlloc(ctx->memory, sumDXHatXHat);
    freeAlloc(ctx->memory, sumDXHat);
    freeAlloc(ctx->memory, invStd);
    freeAlloc(ctx->memory, var);
    freeAlloc(ctx->memory, mean);
  } else {
    f32 *xVals = xContig->values;
    f32 *dyVals = gradContig->values;
    f32 *gammaVals = gammaContig->values;

    f32 *dxVals = dX->values;
    f32 *dGammaVals = dGamma->values;
    f32 *dBetaVals = dBeta->values;

    f32 *mean = allocate(ctx->memory, sizeof(f32) * n);
    f32 *var = allocate(ctx->memory, sizeof(f32) * n);
    f32 *invStd = allocate(ctx->memory, sizeof(f32) * n);
    f32 *sumDXHat = allocate(ctx->memory, sizeof(f32) * n);
    f32 *sumDXHatXHat = allocate(ctx->memory, sizeof(f32) * n);
    if (mean == NULL || var == NULL || invStd == NULL || sumDXHat == NULL || sumDXHatXHat == NULL) {
      if (sumDXHatXHat != NULL) {
        freeAlloc(ctx->memory, sumDXHatXHat);
      }
      if (sumDXHat != NULL) {
        freeAlloc(ctx->memory, sumDXHat);
      }
      if (invStd != NULL) {
        freeAlloc(ctx->memory, invStd);
      }
      if (var != NULL) {
        freeAlloc(ctx->memory, var);
      }
      if (mean != NULL) {
        freeAlloc(ctx->memory, mean);
      }
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    // Recompute batch stats used by batch norm (matching forward training path).
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] = 0.0f;
      var[j] = 0.0f;
      invStd[j] = 0.0f;
      dGammaVals[j] = 0.0f;
      dBetaVals[j] = 0.0f;
      sumDXHat[j] = 0.0f;
      sumDXHatXHat[j] = 0.0f;
    }

    // 1) Recompute mean per feature.
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

    // 2) Recompute variance and inverse std per feature.
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

    // 3) Accumulate parameter grads and helper sums for dX.
    // dBeta = sum(dy), dGamma = sum(dy * xHat).
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

    // 4) Closed-form dX:
    // dX = invStd/m * (m*dxHat - sum(dxHat) - xHat*sum(dxHat*xHat)).
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

    freeAlloc(ctx->memory, sumDXHatXHat);
    freeAlloc(ctx->memory, sumDXHat);
    freeAlloc(ctx->memory, invStd);
    freeAlloc(ctx->memory, var);
    freeAlloc(ctx->memory, mean);
  }

cleanup:
  freeIfContingousCopy(ctx, xContig);
  freeIfContingousCopy(ctx, gammaContig);
  freeIfContingousCopy(ctx, gradContig);

  return (BatchNormBackwardResult) {.dBeta = dBeta, .dGamma = dGamma, .dx2d = dX};
}
