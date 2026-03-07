#include "shapes.h"

#include "../memory.h"
#include "tensor/tensor_internal.h"

static Result init1DTensor(Context *ctx, Tensor *dest, dim_t size, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  u8 *multipliers = allocate(ctx->memory, sizeof(u8));

  dims[0] = size;
  multipliers[0] = 1;

  *dest = (Tensor){.dtype = dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = (Dim){.dims = dims, .numOfDims = 1, .multipliers = multipliers},
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

static Result init2DTensor(Context *ctx, Tensor *dest, dim_t rows, dim_t cols, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 2);
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * 2);

  dims[0] = rows;
  dims[1] = cols;

  Dim shape = {.dims = dims, .numOfDims = 2, .multipliers = multipliers};
  tensor_size_t size = calculateNumValuesAndMultipliers(shape, multipliers);

  *dest = (Tensor){.dtype = dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

Result BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon,
                                Tensor *out, Tensor *mean, Tensor *variance) {
  // Training forward expects flattened activations:
  // x2d: [m, n] where m=batch/items and n=features/channels.
  // gamma, beta: [n] affine parameters.
  // Outputs:
  // out: [m, n], mean: [n], variance: [n].
  if (isInvalidTensor(x2d) || isInvalidTensor(gamma) || isInvalidTensor(beta)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (x2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1 || beta->shape.numOfDims != 1) {
    return ERR_DIM_MISMATCH;
  }

  if (x2d->dtype != gamma->dtype || x2d->dtype != beta->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t m = x2d->shape.dims[0];
  dim_t n = x2d->shape.dims[1];
  if (gamma->shape.dims[0] != n || beta->shape.dims[0] != n) {
    return ERR_DIM_MISMATCH;
  }

  // Go may pass views/slices; normalize to contiguous buffers for tight loops.
  Tensor *xContig = x2d;
  Tensor *gammaContig = gamma;
  Tensor *betaContig = beta;

  if (!xContig->isContigous) {
    xContig = copyToContiguous(ctx, xContig);
  }
  if (!gammaContig->isContigous) {
    gammaContig = copyToContiguous(ctx, gammaContig);
  }
  if (!betaContig->isContigous) {
    betaContig = copyToContiguous(ctx, betaContig);
  }

  Result res = init2DTensor(ctx, out, m, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = init1DTensor(ctx, mean, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = init1DTensor(ctx, variance, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }

  if (x2d->dtype == F64) {
    double *xVals = xContig->values;
    double *gammaVals = gammaContig->values;
    double *betaVals = betaContig->values;
    double *outVals = out->values;
    double *meanVals = mean->values;
    double *varVals = variance->values;
    double *invStd = allocate(ctx->memory, sizeof(double) * n);

    // Initialize per-feature accumulators.
    for (tensor_size_t j = 0; j < n; j++) {
      meanVals[j] = 0.0;
      varVals[j] = 0.0;
    }

    // 1) mean over batch axis for each feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        meanVals[j] += xVals[row + j];
      }
    }

    double mAsDouble = (double)m;
    for (tensor_size_t j = 0; j < n; j++) {
      meanVals[j] /= mAsDouble;
    }

    // 2) variance over batch axis for each feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        double centered = xVals[row + j] - meanVals[j];
        varVals[j] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < n; j++) {
      varVals[j] /= mAsDouble;
      invStd[j] = 1.0 / sqrt(varVals[j] + (double)epsilon);
    }

    // 3) normalize, then apply learned scale (gamma) and shift (beta).
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        double xHat = (xVals[row + j] - meanVals[j]) * invStd[j];
        outVals[row + j] = xHat * gammaVals[j] + betaVals[j];
      }
    }

    freeAlloc(ctx->memory, invStd);
  } else {
    float *xVals = xContig->values;
    float *gammaVals = gammaContig->values;
    float *betaVals = betaContig->values;
    float *outVals = out->values;
    float *meanVals = mean->values;
    float *varVals = variance->values;
    float *invStd = allocate(ctx->memory, sizeof(float) * n);

    // Initialize per-feature accumulators.
    for (tensor_size_t j = 0; j < n; j++) {
      meanVals[j] = 0.0f;
      varVals[j] = 0.0f;
    }

    // 1) mean over batch axis for each feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        meanVals[j] += xVals[row + j];
      }
    }

    float mAsFloat = (float)m;
    for (tensor_size_t j = 0; j < n; j++) {
      meanVals[j] /= mAsFloat;
    }

    // 2) variance over batch axis for each feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        float centered = xVals[row + j] - meanVals[j];
        varVals[j] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < n; j++) {
      varVals[j] /= mAsFloat;
      invStd[j] = 1.0f / sqrtf(varVals[j] + epsilon);
    }

    // 3) normalize, then apply learned scale (gamma) and shift (beta).
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        float xHat = (xVals[row + j] - meanVals[j]) * invStd[j];
        outVals[row + j] = xHat * gammaVals[j] + betaVals[j];
      }
    }

    freeAlloc(ctx->memory, invStd);
  }

cleanup:
  if (xContig != x2d) {
    FreeTensor(ctx, xContig);
  }
  if (gammaContig != gamma) {
    FreeTensor(ctx, gammaContig);
  }
  if (betaContig != beta) {
    FreeTensor(ctx, betaContig);
  }
  return res;
}

Result BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon,
                         Tensor *dX, Tensor *dGamma, Tensor *dBeta) {
  // Backward uses the same flattened [m, n] layout:
  // grad2d is dL/dy, gamma is [n], outputs are dX [m, n], dGamma [n], dBeta [n].
  if (isInvalidTensor(x2d) || isInvalidTensor(grad2d) || isInvalidTensor(gamma)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (x2d->shape.numOfDims != 2 || grad2d->shape.numOfDims != 2 || gamma->shape.numOfDims != 1) {
    return ERR_DIM_MISMATCH;
  }

  if (x2d->dtype != grad2d->dtype || x2d->dtype != gamma->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x2d->dtype != F16 && x2d->dtype != F32 && x2d->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t m = x2d->shape.dims[0];
  dim_t n = x2d->shape.dims[1];

  if (grad2d->shape.dims[0] != m || grad2d->shape.dims[1] != n || gamma->shape.dims[0] != n) {
    return ERR_DIM_MISMATCH;
  }

  // Keep backward math on packed memory for predictable stride-1 access.
  Tensor *xContig = x2d;
  Tensor *gradContig = grad2d;
  Tensor *gammaContig = gamma;

  if (!xContig->isContigous) {
    xContig = copyToContiguous(ctx, xContig);
  }
  if (!gradContig->isContigous) {
    gradContig = copyToContiguous(ctx, gradContig);
  }
  if (!gammaContig->isContigous) {
    gammaContig = copyToContiguous(ctx, gammaContig);
  }

  Result res = init2DTensor(ctx, dX, m, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = init1DTensor(ctx, dGamma, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }
  res = init1DTensor(ctx, dBeta, n, x2d->dtype);
  if (res != OK) {
    goto cleanup;
  }

  if (x2d->dtype == F64) {
    double *xVals = xContig->values;
    double *dyVals = gradContig->values;
    double *gammaVals = gammaContig->values;

    double *dxVals = dX->values;
    double *dGammaVals = dGamma->values;
    double *dBetaVals = dBeta->values;

    double *mean = allocate(ctx->memory, sizeof(double) * n);
    double *var = allocate(ctx->memory, sizeof(double) * n);
    double *invStd = allocate(ctx->memory, sizeof(double) * n);
    double *sumDXHat = allocate(ctx->memory, sizeof(double) * n);
    double *sumDXHatXHat = allocate(ctx->memory, sizeof(double) * n);

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

    double mAsDouble = (double)m;
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] /= mAsDouble;
    }

    // 2) Recompute variance and inverse std per feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        double centered = xVals[row + j] - mean[j];
        var[j] += centered * centered;
      }
    }

    for (tensor_size_t j = 0; j < n; j++) {
      var[j] /= mAsDouble;
      invStd[j] = 1.0 / sqrt(var[j] + (double)epsilon);
    }

    // 3) Accumulate parameter grads and helper sums for dX.
    // dBeta = sum(dy), dGamma = sum(dy * xHat).
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        double centered = xVals[row + j] - mean[j];
        double xHat = centered * invStd[j];
        double dy = dyVals[row + j];
        double dxHat = dy * gammaVals[j];

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
        double centered = xVals[row + j] - mean[j];
        double xHat = centered * invStd[j];
        double dy = dyVals[row + j];
        double dxHat = dy * gammaVals[j];

        double numerator = mAsDouble * dxHat - sumDXHat[j] - xHat * sumDXHatXHat[j];
        dxVals[row + j] = (invStd[j] * numerator) / mAsDouble;
      }
    }

    freeAlloc(ctx->memory, sumDXHatXHat);
    freeAlloc(ctx->memory, sumDXHat);
    freeAlloc(ctx->memory, invStd);
    freeAlloc(ctx->memory, var);
    freeAlloc(ctx->memory, mean);
  } else {
    float *xVals = xContig->values;
    float *dyVals = gradContig->values;
    float *gammaVals = gammaContig->values;

    float *dxVals = dX->values;
    float *dGammaVals = dGamma->values;
    float *dBetaVals = dBeta->values;

    float *mean = allocate(ctx->memory, sizeof(float) * n);
    float *var = allocate(ctx->memory, sizeof(float) * n);
    float *invStd = allocate(ctx->memory, sizeof(float) * n);
    float *sumDXHat = allocate(ctx->memory, sizeof(float) * n);
    float *sumDXHatXHat = allocate(ctx->memory, sizeof(float) * n);

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

    float mAsFloat = (float)m;
    for (tensor_size_t j = 0; j < n; j++) {
      mean[j] /= mAsFloat;
    }

    // 2) Recompute variance and inverse std per feature.
    for (tensor_size_t i = 0; i < m; i++) {
      tensor_size_t row = i * n;
      for (tensor_size_t j = 0; j < n; j++) {
        float centered = xVals[row + j] - mean[j];
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
        float centered = xVals[row + j] - mean[j];
        float xHat = centered * invStd[j];
        float dy = dyVals[row + j];
        float dxHat = dy * gammaVals[j];

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
        float centered = xVals[row + j] - mean[j];
        float xHat = centered * invStd[j];
        float dy = dyVals[row + j];
        float dxHat = dy * gammaVals[j];

        float numerator = mAsFloat * dxHat - sumDXHat[j] - xHat * sumDXHatXHat[j];
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
  if (xContig != x2d) {
    FreeTensor(ctx, xContig);
  }
  if (gradContig != grad2d) {
    FreeTensor(ctx, gradContig);
  }
  if (gammaContig != gamma) {
    FreeTensor(ctx, gammaContig);
  }

  return res;
}
