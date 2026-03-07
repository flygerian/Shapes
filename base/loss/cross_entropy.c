#include "shapes.h"

#include "../memory.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <math.h>

static Result initTensorLike(Context *ctx, Tensor *dest, Tensor *source) {
  u8 numDims = source->shape.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * numDims);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = source->shape.dims[i];
  }

  Dim shape = {.dims = dims, .numOfDims = numDims, .multipliers = multipliers};
  tensor_size_t size = calculateNumValuesAndMultipliers(shape, multipliers);

  *dest = (Tensor){.dtype = source->dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(source->dtype)),
                   .size = size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

Result CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits, Tensor *loss,
                           Tensor *probs) {
  if (isInvalidTensor(yGround) || isInvalidTensor(logits)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (yGround->shape.numOfDims != logits->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (yGround->dtype != logits->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (logits->dtype != F16 && logits->dtype != F32 && logits->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  for (u8 i = 0; i < logits->shape.numOfDims; i++) {
    if (yGround->shape.dims[i] != logits->shape.dims[i]) {
      return ERR_DIM_MISMATCH;
    }
  }

  if (logits->shape.numOfDims == 0) {
    return ERR_DIM_MISMATCH;
  }

  dim_t classCount = logits->shape.dims[logits->shape.numOfDims - 1];
  if (classCount == 0 || logits->size % classCount != 0) {
    return ERR_DIM_MISMATCH;
  }

  tensor_size_t rows = logits->size / classCount;

  Tensor *yContig = yGround;
  Tensor *logitsContig = logits;
  if (!yContig->isContigous) {
    yContig = copyToContiguous(ctx, yContig);
  }
  if (!logitsContig->isContigous) {
    logitsContig = copyToContiguous(ctx, logitsContig);
  }

  Result res = initTensorLike(ctx, probs, logitsContig);
  if (res != OK) {
    goto cleanup;
  }

  if (logitsContig->dtype == F64) {
    double *yVals = yContig->values;
    double *logitVals = logitsContig->values;
    double *probVals = probs->values;
    double totalLoss = 0.0;

    for (tensor_size_t r = 0; r < rows; r++) {
      tensor_size_t base = r * classCount;
      double rowMax = logitVals[base];
      for (tensor_size_t c = 1; c < classCount; c++) {
        double v = logitVals[base + c];
        if (v > rowMax) {
          rowMax = v;
        }
      }

      double sumExp = 0.0;
      for (tensor_size_t c = 0; c < classCount; c++) {
        double e = exp(logitVals[base + c] - rowMax);
        probVals[base + c] = e;
        sumExp += e;
      }

      double rowLoss = 0.0;
      for (tensor_size_t c = 0; c < classCount; c++) {
        double p = probVals[base + c] / sumExp;
        probVals[base + c] = p;
        rowLoss += yVals[base + c] * log(p);
      }

      totalLoss += -rowLoss;
    }

    *loss = singleValueTensor(ctx, VALUE(F64, totalLoss / (double)rows));
  } else {
    float *yVals = yContig->values;
    float *logitVals = logitsContig->values;
    float *probVals = probs->values;
    float totalLoss = 0.0f;

    for (tensor_size_t r = 0; r < rows; r++) {
      tensor_size_t base = r * classCount;
      float rowMax = logitVals[base];
      for (tensor_size_t c = 1; c < classCount; c++) {
        float v = logitVals[base + c];
        if (v > rowMax) {
          rowMax = v;
        }
      }

      float sumExp = 0.0f;
      for (tensor_size_t c = 0; c < classCount; c++) {
        float e = expf(logitVals[base + c] - rowMax);
        probVals[base + c] = e;
        sumExp += e;
      }

      float rowLoss = 0.0f;
      for (tensor_size_t c = 0; c < classCount; c++) {
        float p = probVals[base + c] / sumExp;
        probVals[base + c] = p;
        rowLoss += yVals[base + c] * logf(p);
      }

      totalLoss += -rowLoss;
    }

    *loss = singleValueTensor(ctx, VALUE(logitsContig->dtype, totalLoss / (float)rows));
  }

cleanup:
  if (yContig != yGround) {
    FreeTensor(ctx, yContig);
  }
  if (logitsContig != logits) {
    FreeTensor(ctx, logitsContig);
  }

  return res;
}

Result CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut,
                            Tensor *dLogits) {
  if (isInvalidTensor(yGround) || isInvalidTensor(probs) || isInvalidTensor(gradOut)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (yGround->shape.numOfDims != probs->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (yGround->dtype != probs->dtype || yGround->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (probs->dtype != F16 && probs->dtype != F32 && probs->dtype != F64) {
    return ERR_DTYPE_MISMATCH;
  }

  for (u8 i = 0; i < probs->shape.numOfDims; i++) {
    if (yGround->shape.dims[i] != probs->shape.dims[i]) {
      return ERR_DIM_MISMATCH;
    }
  }

  if (probs->shape.numOfDims == 0) {
    return ERR_DIM_MISMATCH;
  }

  dim_t classCount = probs->shape.dims[probs->shape.numOfDims - 1];
  if (classCount == 0 || probs->size % classCount != 0) {
    return ERR_DIM_MISMATCH;
  }

  bool scalarGradOut = gradOut->size == 1;
  if (!scalarGradOut && gradOut->size != probs->size) {
    return ERR_DIM_MISMATCH;
  }

  tensor_size_t rows = probs->size / classCount;

  Tensor *yContig = yGround;
  Tensor *pContig = probs;
  Tensor *gContig = gradOut;
  if (!yContig->isContigous) {
    yContig = copyToContiguous(ctx, yContig);
  }
  if (!pContig->isContigous) {
    pContig = copyToContiguous(ctx, pContig);
  }
  if (!gContig->isContigous) {
    gContig = copyToContiguous(ctx, gContig);
  }

  Result res = initTensorLike(ctx, dLogits, pContig);
  if (res != OK) {
    goto cleanup;
  }

  if (pContig->dtype == F64) {
    double *yVals = yContig->values;
    double *pVals = pContig->values;
    double *gVals = gContig->values;
    double *dVals = dLogits->values;
    double invRows = 1.0 / (double)rows;
    double scalar = scalarGradOut ? gVals[0] : 1.0;

    for (tensor_size_t i = 0; i < pContig->size; i++) {
      double localGrad = (pVals[i] - yVals[i]) * invRows;
      dVals[i] = localGrad * (scalarGradOut ? scalar : gVals[i]);
    }
  } else {
    float *yVals = yContig->values;
    float *pVals = pContig->values;
    float *gVals = gContig->values;
    float *dVals = dLogits->values;
    float invRows = 1.0f / (float)rows;
    float scalar = scalarGradOut ? gVals[0] : 1.0f;

    for (tensor_size_t i = 0; i < pContig->size; i++) {
      float localGrad = (pVals[i] - yVals[i]) * invRows;
      dVals[i] = localGrad * (scalarGradOut ? scalar : gVals[i]);
    }
  }

cleanup:
  if (yContig != yGround) {
    FreeTensor(ctx, yContig);
  }
  if (pContig != probs) {
    FreeTensor(ctx, pContig);
  }
  if (gContig != gradOut) {
    FreeTensor(ctx, gContig);
  }

  return res;
}
