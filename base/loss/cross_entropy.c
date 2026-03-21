#include "shapes.h"

#include "loss/cross_entropy.h"
#include "../memory.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <math.h>

static bool isCudaContext(Context *ctx) {
  return ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA;
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

  TensorArg yArg = {0};
  TensorArg logitsArg = {0};
  Tensor *yContig = yGround;
  Tensor *logitsContig = logits;
  Result res = materializeTensorOnContext(ctx, yGround, true, &yArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, logits, true, &logitsArg);
  if (res != OK) {
    goto cleanup;
  }
  yContig = yArg.tensor;
  logitsContig = logitsArg.tensor;

  res = initTensorLike(ctx, probs, logitsContig, logitsContig->dtype);
  if (res != OK) {
    goto cleanup;
  }

  if (isCudaContext(ctx)) {
    Value zero = {.dtype = logitsContig->dtype};
    if (logitsContig->dtype == F64) {
      zero.as.f64 = 0.0;
    } else {
      zero.as.f32 = 0.0f;
    }

    *loss = singleValueTensor(ctx, zero);
    if (loss->values == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    res =
        runCudaCrossEntropyForward(ctx, logitsContig->dtype, yContig->values, logitsContig->values,
                                   rows, classCount, probs->values, loss->values);
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
  releaseTensorArg(ctx, &yArg);
  releaseTensorArg(ctx, &logitsArg);

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

  TensorArg yArg = {0};
  TensorArg pArg = {0};
  TensorArg gArg = {0};
  Tensor *yContig = yGround;
  Tensor *pContig = probs;
  Tensor *gContig = gradOut;
  Result res = materializeTensorOnContext(ctx, yGround, true, &yArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, probs, true, &pArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, gradOut, true, &gArg);
  if (res != OK) {
    goto cleanup;
  }
  yContig = yArg.tensor;
  pContig = pArg.tensor;
  gContig = gArg.tensor;

  res = initTensorLike(ctx, dLogits, pContig, pContig->dtype);
  if (res != OK) {
    goto cleanup;
  }

  if (isCudaContext(ctx)) {
    res = runCudaCrossEntropyBackward(ctx, pContig->dtype, yContig->values, pContig->values,
                                      gContig->values, rows, classCount, scalarGradOut,
                                      dLogits->values);
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
  releaseTensorArg(ctx, &yArg);
  releaseTensorArg(ctx, &pArg);
  releaseTensorArg(ctx, &gArg);

  return res;
}
