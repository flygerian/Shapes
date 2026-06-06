#include "shapes.h"
#include "shapes_internal.h"
#include "types.h"
#include <math.h>
#include <sched.h>

static inline shapes_Tensor crossEnthropyFowardCpu(shapes_Context *ctx, shapes_Tensor *logitsContig, shapes_Tensor *yContig, shapes_Tensor *probs, shapes_tensor_size_t rows, shapes_dim_t classCount) {
  if (logitsContig->dtype == F64) {
    double *yVals = yContig->values;
    double *logitVals = logitsContig->values;
    double *probVals = probs->values;
    double totalLoss = 0.0;

    for (shapes_tensor_size_t r = 0; r < rows; r++) {
      shapes_tensor_size_t base = r * classCount;
      double rowMax = logitVals[base];
      for (shapes_tensor_size_t c = 1; c < classCount; c++) {
        double v = logitVals[base + c];
        if (v > rowMax) {
          rowMax = v;
        }
      }

      double sumExp = 0.0;
      for (shapes_tensor_size_t c = 0; c < classCount; c++) {
        double e = exp(logitVals[base + c] - rowMax);
        probVals[base + c] = e;
        sumExp += e;
      }

      double rowLoss = 0.0;
      for (shapes_tensor_size_t c = 0; c < classCount; c++) {
        double p = probVals[base + c] / sumExp;
        probVals[base + c] = p;
        if (yVals[base + c] != 0.0) {
          rowLoss += yVals[base + c] * log(p);
        }
      }

      totalLoss += -rowLoss;
    }

    return shapes_MakeFloat64Tensor(ctx, SCALAR, totalLoss / (double)rows);
  } else {
    float *yVals = yContig->values;
    float *logitVals = logitsContig->values;
    float *probVals = probs->values;
    float totalLoss = 0.0f;

    for (shapes_tensor_size_t r = 0; r < rows; r++) {
      shapes_tensor_size_t base = r * classCount;
      float rowMax = logitVals[base];
      for (shapes_tensor_size_t c = 1; c < classCount; c++) {
        float v = logitVals[base + c];
        if (v > rowMax) {
          rowMax = v;
        }
      }

      float sumExp = 0.0f;
      for (shapes_tensor_size_t c = 0; c < classCount; c++) {
        float e = expf(logitVals[base + c] - rowMax);
        probVals[base + c] = e;
        sumExp += e;
      }

      float rowLoss = 0.0f;
      for (shapes_tensor_size_t c = 0; c < classCount; c++) {
        float p = probVals[base + c] / sumExp;
        probVals[base + c] = p;
        if (yVals[base + c] != 0.0f) {
          rowLoss += yVals[base + c] * logf(p);
        }
      }

      totalLoss += -rowLoss;
    }

    return shapes_MakeFloatTensor(ctx, SCALAR, totalLoss / (double)rows);
  }
}

shapes_TensorPair shapes_loss_CrossEntropyForward(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *logits) {
  PANIC_IF(isInvalidTensor(yGround) || isInvalidTensor(logits), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yGround->shape.numOfDims != logits->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(yGround->dtype != logits->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(isNotFloatType(logits), ERR_DTYPE_MISMATCH);

  for (u8 i = 0; i < logits->shape.numOfDims; i++) {
    PANIC_IF(yGround->shape.dims[i] != logits->shape.dims[i], ERR_DIM_MISMATCH);
  }

  PANIC_IF(logits->shape.numOfDims == 0, ERR_DIM_MISMATCH);

  shapes_dim_t classCount = logits->shape.dims[logits->shape.numOfDims - 1];
  PANIC_IF(classCount == 0 || logits->size % classCount != 0, ERR_DIM_MISMATCH);

  shapes_tensor_size_t rows = logits->size / classCount;

  shapes_Tensor *yContig = materializeTensorOnContext(ctx, yGround);
  shapes_Tensor *logitsContig = materializeTensorOnContext(ctx, logits);

  shapes_Tensor probs = t_Zeros(ctx, logitsContig->shape, logitsContig->dtype);
  shapes_Tensor loss;
  Result res = OK;

  switch (ctx->device->type) {
    case CPU: loss = crossEnthropyFowardCpu(ctx, logitsContig, yContig, &probs, rows, classCount); break;
    case CUDA:
      loss = shapes_MakeZerosTensor(ctx, SCALAR);
      res = shapescuda_CrossEntropyForward(logitsContig->dtype, yContig->values, logitsContig->values, rows, classCount, probs.values, loss.values);
      PANIC_IF(res != OK, res);
      break;
  }

  return (shapes_TensorPair){.a = loss, .b = probs};
}

static inline void crossEnthropyBackwardCpu(shapes_Tensor *pContig, shapes_Tensor *yContig, shapes_Tensor *gContig, shapes_Tensor *dLogits, bool scalarGradOut, f32 rows) {
  if (pContig->dtype == F64) {
    double *yVals = yContig->values;
    double *pVals = pContig->values;
    double *gVals = gContig->values;
    double *dVals = dLogits->values;
    double invRows = 1.0 / (double)rows;
    double scalar = scalarGradOut ? gVals[0] : 1.0;

    for (shapes_tensor_size_t i = 0; i < pContig->size; i++) {
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

    for (shapes_tensor_size_t i = 0; i < pContig->size; i++) {
      float localGrad = (pVals[i] - yVals[i]) * invRows;
      dVals[i] = localGrad * (scalarGradOut ? scalar : gVals[i]);
    }
  }
}

shapes_Tensor shapes_loss_CrossEntropyBackward(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *probs, shapes_Tensor *gradOut) {
  PANIC_IF(isInvalidTensor(yGround) || isInvalidTensor(probs) || isInvalidTensor(gradOut), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yGround->shape.numOfDims != probs->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(yGround->dtype != probs->dtype || yGround->dtype != gradOut->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(probs->dtype != F16 && probs->dtype != F32 && probs->dtype != F64, ERR_DTYPE_MISMATCH);

  for (u8 i = 0; i < probs->shape.numOfDims; i++) {
    PANIC_IF(yGround->shape.dims[i] != probs->shape.dims[i], ERR_DIM_MISMATCH);
  }

  PANIC_IF(probs->shape.numOfDims == 0, ERR_DIM_MISMATCH);

  shapes_dim_t classCount = probs->shape.dims[probs->shape.numOfDims - 1];
  PANIC_IF(classCount == 0 || probs->size % classCount != 0, ERR_DIM_MISMATCH);

  bool scalarGradOut = gradOut->size == 1;
  PANIC_IF(!scalarGradOut && gradOut->size != probs->size, ERR_DIM_MISMATCH);

  shapes_tensor_size_t rows = probs->size / classCount;

  shapes_Tensor *yContig = materializeTensorOnContext(ctx, yGround);
  shapes_Tensor *pContig = materializeTensorOnContext(ctx, probs);
  shapes_Tensor *gContig = materializeTensorOnContext(ctx, gradOut);

  shapes_Tensor dLogits = t_Zeros(ctx, pContig->shape, pContig->dtype);

  switch (ctx->device->type) {
    case CPU: crossEnthropyBackwardCpu(pContig, yContig, gContig, &dLogits, scalarGradOut, rows); break;

    case CUDA: {
      Result res = shapescuda_CrossEntropyBackward(
          pContig->dtype,
          yContig->values,
          pContig->values,
          gContig->values,
          rows, classCount,
          scalarGradOut,
          dLogits.values
      );
      PANIC_IF(res != OK, res);
      break;
    }
  }

  return dLogits;
}
