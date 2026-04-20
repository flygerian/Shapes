#include "common.h"
#include "result/result.h"
#include "shapes.h"

#include "loss/cross_entropy.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool isCudaContext(Context *ctx) {
  return ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA;
}

static bool shouldLogOpTiming(void) {
  const char *value = getenv("SHAPES_LOG_OP_TIMES");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static const char *opTimingDeviceName(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return "CPU(default)";
  }

  switch (ctx->device->type) {
    case CPU: return "CPU";
    case CUDA: return "CUDA";
    default: return "UNKNOWN";
  }
}

static double opTimingNowMs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

typedef struct {
  cudaEvent_t start;
  cudaEvent_t end;
  const char *opName;
  const char *phase;
  const char *deviceName;
  bool active;
} PendingCudaOpTiming;

typedef PendingCudaOpTiming CudaOpPhase;

#define MAX_PENDING_CUDA_OP_TIMINGS 32

static PendingCudaOpTiming pendingCudaOpTimings[MAX_PENDING_CUDA_OP_TIMINGS] = {0};

static void logHostOpTiming(Context *ctx, const char *opName, const char *phase, double startMs) {
  if (!shouldLogOpTiming()) {
    return;
  }

  fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", opName, opTimingDeviceName(ctx),
          phase, opTimingNowMs() - startMs);
}

TensorPair CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits) {
  PANIC_IF(isInvalidTensor(yGround) || isInvalidTensor(logits), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yGround->shape.numOfDims != logits->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(yGround->dtype != logits->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(logits->dtype != F16 && logits->dtype != F32 && logits->dtype != F64,
           ERR_DTYPE_MISMATCH);

  for (u8 i = 0; i < logits->shape.numOfDims; i++) {
    PANIC_IF(yGround->shape.dims[i] != logits->shape.dims[i], ERR_DIM_MISMATCH);
  }

  PANIC_IF(logits->shape.numOfDims == 0, ERR_DIM_MISMATCH);

  dim_t classCount = logits->shape.dims[logits->shape.numOfDims - 1];
  PANIC_IF(classCount == 0 || logits->size % classCount != 0, ERR_DIM_MISMATCH);

  tensor_size_t rows = logits->size / classCount;

  Tensor *yContig = materializeTensorOnContext(ctx, yGround);
  Tensor *logitsContig = materializeTensorOnContext(ctx, logits);

  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  CudaOpPhase cudaPhase = {0};
  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
    phaseStartMs = totalStartMs;
  }

  logHostOpTiming(ctx, "CrossEntropyForward", "materialize_y", phaseStartMs);
  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }
  logHostOpTiming(ctx, "CrossEntropyForward", "materialize_logits", phaseStartMs);

  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }

  Tensor *probs = t_Zeros(ctx, logitsContig->shape, logitsContig->dtype);
  Tensor *loss;
  Result res = OK;
  logHostOpTiming(ctx, "CrossEntropyForward", "init_probs", phaseStartMs);

  if (isCudaContext(ctx)) {
    loss = T_Zeros(ctx, SCALAR);
    res =
        runCudaCrossEntropyForward(ctx, logitsContig->dtype, yContig->values, logitsContig->values,
                                   rows, classCount, probs->values, loss->values);
    PANIC_IF(res != OK, res);

    logHostOpTiming(ctx, "CrossEntropyForward", "cuda_forward", phaseStartMs);
    goto cleanup;
  }

  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
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

    logHostOpTiming(ctx, "CrossEntropyForward", "cpu_forward", phaseStartMs);
    if (shouldLogOpTiming()) {
      phaseStartMs = opTimingNowMs();
    }

    loss = T_Float64(ctx, SCALAR, totalLoss / (double)rows);
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

    logHostOpTiming(ctx, "CrossEntropyForward", "cpu_forward", phaseStartMs);
    if (shouldLogOpTiming()) {
      phaseStartMs = opTimingNowMs();
    }

    loss = T_Float(ctx, SCALAR, totalLoss / (float)rows);
  }

  logHostOpTiming(ctx, "CrossEntropyForward", "init_loss", phaseStartMs);

cleanup:
  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }

  freeIfContingousCopy(ctx, yContig);
  freeIfContingousCopy(ctx, logitsContig);

  logHostOpTiming(ctx, "CrossEntropyForward", "cleanup", phaseStartMs);
  logHostOpTiming(ctx, "CrossEntropyForward", "total_host", totalStartMs);

  return (TensorPair){.a = loss, .b = probs};
}

Tensor *CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut) {
  PANIC_IF(isInvalidTensor(yGround) || isInvalidTensor(probs) || isInvalidTensor(gradOut),
           ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(yGround->shape.numOfDims != probs->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(yGround->dtype != probs->dtype || yGround->dtype != gradOut->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(probs->dtype != F16 && probs->dtype != F32 && probs->dtype != F64, ERR_DTYPE_MISMATCH);

  for (u8 i = 0; i < probs->shape.numOfDims; i++) {
    PANIC_IF(yGround->shape.dims[i] != probs->shape.dims[i], ERR_DIM_MISMATCH);
  }

  PANIC_IF(probs->shape.numOfDims == 0, ERR_DIM_MISMATCH);

  dim_t classCount = probs->shape.dims[probs->shape.numOfDims - 1];
  PANIC_IF(classCount == 0 || probs->size % classCount != 0, ERR_DIM_MISMATCH);

  bool scalarGradOut = gradOut->size == 1;
  PANIC_IF(!scalarGradOut && gradOut->size != probs->size, ERR_DIM_MISMATCH);

  tensor_size_t rows = probs->size / classCount;

  Tensor *yContig = materializeTensorOnContext(ctx, yGround);
  Tensor *pContig = materializeTensorOnContext(ctx, probs);
  Tensor *gContig = materializeTensorOnContext(ctx, gradOut);

  Tensor *dLogits = t_Zeros(ctx, pContig->shape, pContig->dtype);

  if (isCudaContext(ctx)) {
    Result res = runCudaCrossEntropyBackward(ctx, pContig->dtype, yContig->values, pContig->values,
                                             gContig->values, rows, classCount, scalarGradOut,
                                             dLogits->values);
    PANIC_IF(res != OK, res);
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
  freeIfContingousCopy(ctx, yContig);
  freeIfContingousCopy(ctx, pContig);
  freeIfContingousCopy(ctx, gContig);

  return dLogits;
}
