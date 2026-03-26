#include "shapes.h"

#include "loss/cross_entropy.h"
#include "../memory.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <math.h>
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

static bool shouldUseCudaEventTiming(Context *ctx) {
  return shouldLogOpTiming() && isCudaContext(ctx);
}

static void destroyPendingCudaOpTiming(PendingCudaOpTiming *timing) {
  if (timing == NULL || !timing->active) {
    return;
  }

  cudaEventDestroy(timing->start);
  cudaEventDestroy(timing->end);
  timing->active = false;
}

static void drainPendingCudaOpTimings(void) {
  if (!shouldLogOpTiming()) {
    return;
  }

  for (int i = 0; i < MAX_PENDING_CUDA_OP_TIMINGS; i++) {
    PendingCudaOpTiming *timing = &pendingCudaOpTimings[i];
    if (!timing->active) {
      continue;
    }

    cudaError_t queryResult = cudaEventQuery(timing->end);
    if (queryResult == cudaErrorNotReady) {
      continue;
    }

    if (queryResult == cudaSuccess) {
      float elapsedMs = 0.0f;
      cudaEventElapsedTime(&elapsedMs, timing->start, timing->end);
      fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", timing->opName,
              timing->deviceName, timing->phase, (double)elapsedMs);
    } else {
      cudaGetLastError();
    }

    destroyPendingCudaOpTiming(timing);
  }
}

static Result startCudaOpPhase(Context *ctx, const char *opName, const char *phase,
                               CudaOpPhase *timing) {
  if (timing == NULL) {
    return ERR_NULL_PTR;
  }

  memset(timing, 0, sizeof(*timing));
  if (!shouldUseCudaEventTiming(ctx)) {
    return OK;
  }

  drainPendingCudaOpTimings();
  timing->opName = opName;
  timing->phase = phase;
  timing->deviceName = opTimingDeviceName(ctx);
  timing->active = true;

  if (cudaEventCreate(&timing->start) != cudaSuccess ||
      cudaEventCreate(&timing->end) != cudaSuccess) {
    destroyPendingCudaOpTiming(timing);
    return ERR_NO_OP;
  }
  if (cudaEventRecord(timing->start, 0) != cudaSuccess) {
    destroyPendingCudaOpTiming(timing);
    return ERR_NO_OP;
  }

  return OK;
}

static void abandonCudaOpPhase(CudaOpPhase *timing) {
  destroyPendingCudaOpTiming(timing);
}

static Result finishCudaOpPhase(CudaOpPhase *timing) {
  if (timing == NULL || !timing->active) {
    return OK;
  }

  if (cudaEventRecord(timing->end, 0) != cudaSuccess) {
    destroyPendingCudaOpTiming(timing);
    return ERR_NO_OP;
  }

  for (int i = 0; i < MAX_PENDING_CUDA_OP_TIMINGS; i++) {
    if (pendingCudaOpTimings[i].active) {
      continue;
    }

    pendingCudaOpTimings[i] = *timing;
    timing->active = false;
    return OK;
  }

  if (cudaEventSynchronize(timing->end) == cudaSuccess) {
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, timing->start, timing->end);
    fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", timing->opName,
            timing->deviceName, timing->phase, (double)elapsedMs);
  } else {
    cudaGetLastError();
  }
  destroyPendingCudaOpTiming(timing);
  return OK;
}

static void logHostOpTiming(Context *ctx, const char *opName, const char *phase, double startMs) {
  if (!shouldLogOpTiming()) {
    return;
  }

  fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", opName, opTimingDeviceName(ctx),
          phase, opTimingNowMs() - startMs);
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
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  CudaOpPhase cudaPhase = {0};
  if (shouldLogOpTiming()) {
    drainPendingCudaOpTimings();
    totalStartMs = opTimingNowMs();
    phaseStartMs = totalStartMs;
  }
  Result res = materializeTensorOnContext(ctx, yGround, true, &yArg);
  if (res != OK) {
    goto cleanup;
  }
  logHostOpTiming(ctx, "CrossEntropyForward", "materialize_y", phaseStartMs);
  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }
  res = materializeTensorOnContext(ctx, logits, true, &logitsArg);
  if (res != OK) {
    goto cleanup;
  }
  logHostOpTiming(ctx, "CrossEntropyForward", "materialize_logits", phaseStartMs);
  yContig = yArg.tensor;
  logitsContig = logitsArg.tensor;

  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }
  res = initTensorLike(ctx, probs, logitsContig, logitsContig->dtype);
  if (res != OK) {
    goto cleanup;
  }
  logHostOpTiming(ctx, "CrossEntropyForward", "init_probs", phaseStartMs);

  if (isCudaContext(ctx)) {
    Value zero = {.dtype = logitsContig->dtype};
    if (logitsContig->dtype == F64) {
      zero.as.f64 = 0.0;
    } else {
      zero.as.f32 = 0.0f;
    }

    if (shouldLogOpTiming()) {
      phaseStartMs = opTimingNowMs();
    }
    *loss = singleValueTensor(ctx, zero);
    logHostOpTiming(ctx, "CrossEntropyForward", "init_loss", phaseStartMs);
    if (loss->values == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    if (shouldUseCudaEventTiming(ctx)) {
      res = startCudaOpPhase(ctx, "CrossEntropyForward", "cuda_forward", &cudaPhase);
      if (res != OK) {
        goto cleanup;
      }
    } else if (shouldLogOpTiming()) {
      phaseStartMs = opTimingNowMs();
    }
    res =
        runCudaCrossEntropyForward(ctx, logitsContig->dtype, yContig->values, logitsContig->values,
                                   rows, classCount, probs->values, loss->values);
    if (res != OK) {
      abandonCudaOpPhase(&cudaPhase);
      goto cleanup;
    }
    if (shouldUseCudaEventTiming(ctx)) {
      res = finishCudaOpPhase(&cudaPhase);
      if (res != OK) {
        goto cleanup;
      }
    } else {
      logHostOpTiming(ctx, "CrossEntropyForward", "cuda_forward", phaseStartMs);
    }
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

    logHostOpTiming(ctx, "CrossEntropyForward", "cpu_forward", phaseStartMs);
    if (shouldLogOpTiming()) {
      phaseStartMs = opTimingNowMs();
    }
    *loss = singleValueTensor(ctx, VALUE(logitsContig->dtype, totalLoss / (float)rows));
  }
  logHostOpTiming(ctx, "CrossEntropyForward", "init_loss", phaseStartMs);

cleanup:
  if (shouldLogOpTiming()) {
    phaseStartMs = opTimingNowMs();
  }
  releaseTensorArg(ctx, &yArg);
  releaseTensorArg(ctx, &logitsArg);
  logHostOpTiming(ctx, "CrossEntropyForward", "cleanup", phaseStartMs);
  logHostOpTiming(ctx, "CrossEntropyForward", "total_host", totalStartMs);

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
