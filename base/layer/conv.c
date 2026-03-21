#include "im2col.h"
#include "common.h"
#include "cblas.h"
#include "memory.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/tensor_internal.h"
#include <assert.h>
#include <iso646.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int clampBlasThreadCount(int threadCount) {
  int maxThreads = openblas_get_num_procs();
  if (maxThreads < 1) {
    maxThreads = 1;
  }

  if (threadCount < 1) {
    return 0;
  }

  return threadCount > maxThreads ? maxThreads : threadCount;
}

static bool shouldLogOpTiming(void) {
  const char *value = getenv("SHAPES_LOG_OP_TIMES");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static double opTimingNowMs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static Result syncForOpTiming(Context *ctx) {
  if (!shouldLogOpTiming()) {
    return OK;
  }

  return Flush(ctx);
}

static void logOpTiming(Context *ctx, const char *opName, const char *phase, double startMs) {
  if (!shouldLogOpTiming()) {
    return;
  }

  const char *deviceName = "CPU(default)";
  if (ctx != NULL && ctx->device != NULL) {
    switch (ctx->device->type) {
      case CPU: deviceName = "CPU"; break;
      case CUDA: deviceName = "CUDA"; break;
      default: deviceName = "UNKNOWN"; break;
    }
  }

  fprintf(stderr, "[opTiming] op=%s device=%s phase=%s ms=%.3f\n", opName, deviceName, phase,
          opTimingNowMs() - startMs);
}

Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Tensor *t, Tensor *dest, Tensor *colBufferDest) {
  TensorArg inputArg = {0};
  TensorArg kernelArg = {0};
  Tensor *inputContig = t;
  Tensor *kernelContig = kernels;
  Tensor gemmOutput;
  Tensor permutedOutput;
  bool gemmOutputInitialized = false;
  bool permutedOutputInitialized = false;
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;

  Result result;

  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }

  if (t == NULL || dest == NULL || ctx == NULL) {
    result = ERR_NULL_TENSOR_PROVIDED;
    goto cleanup;
  }

  if (stride == 0) {
    result = ERR_CONV2D_KERNEL_STRIDE_ZERO;
    goto cleanup;
  }

  if (isNotFloatType(t)) {
    result = ERR_CONV2D_KERNEL_NOT_FLOAT;
    goto cleanup;
  }

  if (t->shape.numOfDims < 4) {
    result = ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
    goto cleanup;
  }

  if (inChannels < 1) {
    result = ERR_CONV2D_IN_CHANNELS_ZERO;
    goto cleanup;
  }

  if (outChannels < 1) {
    result = ERR_CONV2D_OUT_CHANNELS_ZERO;
    goto cleanup;
  }

  if (kernels == NULL) {
    result = ERR_NULL_TENSOR_PROVIDED;
    goto cleanup;
  }

  if (kernels->shape.numOfDims < 2 || kernels->shape.dims == NULL) {
    result = ERR_CONV2D_KERNEL_NOT_2D;
    goto cleanup;
  }

  if (kernels->shape.numOfDims < 4) {
    result = ERR_DIM_MISMATCH;
    goto cleanup;
  }

  if (kernels->dtype != t->dtype) {
    result = ERR_DTYPE_MISMATCH;
    goto cleanup;
  }

  dim_t kernelHeight = kernels->shape.dims[2];
  dim_t kernelWidth = kernels->shape.dims[3];
  dim_t batch = t->shape.dims[0];
  dim_t height = t->shape.dims[1];
  dim_t width = t->shape.dims[2];
  dim_t numChannels = t->shape.dims[3];

  if (kernelHeight == 0 || kernelWidth == 0 || numChannels != inChannels || height < kernelHeight ||
      width < kernelWidth) {
    result = ERR_DIM_MISMATCH;
    goto cleanup;
  }

  if (kernels->shape.dims[0] != outChannels || kernels->shape.dims[1] != inChannels ||
      kernels->shape.dims[2] != kernelHeight || kernels->shape.dims[3] != kernelWidth) {
    result = ERR_DIM_MISMATCH;
    goto cleanup;
  }

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  result = syncForOpTiming(ctx);
  if (result != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  result = init4DTensor(ctx, &gemmOutput, batch, outputChannelHeight, outputChannelWidth,
                        outChannels, t->dtype);
  if (result != OK) {
    goto cleanup;
  }
  logOpTiming(ctx, "Conv2d", "alloc_gemm_output", phaseStartMs);

  gemmOutputInitialized = true;
  result = syncForOpTiming(ctx);
  if (result != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  result = materializeTensorOnContext(ctx, t, true, &inputArg);
  if (result != OK) {
    goto cleanup;
  }
  result = materializeTensorOnContext(ctx, kernels, true, &kernelArg);
  if (result != OK) {
    goto cleanup;
  }
  inputContig = inputArg.tensor;
  kernelContig = kernelArg.tensor;
  logOpTiming(ctx, "Conv2d", "materialize", phaseStartMs);

  tensor_size_t patchSize = inChannels * kernelHeight * kernelWidth;
  tensor_size_t positions = outputChannelHeight * outputChannelWidth;

  Tensor *colBuffer;
  tensor_size_t colBufferSize;

  if (t->dtype == F64) {
    f64 *kernelValues = kernelContig->values;

    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    colBuffer = im2colF64(ctx, inputContig, kernelHeight, kernelWidth, stride);
    if (colBuffer == NULL) {
      result = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2d", "im2col", phaseStartMs);

    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels,
            (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);
    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2d", "gemm", phaseStartMs);
  } else {
    f32 *kernelValues = kernelContig->values;

    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    colBuffer = im2colF32(ctx, inputContig, kernelHeight, kernelWidth, stride);
    if (colBuffer == NULL) {
      result = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2d", "im2col", phaseStartMs);

    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels,
            (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);
    result = syncForOpTiming(ctx);
    if (result != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2d", "gemm", phaseStartMs);
  }

  if (colBufferDest != NULL) {
    *colBufferDest = *colBuffer;
    freeAlloc(ctx->memory, colBuffer);
  }

  *dest = gemmOutput;
  gemmOutputInitialized = false;

  result = OK;
  logOpTiming(ctx, "Conv2d", "total", totalStartMs);

cleanup:
  releaseTensorArg(ctx, &inputArg);
  releaseTensorArg(ctx, &kernelArg);
  if (permutedOutputInitialized) {
    freeTensorBuffers(ctx, &permutedOutput);
  }
  if (gemmOutputInitialized) {
    freeTensorBuffers(ctx, &gemmOutput);
  }

  return result;
}

Result Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels,
                      Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer, u8 stride) {
  TensorArg inputArg = {0};
  TensorArg kernelArg = {0};
  TensorArg outputGradArg = {0};
  TensorArg colBufferArg = {0};
  TensorArg dInputArg = {0};
  TensorArg dKernelArg = {0};
  Tensor *inputContig = input;
  Tensor *kernelContig = kernels;
  Tensor *outputGradContig = outputGrad;
  Tensor *colBufferContig = colBuffer;
  Tensor *dInputWork = dInput;
  Tensor *dKernelWork = dKernels;
  void *dColBuffer = NULL;
  bool outGradNhwcInitialized = false;
  bool outputGradPermutedCopied = false;
  Tensor outGradNhwc = {0};
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  Result res = OK;

  if (shouldLogOpTiming()) {
    totalStartMs = opTimingNowMs();
  }

  if (ctx == NULL || isInvalidTensor(input) || isInvalidTensor(kernels) || isInvalidTensor(outputGrad) ||
      dInput == NULL || dKernels == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (colBuffer == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(input) || isNotFloatType(kernels) || isNotFloatType(outputGrad)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (input->shape.numOfDims != 4 || kernels->shape.numOfDims != 4 ||
      outputGrad->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (input->dtype != kernels->dtype || input->dtype != outputGrad->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t batch = input->shape.dims[0];
  dim_t h = input->shape.dims[1];
  dim_t w = input->shape.dims[2];
  dim_t inChannels = input->shape.dims[3];
  dim_t outChannels = kernels->shape.dims[0];
  dim_t kernelInChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  if (outputGrad->shape.dims[0] != batch || outputGrad->shape.dims[1] != outH ||
      outputGrad->shape.dims[2] != outW || outputGrad->shape.dims[3] != outChannels) {
    return ERR_DIM_MISMATCH;
  }

  res = syncForOpTiming(ctx);
  if (res != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  res = materializeTensorOnContext(ctx, input, true, &inputArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, kernels, true, &kernelArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, outputGrad, true, &outputGradArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, colBuffer, true, &colBufferArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, dInput, true, &dInputArg);
  if (res != OK) {
    goto cleanup;
  }
  res = materializeTensorOnContext(ctx, dKernels, true, &dKernelArg);
  if (res != OK) {
    goto cleanup;
  }
  inputContig = inputArg.tensor;
  kernelContig = kernelArg.tensor;
  outputGradContig = outputGradArg.tensor;
  colBufferContig = colBufferArg.tensor;
  dInputWork = dInputArg.tensor;
  dKernelWork = dKernelArg.tensor;
  logOpTiming(ctx, "Conv2dBackward", "materialize", phaseStartMs);

  res = syncForOpTiming(ctx);
  if (res != OK) {
    goto cleanup;
  }
  phaseStartMs = opTimingNowMs();
  res = clearTensorValues(dInputWork);
  if (res != OK) {
    goto cleanup;
  }
  res = clearTensorValues(dKernelWork);
  if (res != OK) {
    goto cleanup;
  }
  logOpTiming(ctx, "Conv2dBackward", "clear_grads", phaseStartMs);

  dim_t C_in = input->shape.dims[3];
  dim_t kS = C_in * kH * kW;

  if (input->dtype == F64) {
    f64 *wValues = kernelContig->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f64 *dWValues = dKernelWork->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f64 *dOutput = outputGradContig->values; // (B,outH, outW, C_out)

    dim_t batch = input->shape.dims[0];
    dim_t outH = outputGradContig->shape.dims[1];
    dim_t outW = outputGradContig->shape.dims[2];
    dim_t C_out = outputGradContig->shape.dims[3];

    dim_t outputPositions = batch * outH * outW;

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    dColBuffer = allocateOnCtx(ctx, (outputPositions * kS) * sizeof(f64));
    if (dColBuffer == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "alloc_dcol_buffer", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, F64, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out,
            colBufferContig->values, kS, false, dWValues, kS);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "gemm_dk", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, F64, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out,
            wValues, kS, false, dColBuffer, kS);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "gemm_dcol", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    res = col2imAccumulateF64(dInputWork, dColBuffer, kH, kW, stride);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "col2im", phaseStartMs);
  } else {
    f32 *wValues = kernelContig->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f32 *dWValues = dKernelWork->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f32 *dOutput = outputGradContig->values; // (B,outH, outW, C_out)

    dim_t batch = input->shape.dims[0];
    dim_t outH = outputGradContig->shape.dims[1];
    dim_t outW = outputGradContig->shape.dims[2];
    dim_t C_out = outputGradContig->shape.dims[3];

    dim_t outputPositions = batch * outH * outW;

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    dColBuffer = allocateOnCtx(ctx, (outputPositions * kS) * sizeof(f32));
    if (dColBuffer == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "alloc_dcol_buffer", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, F32, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out,
            colBufferContig->values, kS, false, dWValues, kS);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "gemm_dk", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    runGemm(ctx, F32, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out,
            wValues, kS, false, dColBuffer, kS);
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "gemm_dcol", phaseStartMs);

    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    res = col2imAccumulateF32(dInputWork, dColBuffer, kH, kW, stride);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "col2im", phaseStartMs);
  }

  if (dInputWork != dInput) {
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    size_t dInputBytes = dInputWork->size * getBytesForDtype(dInputWork->dtype);
    res = copyBetweenContexts(ctx, dInput->context, dInputWork->values, dInput->values, dInputBytes);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "copy_dx_back", phaseStartMs);
  }
  if (dKernelWork != dKernels) {
    res = syncForOpTiming(ctx);
    if (res != OK) {
      goto cleanup;
    }
    phaseStartMs = opTimingNowMs();
    size_t dKernelBytes = dKernelWork->size * getBytesForDtype(dKernelWork->dtype);
    res = copyBetweenContexts(ctx, dKernels->context, dKernelWork->values, dKernels->values,
                              dKernelBytes);
    if (res != OK) {
      goto cleanup;
    }
    logOpTiming(ctx, "Conv2dBackward", "copy_dk_back", phaseStartMs);
  }

  res = OK;
  logOpTiming(ctx, "Conv2dBackward", "total", totalStartMs);

cleanup:
  releaseTensorArg(ctx, &inputArg);
  releaseTensorArg(ctx, &kernelArg);
  releaseTensorArg(ctx, &outputGradArg);
  releaseTensorArg(ctx, &colBufferArg);
  releaseTensorArg(ctx, &dInputArg);
  releaseTensorArg(ctx, &dKernelArg);

  if (outputGradPermutedCopied && outputGradContig != NULL) {
    FreeTensor(ctx, outputGradContig);
  }
  if (outGradNhwcInitialized) {
    freeTensorBuffers(ctx, &outGradNhwc);
  }

  if (dColBuffer != NULL) {
    freeOnCtx(ctx, dColBuffer);
  }

  return res;
}

Result ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride,
                       Tensor *kernels, Dim kernelShape, Tensor *t, Tensor *dest) {
  if (t == NULL || dest == NULL || ctx == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(t)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (t->shape.numOfDims < 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (inChannels < 1) {
    return ERR_CONV2D_IN_CHANNELS_ZERO;
  }

  if (outChannels < 1) {
    return ERR_CONV2D_OUT_CHANNELS_ZERO;
  }

  if (kernelShape.numOfDims < 2 || kernelShape.dims == NULL) {
    return ERR_CONV2D_KERNEL_NOT_2D;
  }

  if (kernels == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  dim_t kH = kernelShape.dims[0];
  dim_t kW = kernelShape.dims[1];
  dim_t batch = t->shape.dims[0];
  dim_t h = t->shape.dims[1];
  dim_t w = t->shape.dims[2];
  dim_t c = t->shape.dims[3];

  if (kH == 0 || kW == 0 || c != inChannels) {
    return ERR_DIM_MISMATCH;
  }

  if (kernels->shape.numOfDims < 4) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->shape.dims[0] != inChannels || kernels->shape.dims[1] != outChannels ||
      kernels->shape.dims[2] != kH || kernels->shape.dims[3] != kW) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->dtype != t->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  Result res = init4DTensor(ctx, dest, batch, outH, outW, outChannels, t->dtype);
  if (res != OK) {
    return res;
  }

  memset(dest->values, 0, dest->size * getBytesForDtype(dest->dtype));

  if (t->dtype == F64) {
    f64 *input = t->values;
    f64 *kernelValues = kernels->values;
    f64 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * h + ih) * w + iw) * inChannels) + ic;
            f64 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx =
                      (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  outValues[outIdx] += inputValue * kernelValues[kernelIdx];
                }
              }
            }
          }
        }
      }
    }
  } else {
    f32 *input = t->values;
    f32 *kernelValues = kernels->values;
    f32 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * h + ih) * w + iw) * inChannels) + ic;
            f32 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx =
                      (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  outValues[outIdx] += inputValue * kernelValues[kernelIdx];
                }
              }
            }
          }
        }
      }
    }
  }

  return OK;
}

Result ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride,
                               Tensor *dX, Tensor *dKernels) {
  if (isInvalidTensor(x) || isInvalidTensor(kernels) || isInvalidTensor(gradOut) || dX == NULL ||
      dKernels == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(x) || isNotFloatType(kernels) || isNotFloatType(gradOut)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4 || kernels->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (x->dtype != kernels->dtype || x->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t h = x->shape.dims[1];
  dim_t w = x->shape.dims[2];
  dim_t inChannels = x->shape.dims[3];
  dim_t kernelInChannels = kernels->shape.dims[0];
  dim_t outChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outH ||
      gradOut->shape.dims[2] != outW || gradOut->shape.dims[3] != outChannels) {
    return ERR_DIM_MISMATCH;
  }

  Result res = initTensorLike(ctx, dX, x, x->dtype);
  if (res != OK) {
    return res;
  }
  res = initTensorLike(ctx, dKernels, kernels, kernels->dtype);
  if (res != OK) {
    return res;
  }

  memset(dX->values, 0, dX->size * getBytesForDtype(dX->dtype));
  memset(dKernels->values, 0, dKernels->size * getBytesForDtype(dKernels->dtype));

  if (x->dtype == F64) {
    f64 *xValues = x->values;
    f64 *kernelValues = kernels->values;
    f64 *gradValues = gradOut->values;
    f64 *dxValues = dX->values;
    f64 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * h + ih) * w + iw) * inChannels) + ic;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  f64 grad = gradValues[gradIdx];
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  } else {
    f32 *xValues = x->values;
    f32 *kernelValues = kernels->values;
    f32 *gradValues = gradOut->values;
    f32 *dxValues = dX->values;
    f32 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * h + ih) * w + iw) * inChannels) + ic;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  f32 grad = gradValues[gradIdx];
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  }

  return OK;
}
