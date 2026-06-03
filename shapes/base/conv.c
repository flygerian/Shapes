#include "result.h"
#include "shapes.h"
#include "shapes_internal.h"
#include <assert.h>
#include <iso646.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static Result addConvBiasCpu(Tensor *output, Tensor *bias) {
  if (output == NULL || bias == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  dim_t channels = output->shape.dims[3];
  tensor_size_t numValues = output->size;

  if (output->dtype == F64) {
    f64 *outValues = output->values;
    f64 *biasValues = bias->values;
    for (tensor_size_t i = 0; i < numValues; i++) {
      outValues[i] += biasValues[i % channels];
    }
    return OK;
  }

  f32 *outValues = output->values;
  f32 *biasValues = bias->values;
  for (tensor_size_t i = 0; i < numValues; i++) {
    outValues[i] += biasValues[i % channels];
  }
  return OK;
}

static Result addConvBias(Context *ctx, Tensor *output, Tensor *bias) {
  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    return runCudaConvBiasAdd(output->dtype, output->values, bias->values, output->size, output->shape.dims[3]);
  }

  return addConvBiasCpu(output, bias);
}

static void accumulateConvBiasGradCuda(Context *ctx, Tensor *outputGrad, Tensor *dBias) {
  PANIC_IF(ctx == NULL || outputGrad == NULL || dBias == NULL, ERR_NULL_TENSOR_PROVIDED);

  dim_t channels = outputGrad->shape.dims[3];
  tensor_size_t rows = outputGrad->size / channels;
  size_t oneBytes = rows * getBytesForDtype(outputGrad->dtype);
  double phaseStartMs = 0.0;
  void *ones = allocate(ctx->memory, oneBytes);

  PANIC_IF(ones == NULL, ERR_OUT_OF_MEMORY);

  Value one = {.dtype = outputGrad->dtype};
  if (outputGrad->dtype == F64) {
    one.as.f64 = 1.0;
  } else {
    one.as.f32 = 1.0f;
  }

  Result res = runCudaFillTensor(outputGrad->dtype, ones, rows, one);
  PANIC_IF(res != OK, res);

  runGemm(ctx, outputGrad->dtype, CblasNoTrans, CblasNoTrans, 1, (int)channels, (int)rows, ones, (int)rows, outputGrad->values, (int)channels, false, dBias->values, (int)channels);
}

static void accumulateConvBiasGradCpu(Tensor *outputGrad, Tensor *dBias) {
  PANIC_IF(outputGrad == NULL || dBias == NULL, ERR_NULL_TENSOR_PROVIDED);

  dim_t channels = outputGrad->shape.dims[3];
  PANIC_IF(channels == 0, ERR_DTYPE_MISMATCH);

  tensor_size_t numValues = outputGrad->size;

  if (outputGrad->dtype == F64) {
    f64 *gradValues = outputGrad->values;
    f64 *biasGradValues = dBias->values;
    for (dim_t c = 0; c < channels; c++) {
      biasGradValues[c] = 0.0;
    }
    for (tensor_size_t i = 0; i < numValues; i++) {
      biasGradValues[i % channels] += gradValues[i];
    }
    return;
  }

  f32 *gradValues = outputGrad->values;
  f32 *biasGradValues = dBias->values;
  for (dim_t c = 0; c < channels; c++) {
    biasGradValues[c] = 0.0f;
  }
  for (tensor_size_t i = 0; i < numValues; i++) {
    biasGradValues[i % channels] += gradValues[i];
  }
}

static void accumulateConvBiasGrad(Context *ctx, Tensor *outputGrad, Tensor *dBias) {
  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    return accumulateConvBiasGradCuda(ctx, outputGrad, dBias);
  }

  return accumulateConvBiasGradCpu(outputGrad, dBias);
}

Result shapes_layer_Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Tensor *bias, bool withBias, Tensor *t, Tensor *dest, Tensor *colBufferDest) {
  Tensor *inputContig = t;
  Tensor *kernelContig = kernels;
  Tensor *biasContig = bias;
  Tensor gemmOutput;
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;

  Result result;

  PANIC_IF(t == NULL || dest == NULL || ctx == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(stride == 0, ERR_CONV2D_KERNEL_STRIDE_ZERO);
  PANIC_IF(isNotFloatType(t), ERR_CONV2D_KERNEL_NOT_FLOAT);

  PANIC_IF(t->shape.numOfDims < 4, ERR_CONV2D_INVALID_NUM_TENSOR_DIM);
  PANIC_IF(inChannels < 1, ERR_CONV2D_IN_CHANNELS_ZERO);

  PANIC_IF(outChannels < 1, ERR_CONV2D_OUT_CHANNELS_ZERO);
  PANIC_IF(kernels == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(kernels->shape.numOfDims < 2 || kernels->shape.dims == NULL, ERR_CONV2D_KERNEL_NOT_2D);

  PANIC_IF(kernels->shape.numOfDims < 4, ERR_DIM_MISMATCH);

  PANIC_IF(kernels->dtype != t->dtype, ERR_DTYPE_MISMATCH);
  if (withBias) {
    PANIC_IF(bias == NULL, ERR_NULL_TENSOR_PROVIDED);
    PANIC_IF(bias->dtype != t->dtype, ERR_DTYPE_MISMATCH);
    PANIC_IF(bias->shape.numOfDims != 4 || bias->shape.dims == NULL, ERR_DIM_MISMATCH);
    PANIC_IF(bias->shape.dims[0] != 1 || bias->shape.dims[1] != 1 || bias->shape.dims[2] != 1 || bias->shape.dims[3] != outChannels, ERR_DIM_MISMATCH);
  }

  dim_t kernelHeight = kernels->shape.dims[2];
  dim_t kernelWidth = kernels->shape.dims[3];
  dim_t batch = t->shape.dims[0];
  dim_t height = t->shape.dims[1];
  dim_t width = t->shape.dims[2];
  dim_t numChannels = t->shape.dims[3];

  PANIC_IF(kernelHeight == 0 || kernelWidth == 0 || numChannels != inChannels || height < kernelHeight || width < kernelWidth, ERR_DIM_MISMATCH);

  PANIC_IF(kernels->shape.dims[0] != outChannels || kernels->shape.dims[1] != inChannels || kernels->shape.dims[2] != kernelHeight || kernels->shape.dims[3] != kernelWidth, ERR_DIM_MISMATCH);

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  Tensor *createdGemmOutput = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdGemmOutput == NULL, ALLOCATION_FAILED);
  *createdGemmOutput = t_Zeros(ctx, SHAPE4D(batch, outputChannelHeight, outputChannelWidth, outChannels), t->dtype);
  PANIC_IF(createdGemmOutput == NULL, ERR_OUT_OF_MEMORY);
  gemmOutput = *createdGemmOutput;

  inputContig = materializeTensorOnContext(ctx, t);
  kernelContig = materializeTensorOnContext(ctx, kernels);
  if (withBias) {
    biasContig = materializeTensorOnContext(ctx, bias);
  }

  tensor_size_t patchSize = inChannels * kernelHeight * kernelWidth;
  tensor_size_t positions = outputChannelHeight * outputChannelWidth;

  Tensor *colBuffer;
  tensor_size_t colBufferSize;

  if (t->dtype == F64) {
    f64 *kernelValues = kernelContig->values;

    colBuffer = im2colF64(ctx, inputContig, kernelHeight, kernelWidth, stride);
    PANIC_IF(colBuffer == NULL, ERR_OUT_OF_MEMORY);

    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels, (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);

  } else {
    f32 *kernelValues = kernelContig->values;

    colBuffer = im2colF32(ctx, inputContig, kernelHeight, kernelWidth, stride);

    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels, (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);
  }

  if (colBufferDest != NULL) {
    *colBufferDest = *colBuffer;
  }

  if (withBias) {
    result = addConvBias(ctx, &gemmOutput, biasContig);
    PANIC_IF(result != OK, result);
  }

  *dest = gemmOutput;

  result = OK;

  return result;
}

Result shapes_layer_Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels, Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer, Tensor *dBias, bool withBias, u8 stride) {
  Tensor *inputContig = input;
  Tensor *kernelContig = kernels;
  Tensor *outputGradContig = outputGrad;
  Tensor *colBufferContig = colBuffer;
  void *dColBuffer = NULL;
  double totalStartMs = 0.0;
  double phaseStartMs = 0.0;
  Result res = OK;

  if (ctx == NULL || isInvalidTensor(input) || isInvalidTensor(kernels) || isInvalidTensor(outputGrad) || dInput == NULL || dKernels == NULL) {
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

  if (input->shape.numOfDims != 4 || kernels->shape.numOfDims != 4 || outputGrad->shape.numOfDims != 4) {
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

  if (outputGrad->shape.dims[0] != batch || outputGrad->shape.dims[1] != outH || outputGrad->shape.dims[2] != outW || outputGrad->shape.dims[3] != outChannels) {
    return ERR_DIM_MISMATCH;
  }
  if (withBias) {
    if (isInvalidTensor(dBias)) {
      return ERR_NULL_TENSOR_PROVIDED;
    }
    if (dBias->dtype != input->dtype) {
      return ERR_DTYPE_MISMATCH;
    }
    if (dBias->shape.numOfDims != 4 || dBias->shape.dims == NULL) {
      return ERR_DIM_MISMATCH;
    }
    if (dBias->shape.dims[0] != 1 || dBias->shape.dims[1] != 1 || dBias->shape.dims[2] != 1 || dBias->shape.dims[3] != outChannels) {
      return ERR_DIM_MISMATCH;
    }
  }

  kernelContig = materializeTensorOnContext(ctx, kernels);
  outputGradContig = materializeTensorOnContext(ctx, outputGrad);
  colBufferContig = materializeTensorOnContext(ctx, colBuffer);
  PANIC_IF(!isSameContext(dInput->context, ctx), ERR_NO_OP);
  PANIC_IF(!isSameContext(dKernels->context, ctx), ERR_NO_OP);
  if (withBias) {
    PANIC_IF(!isSameContext(dBias->context, ctx), ERR_NO_OP);
  }

  res = clearTensorValues(dInput);
  if (res != OK) {
    return res;
  }
  res = clearTensorValues(dKernels);
  if (res != OK) {
    return res;
  }
  if (withBias) {
    res = clearTensorValues(dBias);
    if (res != OK) {
      return res;
    }
  }

  dim_t C_in = input->shape.dims[3];
  dim_t kS = C_in * kH * kW;

  if (input->dtype == F64) {
    f64 *wValues = kernelContig->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f64 *dWValues = dKernels->values;        // (C_out, Cin, kH, kW) -> (C_out, kS)
    f64 *dOutput = outputGradContig->values; // (B,outH, outW, C_out)

    dim_t batch = input->shape.dims[0];
    dim_t outH = outputGradContig->shape.dims[1];
    dim_t outW = outputGradContig->shape.dims[2];
    dim_t C_out = outputGradContig->shape.dims[3];

    dim_t outputPositions = batch * outH * outW;

    Tensor dColBufferTensor = t_Empty(ctx, SHAPE1D(outputPositions * kS), F64);
    dColBuffer = dColBufferTensor.values;

    runGemm(ctx, F64, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out, colBufferContig->values, kS, false, dWValues, kS);

    runGemm(ctx, F64, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out, wValues, kS, false, dColBuffer, kS);

    col2imAccumulateF64(dInput, dColBuffer, kH, kW, stride);
  } else {
    f32 *wValues = kernelContig->values;     // (C_out, Cin, kH, kW) -> (C_out, kS)
    f32 *dWValues = dKernels->values;        // (C_out, Cin, kH, kW) -> (C_out, kS)
    f32 *dOutput = outputGradContig->values; // (B,outH, outW, C_out)

    dim_t batch = input->shape.dims[0];
    dim_t outH = outputGradContig->shape.dims[1];
    dim_t outW = outputGradContig->shape.dims[2];
    dim_t C_out = outputGradContig->shape.dims[3];

    dim_t outputPositions = batch * outH * outW;

    Tensor dColBufferTensor = t_Empty(ctx, SHAPE1D(outputPositions * kS), F32);
    dColBuffer = dColBufferTensor.values;

    runGemm(ctx, F32, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out, colBufferContig->values, kS, false, dWValues, kS);

    runGemm(ctx, F32, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out, wValues, kS, false, dColBuffer, kS);

    col2imAccumulateF32(dInput, dColBuffer, kH, kW, stride);
  }

  if (withBias) {
    accumulateConvBiasGrad(ctx, outputGradContig, dBias);
  }

  res = OK;

  return res;
}

Result shapes_layer_ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Dim kernelShape, Tensor *t, Tensor *dest) {
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
  if (kernels->shape.dims[0] != inChannels || kernels->shape.dims[1] != outChannels || kernels->shape.dims[2] != kH || kernels->shape.dims[3] != kW) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->dtype != t->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  Tensor *createdDest = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDest == NULL, ALLOCATION_FAILED);
  *createdDest = t_Zeros(ctx, SHAPE4D(batch, outH, outW, outChannels), t->dtype);
  if (createdDest == NULL) {
    return ERR_OUT_OF_MEMORY;
  }
  *dest = *createdDest;

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
                  dim_t outIdx = (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
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
                  dim_t outIdx = (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
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

Result shapes_layer_ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride, Tensor *dX, Tensor *dKernels) {
  if (isInvalidTensor(x) || isInvalidTensor(kernels) || isInvalidTensor(gradOut) || dX == NULL || dKernels == NULL) {
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

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outH || gradOut->shape.dims[2] != outW || gradOut->shape.dims[3] != outChannels) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *createdDX = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *createdDX = t_Zeros(ctx, x->shape, x->dtype);
  if (createdDX == NULL) {
    return ALLOCATION_FAILED;
  }
  *dX = *createdDX;

  Tensor *createdDKernels = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDKernels == NULL, ALLOCATION_FAILED);
  *createdDKernels = t_Zeros(ctx, kernels->shape, kernels->dtype);
  if (createdDKernels == NULL) {
    return ALLOCATION_FAILED;
  }
  *dKernels = *createdDKernels;

  Result res = OK;

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
                  dim_t gradIdx = (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
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
                  dim_t gradIdx = (((b * outH + (outY + ky)) * outW + outX + kx) * outChannels) + oc;
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
