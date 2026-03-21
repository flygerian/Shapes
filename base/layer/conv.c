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

Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Tensor *t, Tensor *dest, Tensor *colBufferDest) {
  Tensor *inputContig = t;
  Tensor *kernelContig = kernels;
  Tensor gemmOutput;
  Tensor permutedOutput;
  bool gemmOutputInitialized = false;
  bool permutedOutputInitialized = false;

  Result result;

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
  dim_t numChannels = t->shape.dims[1];
  dim_t height = t->shape.dims[2];
  dim_t width = t->shape.dims[3];

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

  result = init4DTensor(ctx, &gemmOutput, batch, outputChannelHeight, outputChannelWidth,
                        outChannels, t->dtype);
  if (result != OK) {
    goto cleanup;
  }

  gemmOutputInitialized = true;
  if (!inputContig->isContigous) {
    inputContig = copyToContiguous(ctx, inputContig);
  }
  if (!kernelContig->isContigous) {
    // If this happens something has gone terribly wrong
    result = ERR_CONV2D_KERNEL_NOT_CONTIGOUS;
    goto cleanup;
  }

  tensor_size_t patchSize = inChannels * kernelHeight * kernelWidth;
  tensor_size_t positions = outputChannelHeight * outputChannelWidth;

  Tensor *colBuffer;
  tensor_size_t colBufferSize;

  if (t->dtype == F64) {
    f64 *kernelValues = kernelContig->values;

    colBuffer = im2colF64(ctx, inputContig, kernelHeight, kernelWidth, stride);

    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels,
            (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);
  } else {
    f32 *kernelValues = kernelContig->values;

    colBuffer = im2colF32(ctx, inputContig, kernelHeight, kernelWidth, stride);

    runGemm(ctx, t->dtype, CblasNoTrans, CblasTrans, (int)batch * positions, (int)outChannels,
            (int)patchSize, colBuffer->values, (int)patchSize, kernelValues, (int)patchSize, false,
            gemmOutput.values, (int)outChannels);
  }

  if (colBufferDest != NULL) {
    *colBufferDest = *colBuffer;
    freeAlloc(ctx->memory, colBuffer);
  }

  dim_t orderDims[4] = {0, 3, 1, 2};
  Dim order = {.dims = orderDims, .numOfDims = 4, .multipliers = NULL};
  result = Permute(ctx, &gemmOutput, &permutedOutput, order);

  if (result != OK) {
    goto cleanup;
  }
  permutedOutputInitialized = true;

  Tensor *contiguousDest = copyToContiguous(ctx, &permutedOutput);
  if (contiguousDest == NULL) {
    result = ERR_OUT_OF_MEMORY;
    goto cleanup;
  }
  *dest = *contiguousDest;
  freeAlloc(ctx->memory, contiguousDest);

  result = OK;

cleanup:
  if (inputContig != t) {
    FreeTensor(ctx, inputContig);
  }
  if (kernelContig != kernels) {
    FreeTensor(ctx, kernelContig);
  }
  if (permutedOutputInitialized) {
    if (permutedOutput.shape.dims != NULL) {
      freeAlloc(ctx->memory, permutedOutput.shape.dims);
    }
    if (permutedOutput.shape.multipliers != NULL) {
      freeAlloc(ctx->memory, permutedOutput.shape.multipliers);
    }
    if (permutedOutput.boundary != NULL) {
      freeAlloc(ctx->memory, permutedOutput.boundary);
    }
  }
  if (gemmOutputInitialized) {
    freeTensorBuffers(ctx, &gemmOutput);
  }

  return result;
}

Result Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels,
                       Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer, u8 stride) {
  Tensor *inputContig = input;
  Tensor *kernelContig = kernels;
  Tensor *outputGradContig = outputGrad;
  void *dColBuffer = NULL;
  bool dInputInitialized = false;
  bool dKernelsInitialized = false;

  if (isInvalidTensor(input) || isInvalidTensor(kernels) || isInvalidTensor(outputGrad) ||
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
  dim_t inChannels = input->shape.dims[1];
  dim_t h = input->shape.dims[2];
  dim_t w = input->shape.dims[3];
  dim_t outChannels = kernels->shape.dims[0];
  dim_t kernelInChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  if (outputGrad->shape.dims[0] != batch || outputGrad->shape.dims[1] != outChannels ||
      outputGrad->shape.dims[2] != outH || outputGrad->shape.dims[3] != outW) {
    return ERR_DIM_MISMATCH;
  }

  Result res;

  if (!inputContig->isContigous) {
    inputContig = copyToContiguous(ctx, inputContig);
  }
  if (!kernelContig->isContigous) {
    kernelContig = copyToContiguous(ctx, kernelContig);
  }

  // Expecting outputgrad to be (NCHW) so we need to permute it and copy to contigous
  // to make it NHWC because it was permuted to NCHW in Conv2d
  Tensor outGradNhwc;
  dim_t orderDims[4] = {0, 2, 3, 1};
  Dim order = {.dims = orderDims, .numOfDims = 4, .multipliers = NULL};
  Permute(ctx, outputGrad, &outGradNhwc, order);
  outputGradContig = copyToContiguous(ctx, &outGradNhwc);

  dim_t C_in = input->shape.dims[1];
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

    dColBuffer = allocate(ctx->memory, (outputPositions * kS) * sizeof(f64));
    if (dColBuffer == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    runGemm(ctx, F64, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out, colBuffer->values,
            kS, false, dWValues, kS);

    runGemm(ctx, F64, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out, wValues,
            kS, false, dColBuffer, kS);

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

    dColBuffer = allocate(ctx->memory, (outputPositions * kS) * sizeof(f32));
    if (dColBuffer == NULL) {
      res = ERR_OUT_OF_MEMORY;
      goto cleanup;
    }

    runGemm(ctx, F32, CblasTrans, CblasNoTrans, C_out, kS, outputPositions, dOutput, C_out, colBuffer->values,
            kS, false, dWValues, kS);

    runGemm(ctx, F32, CblasNoTrans, CblasNoTrans, outputPositions, kS, C_out, dOutput, C_out, wValues,
            kS, false, dColBuffer, kS);

    col2imAccumulateF32(dInput, dColBuffer, kH, kW, stride);
  }

  res = OK;

cleanup:
  freeTensorBuffers(ctx, &outGradNhwc);

  if (dColBuffer != NULL) {
    freeAlloc(ctx->memory, dColBuffer);
  }
  if (inputContig != input) {
    FreeTensor(ctx, inputContig);
  }
  if (kernelContig != kernels) {
    FreeTensor(ctx, kernelContig);
  }
  if (outputGradContig != outputGrad) {
    FreeTensor(ctx, outputGradContig);
  }
  if (res != OK && dInputInitialized) {
    freeTensorBuffers(ctx, dInput);
  }
  if (res != OK && dKernelsInitialized) {
    freeTensorBuffers(ctx, dKernels);
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
  dim_t c = t->shape.dims[1];
  dim_t h = t->shape.dims[2];
  dim_t w = t->shape.dims[3];

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

  Result res = init4DTensor(ctx, dest, batch, outChannels, outH, outW, t->dtype);
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
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            f64 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx = (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
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
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            f32 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx = (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
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
  dim_t inChannels = x->shape.dims[1];
  dim_t h = x->shape.dims[2];
  dim_t w = x->shape.dims[3];
  dim_t kernelInChannels = kernels->shape.dims[0];
  dim_t outChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outChannels ||
      gradOut->shape.dims[2] != outH || gradOut->shape.dims[3] != outW) {
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
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
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
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
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
