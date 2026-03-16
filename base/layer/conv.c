#include "im2col.h"
#include "common.h"
#include "cblas.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void runGemm(Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n,
                    int k, const void *a, int lda, const void *b, int ldb, bool accumulate, void *c,
                    int ldc) {
  if (dtype == F64) {
    cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, 1.0, (const double *)a, lda,
                (const double *)b, ldb, accumulate ? 1.0 : 0.0, (double *)c, ldc);
    return;
  }

  cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, 1.0f, (const float *)a, lda, (const float *)b,
              ldb, accumulate ? 1.0f : 0.0f, (float *)c, ldc);
}

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

static int parseConvThreadOverride(void) {
  const char *override = getenv("SHAPES_CONV_THREADS");
  if (override == NULL || *override == '\0') {
    return 0;
  }

  char *end = NULL;
  long parsed = strtol(override, &end, 10);
  if (end == override || *end != '\0' || parsed < 1) {
    return 0;
  }

  return clampBlasThreadCount((int)parsed);
}

static int chooseConvThreadCount(tensor_size_t patchSize, tensor_size_t positions,
                                 size_t outChannels) {
  int override = parseConvThreadOverride();
  if (override > 0) {
    return override;
  }

  size_t gemmWork = outChannels * patchSize * positions;
  if (gemmWork < 262144 || positions < 1024) {
    return 1;
  }

  // Conv still lowers to relatively skinny GEMMs here, so thread overhead can
  // dominate if we scale too aggressively.
  int desiredByChannels = (int)(outChannels / 64);
  int desiredByPatch = (int)(patchSize / 256);
  int desiredThreads = desiredByChannels < desiredByPatch ? desiredByChannels : desiredByPatch;
  if (desiredThreads < 1) {
    desiredThreads = 1;
  }

  return clampBlasThreadCount(desiredThreads);
}

static int beginConvThreadScope(tensor_size_t patchSize, tensor_size_t positions,
                                size_t outChannels) {
  int desiredThreads = chooseConvThreadCount(patchSize, positions, outChannels);
  int previousThreads = openblas_get_num_threads();

  if (desiredThreads < 1 || desiredThreads == previousThreads) {
    return 0;
  }

  openblas_set_num_threads_local(desiredThreads);
  return previousThreads;
}

static void endConvThreadScope(int previousThreads) {
  if (previousThreads > 0) {
    openblas_set_num_threads_local(previousThreads);
  }
}

Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Dim kernelShape, Tensor *t, Tensor *dest) {
  Tensor *inputContig = t;
  Tensor *kernelContig = kernels;
  void *colBuffer = NULL;
  bool destInitialized = false;

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

  if (kH == 0 || kW == 0 || c != inChannels || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  if (kernels->shape.numOfDims < 4) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->shape.dims[0] != outChannels || kernels->shape.dims[1] != inChannels ||
      kernels->shape.dims[2] != kH || kernels->shape.dims[3] != kW) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->dtype != t->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  Result res = init4DTensor(ctx, dest, batch, outChannels, outH, outW, t->dtype);
  if (res != OK) {
    return res;
  }
  destInitialized = true;

  if (!inputContig->isContigous) {
    inputContig = copyToContiguous(ctx, inputContig);
  }
  if (!kernelContig->isContigous) {
    kernelContig = copyToContiguous(ctx, kernelContig);
  }

  tensor_size_t patchSize = inChannels * kH * kW;
  tensor_size_t positions = outH * outW;
  size_t elemSize = getBytesForDtype(t->dtype);
  int previousBlasThreads = beginConvThreadScope(patchSize, positions, outChannels);
  colBuffer = allocate(ctx->memory, patchSize * positions * elemSize);
  if (colBuffer == NULL) {
    res = ERR_OUT_OF_MEMORY;
    goto cleanup;
  }

  if (t->dtype == F64) {
    f64 *input = inputContig->values;
    f64 *kernelValues = kernelContig->values;
    f64 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      f64 *inputBatch = input + b * inChannels * h * w;
      f64 *outBatch = outValues + b * outChannels * positions;

      im2colNchwF64(inputBatch, inChannels, h, w, kH, kW, stride, outH, outW, colBuffer);
      runGemm(t->dtype, CblasNoTrans, CblasNoTrans, (int)outChannels, (int)positions,
              (int)patchSize, kernelValues, (int)patchSize, colBuffer, (int)positions, false,
              outBatch, (int)positions);
    }
  } else {
    f32 *input = inputContig->values;
    f32 *kernelValues = kernelContig->values;
    f32 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      f32 *inputBatch = input + b * inChannels * h * w;
      f32 *outBatch = outValues + b * outChannels * positions;

      im2colNchwF32(inputBatch, inChannels, h, w, kH, kW, stride, outH, outW, colBuffer);
      runGemm(t->dtype, CblasNoTrans, CblasNoTrans, (int)outChannels, (int)positions,
              (int)patchSize, kernelValues, (int)patchSize, colBuffer, (int)positions, false,
              outBatch, (int)positions);
    }
  }

  res = OK;

cleanup:
  endConvThreadScope(previousBlasThreads);
  if (colBuffer != NULL) {
    freeAlloc(ctx->memory, colBuffer);
  }
  if (inputContig != t) {
    FreeTensor(ctx, inputContig);
  }
  if (kernelContig != kernels) {
    FreeTensor(ctx, kernelContig);
  }
  if (res != OK && destInitialized) {
    freeTensorBuffers(ctx, dest);
  }

  return res;
}

Result Conv2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride,
                      Tensor *dX, Tensor *dKernels) {
  Tensor *xContig = x;
  Tensor *kernelContig = kernels;
  Tensor *gradContig = gradOut;
  void *colBuffer = NULL;
  void *dColBuffer = NULL;
  bool dxInitialized = false;
  bool dKernelsInitialized = false;

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
  dim_t outChannels = kernels->shape.dims[0];
  dim_t kernelInChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outChannels ||
      gradOut->shape.dims[2] != outH || gradOut->shape.dims[3] != outW) {
    return ERR_DIM_MISMATCH;
  }

  Result res = initTensorLike(ctx, dX, x, x->dtype);
  if (res != OK) {
    return res;
  }
  dxInitialized = true;
  res = initTensorLike(ctx, dKernels, kernels, kernels->dtype);
  if (res != OK) {
    freeTensorBuffers(ctx, dX);
    return res;
  }
  dKernelsInitialized = true;

  memset(dX->values, 0, dX->size * getBytesForDtype(dX->dtype));
  memset(dKernels->values, 0, dKernels->size * getBytesForDtype(dKernels->dtype));

  if (!xContig->isContigous) {
    xContig = copyToContiguous(ctx, xContig);
  }
  if (!kernelContig->isContigous) {
    kernelContig = copyToContiguous(ctx, kernelContig);
  }
  if (!gradContig->isContigous) {
    gradContig = copyToContiguous(ctx, gradContig);
  }

  tensor_size_t patchSize = inChannels * kH * kW;
  tensor_size_t positions = outH * outW;
  size_t elemSize = getBytesForDtype(x->dtype);
  int previousBlasThreads = beginConvThreadScope(patchSize, positions, outChannels);

  colBuffer = allocate(ctx->memory, patchSize * positions * elemSize);
  dColBuffer = allocate(ctx->memory, patchSize * positions * elemSize);
  if (colBuffer == NULL || dColBuffer == NULL) {
    res = ERR_OUT_OF_MEMORY;
    goto cleanup;
  }

  if (x->dtype == F64) {
    f64 *xValues = xContig->values;
    f64 *kernelValues = kernelContig->values;
    f64 *gradValues = gradContig->values;
    f64 *dxValues = dX->values;
    f64 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      f64 *xBatch = xValues + b * inChannels * h * w;
      f64 *gradBatch = gradValues + b * outChannels * positions;
      f64 *dxBatch = dxValues + b * inChannels * h * w;

      im2colNchwF64(xBatch, inChannels, h, w, kH, kW, stride, outH, outW, colBuffer);
      runGemm(x->dtype, CblasNoTrans, CblasTrans, (int)outChannels, (int)patchSize, (int)positions,
              gradBatch, (int)positions, colBuffer, (int)positions, true, dKernelValues,
              (int)patchSize);
      runGemm(x->dtype, CblasTrans, CblasNoTrans, (int)patchSize, (int)positions, (int)outChannels,
              kernelValues, (int)patchSize, gradBatch, (int)positions, false, dColBuffer,
              (int)positions);
      col2imNchwAddF64(dColBuffer, inChannels, h, w, kH, kW, stride, outH, outW, dxBatch);
    }
  } else {
    f32 *xValues = xContig->values;
    f32 *kernelValues = kernelContig->values;
    f32 *gradValues = gradContig->values;
    f32 *dxValues = dX->values;
    f32 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      f32 *xBatch = xValues + b * inChannels * h * w;
      f32 *gradBatch = gradValues + b * outChannels * positions;
      f32 *dxBatch = dxValues + b * inChannels * h * w;

      im2colNchwF32(xBatch, inChannels, h, w, kH, kW, stride, outH, outW, colBuffer);
      runGemm(x->dtype, CblasNoTrans, CblasTrans, (int)outChannels, (int)patchSize, (int)positions,
              gradBatch, (int)positions, colBuffer, (int)positions, true, dKernelValues,
              (int)patchSize);
      runGemm(x->dtype, CblasTrans, CblasNoTrans, (int)patchSize, (int)positions, (int)outChannels,
              kernelValues, (int)patchSize, gradBatch, (int)positions, false, dColBuffer,
              (int)positions);
      col2imNchwAddF32(dColBuffer, inChannels, h, w, kH, kW, stride, outH, outW, dxBatch);
    }
  }

  res = OK;

cleanup:
  endConvThreadScope(previousBlasThreads);
  if (colBuffer != NULL) {
    freeAlloc(ctx->memory, colBuffer);
  }
  if (dColBuffer != NULL) {
    freeAlloc(ctx->memory, dColBuffer);
  }
  if (xContig != x) {
    FreeTensor(ctx, xContig);
  }
  if (kernelContig != kernels) {
    FreeTensor(ctx, kernelContig);
  }
  if (gradContig != gradOut) {
    FreeTensor(ctx, gradContig);
  }
  if (res != OK && dxInitialized) {
    freeTensorBuffers(ctx, dX);
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
