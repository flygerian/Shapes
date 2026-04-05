#include "../common.h"
#include "../layer/pool.h"
#include "../result/result.h"
#include <cuda_runtime.h>

static __device__ dim_t adaptivePoolStart(dim_t outIdx, dim_t inputSize,
                                          dim_t outputSize) {
  return (outIdx * inputSize) / outputSize;
}

static __device__ dim_t adaptivePoolEnd(dim_t outIdx, dim_t inputSize,
                                        dim_t outputSize) {
  return ((outIdx + 1) * inputSize + outputSize - 1) / outputSize;
}

template <typename T>
__global__ static void
maxPool2dKernel(const T *input, dim_t batch, dim_t channels, dim_t h, dim_t w,
                dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW,
                T *output, u64 *indices) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  dim_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  dim_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  dim_t outY = rowIdx % outH;
  dim_t batchIdx = rowIdx / outH;
  dim_t startY = outY * stride;
  dim_t startX = outX * stride;
  size_t maxIdx =
      ((((size_t)batchIdx * h + startY) * w + startX) * channels) + channelIdx;
  T maxValue = input[maxIdx];
  for (dim_t ky = 0; ky < kH; ky++) {
    for (dim_t kx = 0; kx < kW; kx++) {
      size_t inputIdx =
          ((((size_t)batchIdx * h + (startY + ky)) * w + (startX + kx)) *
           channels) +
          channelIdx;
      T candidate = input[inputIdx];
      if (candidate > maxValue) {
        maxValue = candidate;
        maxIdx = inputIdx;
      }
    }
  }

  output[idx] = maxValue;
  if (indices != nullptr) {
    indices[idx] = (u64)maxIdx;
  }
}

template <typename T>
__global__ static void
maxPool2dBackwardKernel(const T *input, const T *gradOut, dim_t batch,
                        dim_t channels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                        u8 stride, dim_t outH, dim_t outW, T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  dim_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  dim_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  dim_t outY = rowIdx % outH;
  dim_t batchIdx = rowIdx / outH;
  dim_t startY = outY * stride;
  dim_t startX = outX * stride;
  size_t maxIdx =
      ((((size_t)batchIdx * h + startY) * w + startX) * channels) + channelIdx;
  T maxValue = input[maxIdx];
  for (dim_t ky = 0; ky < kH; ky++) {
    for (dim_t kx = 0; kx < kW; kx++) {
      size_t inputIdx =
          ((((size_t)batchIdx * h + (startY + ky)) * w + (startX + kx)) *
           channels) +
          channelIdx;
      T candidate = input[inputIdx];
      if (candidate > maxValue) {
        maxValue = candidate;
        maxIdx = inputIdx;
      }
    }
  }

  atomicAdd(&dX[maxIdx], gradOut[idx]);
}

template <typename T>
__global__ static void
maxPool2dBackwardIndicesKernel(const T *gradOut, const u64 *indices,
                               tensor_size_t numGradValues, T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numGradValues) {
    return;
  }

  atomicAdd(&dX[indices[idx]], gradOut[idx]);
}

template <typename T>
__global__ static void
adaptiveAvgPool2dKernel(const T *input, dim_t batch, dim_t channels, dim_t h,
                        dim_t w, dim_t outH, dim_t outW, T *output) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  dim_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  dim_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  dim_t outY = rowIdx % outH;
  dim_t batchIdx = rowIdx / outH;
  dim_t startY = adaptivePoolStart(outY, h, outH);
  dim_t endY = adaptivePoolEnd(outY, h, outH);
  dim_t startX = adaptivePoolStart(outX, w, outW);
  dim_t endX = adaptivePoolEnd(outX, w, outW);
  dim_t count = (endY - startY) * (endX - startX);

  T sum = (T)0;
  for (dim_t iy = startY; iy < endY; iy++) {
    for (dim_t ix = startX; ix < endX; ix++) {
      sum += input[((((size_t)batchIdx * h + iy) * w + ix) * channels) +
                   channelIdx];
    }
  }

  output[idx] = sum / (T)count;
}

template <typename T>
__global__ static void
adaptiveAvgPool2dBackwardKernel(const T *gradOut, dim_t batch, dim_t channels,
                                dim_t h, dim_t w, dim_t outH, dim_t outW,
                                T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  dim_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  dim_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  dim_t outY = rowIdx % outH;
  dim_t batchIdx = rowIdx / outH;
  dim_t startY = adaptivePoolStart(outY, h, outH);
  dim_t endY = adaptivePoolEnd(outY, h, outH);
  dim_t startX = adaptivePoolStart(outX, w, outW);
  dim_t endX = adaptivePoolEnd(outX, w, outW);
  dim_t count = (endY - startY) * (endX - startX);
  T scaledGrad = gradOut[idx] / (T)count;

  for (dim_t iy = startY; iy < endY; iy++) {
    for (dim_t ix = startX; ix < endX; ix++) {
      atomicAdd(
          &dX[((((size_t)batchIdx * h + iy) * w + ix) * channels) + channelIdx],
          scaledGrad);
    }
  }
}

static Result finishPoolLaunch() {
  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename T>
static Result launchMaxPool2d(const void *input, dim_t batch, dim_t channels,
                              dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride,
                              void *output, void *indices) {
  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  maxPool2dKernel<<<blocks, threadsPerBlock>>>(
      (const T *)input, batch, channels, h, w, kH, kW, stride, outH, outW,
      (T *)output, (u64 *)indices);
  return finishPoolLaunch();
}

template <typename T>
static Result launchMaxPool2dBackward(const void *input, const void *gradOut,
                                      dim_t batch, dim_t channels, dim_t h,
                                      dim_t w, dim_t kH, dim_t kW, u8 stride,
                                      void *dX) {
  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  maxPool2dBackwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)input, (const T *)gradOut, batch, channels, h, w, kH, kW,
      stride, outH, outW, (T *)dX);
  return finishPoolLaunch();
}

template <typename T>
static Result
launchMaxPool2dBackwardWithIndices(const void *gradOut, const void *indices,
                                   tensor_size_t numGradValues, void *dX) {
  int threadsPerBlock = 256;
  int blocks = (int)((numGradValues + (tensor_size_t)threadsPerBlock - 1) /
                     (tensor_size_t)threadsPerBlock);
  maxPool2dBackwardIndicesKernel<<<blocks, threadsPerBlock>>>(
      (const T *)gradOut, (const u64 *)indices, numGradValues, (T *)dX);
  return finishPoolLaunch();
}

template <typename T>
static Result launchAdaptiveAvgPool2d(const void *input, dim_t batch,
                                      dim_t channels, dim_t h, dim_t w,
                                      dim_t outH, dim_t outW, void *output) {
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  adaptiveAvgPool2dKernel<<<blocks, threadsPerBlock>>>(
      (const T *)input, batch, channels, h, w, outH, outW, (T *)output);
  return finishPoolLaunch();
}

template <typename T>
static Result launchAdaptiveAvgPool2dBackward(const void *gradOut, dim_t batch,
                                              dim_t channels, dim_t h, dim_t w,
                                              dim_t outH, dim_t outW,
                                              void *dX) {
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  adaptiveAvgPool2dBackwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)gradOut, batch, channels, h, w, outH, outW, (T *)dX);
  return finishPoolLaunch();
}

extern "C" Result runCudaMaxPool2d(Context *ctx, Dtype dtype, const void *input,
                                   dim_t batch, dim_t channels, dim_t h,
                                   dim_t w, dim_t kH, dim_t kW, u8 stride,
                                   void *output) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchMaxPool2d<f32>(input, batch, channels, h, w, kH, kW, stride,
                                output, nullptr);
  case F64:
    return launchMaxPool2d<f64>(input, batch, channels, h, w, kH, kW, stride,
                                output, nullptr);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaMaxPool2dWithIndices(Context *ctx, Dtype dtype,
                                              const void *input, dim_t batch,
                                              dim_t channels, dim_t h, dim_t w,
                                              dim_t kH, dim_t kW, u8 stride,
                                              void *output, void *indices) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchMaxPool2d<f32>(input, batch, channels, h, w, kH, kW, stride,
                                output, indices);
  case F64:
    return launchMaxPool2d<f64>(input, batch, channels, h, w, kH, kW, stride,
                                output, indices);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaMaxPool2dBackward(Context *ctx, Dtype dtype,
                                           const void *input,
                                           const void *gradOut, dim_t batch,
                                           dim_t channels, dim_t h, dim_t w,
                                           dim_t kH, dim_t kW, u8 stride,
                                           void *dX) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchMaxPool2dBackward<f32>(input, gradOut, batch, channels, h, w,
                                        kH, kW, stride, dX);
  case F64:
    return launchMaxPool2dBackward<f64>(input, gradOut, batch, channels, h, w,
                                        kH, kW, stride, dX);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result
runCudaMaxPool2dBackwardWithIndices(Context *ctx, Dtype dtype,
                                    const void *gradOut, const void *indices,
                                    tensor_size_t numGradValues, void *dX) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchMaxPool2dBackwardWithIndices<f32>(gradOut, indices,
                                                   numGradValues, dX);
  case F64:
    return launchMaxPool2dBackwardWithIndices<f64>(gradOut, indices,
                                                   numGradValues, dX);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaAdaptiveAvgPool2d(Context *ctx, Dtype dtype,
                                           const void *input, dim_t batch,
                                           dim_t channels, dim_t h, dim_t w,
                                           dim_t outH, dim_t outW,
                                           void *output) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchAdaptiveAvgPool2d<f32>(input, batch, channels, h, w, outH,
                                        outW, output);
  case F64:
    return launchAdaptiveAvgPool2d<f64>(input, batch, channels, h, w, outH,
                                        outW, output);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaAdaptiveAvgPool2dBackward(Context *ctx, Dtype dtype,
                                                   const void *gradOut,
                                                   dim_t batch, dim_t channels,
                                                   dim_t h, dim_t w, dim_t outH,
                                                   dim_t outW, void *dX) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
  case F32:
    return launchAdaptiveAvgPool2dBackward<f32>(gradOut, batch, channels, h, w,
                                                outH, outW, dX);
  case F64:
    return launchAdaptiveAvgPool2dBackward<f64>(gradOut, batch, channels, h, w,
                                                outH, outW, dX);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
