#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>

static __device__ size_t adaptivePoolStart(size_t outIdx, size_t inputSize,
                                           size_t outputSize) {
  return (outIdx * inputSize) / outputSize;
}

static __device__ size_t adaptivePoolEnd(size_t outIdx, size_t inputSize,
                                         size_t outputSize) {
  return ((outIdx + 1) * inputSize + outputSize - 1) / outputSize;
}

template <typename T>
__global__ static void
maxPool2dKernel(const T *input, size_t batch, size_t channels, size_t h,
                size_t w, size_t kH, size_t kW, u8 stride, size_t outH,
                size_t outW, T *output, u64 *indices) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  size_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  size_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  size_t outY = rowIdx % outH;
  size_t batchIdx = rowIdx / outH;
  size_t startY = outY * stride;
  size_t startX = outX * stride;
  size_t maxIdx =
      ((((size_t)batchIdx * h + startY) * w + startX) * channels) + channelIdx;
  T maxValue = input[maxIdx];
  for (size_t ky = 0; ky < kH; ky++) {
    for (size_t kx = 0; kx < kW; kx++) {
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
maxPool2dBackwardKernel(const T *input, const T *gradOut, size_t batch,
                        size_t channels, size_t h, size_t w, size_t kH,
                        size_t kW, u8 stride, size_t outH, size_t outW, T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  size_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  size_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  size_t outY = rowIdx % outH;
  size_t batchIdx = rowIdx / outH;
  size_t startY = outY * stride;
  size_t startX = outX * stride;
  size_t maxIdx =
      ((((size_t)batchIdx * h + startY) * w + startX) * channels) + channelIdx;
  T maxValue = input[maxIdx];
  for (size_t ky = 0; ky < kH; ky++) {
    for (size_t kx = 0; kx < kW; kx++) {
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
                               size_t numGradValues, T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numGradValues) {
    return;
  }

  atomicAdd(&dX[indices[idx]], gradOut[idx]);
}

template <typename T>
__global__ static void
adaptiveAvgPool2dKernel(const T *input, size_t batch, size_t channels, size_t h,
                        size_t w, size_t outH, size_t outW, T *output) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  size_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  size_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  size_t outY = rowIdx % outH;
  size_t batchIdx = rowIdx / outH;
  size_t startY = adaptivePoolStart(outY, h, outH);
  size_t endY = adaptivePoolEnd(outY, h, outH);
  size_t startX = adaptivePoolStart(outX, w, outW);
  size_t endX = adaptivePoolEnd(outX, w, outW);
  size_t count = (endY - startY) * (endX - startX);

  T sum = (T)0;
  for (size_t iy = startY; iy < endY; iy++) {
    for (size_t ix = startX; ix < endX; ix++) {
      sum += input[((((size_t)batchIdx * h + iy) * w + ix) * channels) +
                   channelIdx];
    }
  }

  output[idx] = sum / (T)count;
}

template <typename T>
__global__ static void
adaptiveAvgPool2dBackwardKernel(const T *gradOut, size_t batch, size_t channels,
                                size_t h, size_t w, size_t outH, size_t outW,
                                T *dX) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * outH * outW * channels;
  if (idx >= total) {
    return;
  }

  size_t channelIdx = idx % channels;
  size_t pixelIdx = idx / channels;
  size_t outX = pixelIdx % outW;
  size_t rowIdx = pixelIdx / outW;
  size_t outY = rowIdx % outH;
  size_t batchIdx = rowIdx / outH;
  size_t startY = adaptivePoolStart(outY, h, outH);
  size_t endY = adaptivePoolEnd(outY, h, outH);
  size_t startX = adaptivePoolStart(outX, w, outW);
  size_t endX = adaptivePoolEnd(outX, w, outW);
  size_t count = (endY - startY) * (endX - startX);
  T scaledGrad = gradOut[idx] / (T)count;

  for (size_t iy = startY; iy < endY; iy++) {
    for (size_t ix = startX; ix < endX; ix++) {
      atomicAdd(
          &dX[((((size_t)batchIdx * h + iy) * w + ix) * channels) + channelIdx],
          scaledGrad);
    }
  }
}

static Result finishPoolLaunch() {
  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchMaxPool2d(const void *input, size_t batch, size_t channels,
                              size_t h, size_t w, size_t kH, size_t kW,
                              u8 stride, void *output, void *indices) {
  size_t outH = (h - kH) / stride + 1;
  size_t outW = (w - kW) / stride + 1;
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
                                      size_t batch, size_t channels, size_t h,
                                      size_t w, size_t kH, size_t kW, u8 stride,
                                      void *dX) {
  size_t outH = (h - kH) / stride + 1;
  size_t outW = (w - kW) / stride + 1;
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
                                   size_t numGradValues, void *dX) {
  int threadsPerBlock = 256;
  int blocks = (int)((numGradValues + (size_t)threadsPerBlock - 1) /
                     (size_t)threadsPerBlock);
  maxPool2dBackwardIndicesKernel<<<blocks, threadsPerBlock>>>(
      (const T *)gradOut, (const u64 *)indices, numGradValues, (T *)dX);
  return finishPoolLaunch();
}

template <typename T>
static Result launchAdaptiveAvgPool2d(const void *input, size_t batch,
                                      size_t channels, size_t h, size_t w,
                                      size_t outH, size_t outW, void *output) {
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  adaptiveAvgPool2dKernel<<<blocks, threadsPerBlock>>>(
      (const T *)input, batch, channels, h, w, outH, outW, (T *)output);
  return finishPoolLaunch();
}

template <typename T>
static Result launchAdaptiveAvgPool2dBackward(const void *gradOut, size_t batch,
                                              size_t channels, size_t h,
                                              size_t w, size_t outH,
                                              size_t outW, void *dX) {
  size_t total = (size_t)batch * channels * outH * outW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  adaptiveAvgPool2dBackwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)gradOut, batch, channels, h, w, outH, outW, (T *)dX);
  return finishPoolLaunch();
}

extern "C" Result runCudaMaxPool2d(shapes_Dtype dtype, const void *input,
                                   size_t batch, size_t channels, size_t h,
                                   size_t w, size_t kH, size_t kW, u8 stride,
                                   void *output) {
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

extern "C" Result
runCudaMaxPool2dWithIndices(shapes_Dtype dtype, const void *input, size_t batch,
                            size_t channels, size_t h, size_t w, size_t kH,
                            size_t kW, u8 stride, void *output, void *indices) {
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

extern "C" Result runCudaMaxPool2dBackward(shapes_Dtype dtype,
                                           const void *input,
                                           const void *gradOut, size_t batch,
                                           size_t channels, size_t h, size_t w,
                                           size_t kH, size_t kW, u8 stride,
                                           void *dX) {
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

extern "C" Result runCudaMaxPool2dBackwardWithIndices(shapes_Dtype dtype,
                                                      const void *gradOut,
                                                      const void *indices,
                                                      size_t numGradValues,
                                                      void *dX) {
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

extern "C" Result runCudaAdaptiveAvgPool2d(shapes_Dtype dtype,
                                           const void *input, size_t batch,
                                           size_t channels, size_t h, size_t w,
                                           size_t outH, size_t outW,
                                           void *output) {
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

extern "C" Result
runCudaAdaptiveAvgPool2dBackward(shapes_Dtype dtype, const void *gradOut,
                                 size_t batch, size_t channels, size_t h,
                                 size_t w, size_t outH, size_t outW, void *dX) {
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
