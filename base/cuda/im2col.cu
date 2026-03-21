#include "../layer/im2col.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__global__ static void im2colKernel(const T *input, dim_t batch, dim_t inChannels, dim_t h, dim_t w,
                                    dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW,
                                    T *colBuffer) {
  dim_t positions = outH * outW;
  dim_t patchSize = inChannels * kH * kW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * positions * patchSize;
  if (idx >= total) {
    return;
  }

  dim_t row = idx / patchSize;
  dim_t patchIdx = idx % patchSize;
  dim_t batchIdx = row / positions;
  dim_t posIdx = row % positions;
  dim_t outY = posIdx / outW;
  dim_t outX = posIdx % outW;
  dim_t kernelArea = kH * kW;
  dim_t channel = patchIdx / kernelArea;
  dim_t kernelOffset = patchIdx % kernelArea;
  dim_t kernelY = kernelOffset / kW;
  dim_t kernelX = kernelOffset % kW;
  dim_t inY = outY * stride + kernelY;
  dim_t inX = outX * stride + kernelX;

  size_t inputIdx = ((((size_t)batchIdx * h + inY) * w + inX) * inChannels) + channel;
  colBuffer[idx] = input[inputIdx];
}

template <typename T>
__global__ static void col2imKernel(T *dest, const T *colBuffer, dim_t batch, dim_t inChannels,
                                    dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, dim_t outH,
                                    dim_t outW) {
  dim_t positions = outH * outW;
  dim_t patchSize = inChannels * kH * kW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * positions * patchSize;
  if (idx >= total) {
    return;
  }

  dim_t row = idx / patchSize;
  dim_t patchIdx = idx % patchSize;
  dim_t batchIdx = row / positions;
  dim_t posIdx = row % positions;
  dim_t outY = posIdx / outW;
  dim_t outX = posIdx % outW;
  dim_t kernelArea = kH * kW;
  dim_t channel = patchIdx / kernelArea;
  dim_t kernelOffset = patchIdx % kernelArea;
  dim_t kernelY = kernelOffset / kW;
  dim_t kernelX = kernelOffset % kW;
  dim_t inY = outY * stride + kernelY;
  dim_t inX = outX * stride + kernelX;

  size_t destIdx = ((((size_t)batchIdx * h + inY) * w + inX) * inChannels) + channel;
  atomicAdd(&dest[destIdx], colBuffer[idx]);
}

template <typename T>
static Result launchIm2colKernel(const void *input, dim_t batch, dim_t inChannels, dim_t h, dim_t w,
                                 dim_t kH, dim_t kW, u8 stride, void *colBuffer) {
  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;
  size_t total = (size_t)batch * outH * outW * inChannels * kH * kW;
  int threadsPerBlock = 256;
  int blocks = (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  im2colKernel<<<blocks, threadsPerBlock>>>((const T *)input, batch, inChannels, h, w, kH, kW,
                                            stride, outH, outW, (T *)colBuffer);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename T>
static Result launchCol2imKernel(void *dest, const void *colBuffer, dim_t batch, dim_t inChannels,
                                 dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride) {
  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;
  size_t total = (size_t)batch * outH * outW * inChannels * kH * kW;
  int threadsPerBlock = 256;
  int blocks = (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  col2imKernel<<<blocks, threadsPerBlock>>>((T *)dest, (const T *)colBuffer, batch, inChannels, h,
                                            w, kH, kW, stride, outH, outW);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaIm2col(Context *ctx, Dtype dtype, const void *input, dim_t batch,
                                dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                                u8 stride, void *colBuffer) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
    case F32: return launchIm2colKernel<f32>(input, batch, inChannels, h, w, kH, kW, stride, colBuffer);
    case F64: return launchIm2colKernel<f64>(input, batch, inChannels, h, w, kH, kW, stride, colBuffer);
    default: return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaCol2imAccumulate(Context *ctx, Dtype dtype, void *dest, const void *colBuffer,
                                          dim_t batch, dim_t inChannels, dim_t h, dim_t w, dim_t kH,
                                          dim_t kW, u8 stride) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (dtype) {
    case F32:
      return launchCol2imKernel<f32>(dest, colBuffer, batch, inChannels, h, w, kH, kW, stride);
    case F64:
      return launchCol2imKernel<f64>(dest, colBuffer, batch, inChannels, h, w, kH, kW, stride);
    default: return ERR_DTYPE_MISMATCH;
  }
}
