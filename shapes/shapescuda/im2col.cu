#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>
#include <stddef.h>

__global__ static void im2colNhwcF32Kernel3x3(const f32 *input, size_t batch,
                                              size_t inChannels, size_t h,
                                              size_t w, u8 stride, size_t outH,
                                              size_t outW, f32 *colBuffer) {
  size_t outX = (size_t)blockIdx.x;
  size_t outY = (size_t)blockIdx.y;
  size_t batchIdx = (size_t)blockIdx.z;
  size_t channel = (size_t)blockIdx.x * 0 + (size_t)threadIdx.x;
  size_t patchSize = (size_t)inChannels * 9;
  size_t rowBase = (((size_t)batchIdx * outH + outY) * outW + outX) * patchSize;
  size_t inY = outY * stride;
  size_t inX = outX * stride;
  size_t inputBase = ((((size_t)batchIdx * h + inY) * w) + inX) * inChannels;
  size_t rowStride = (size_t)w * inChannels;

  for (size_t c = channel; c < inChannels; c += (size_t)blockDim.x) {
    const f32 *topLeft = input + inputBase + c;
    const f32 *midLeft = topLeft + rowStride;
    const f32 *botLeft = midLeft + rowStride;
    f32 *dest = colBuffer + rowBase + (size_t)c * 9;

    dest[0] = topLeft[0];
    dest[1] = topLeft[inChannels];
    dest[2] = topLeft[(size_t)2 * inChannels];
    dest[3] = midLeft[0];
    dest[4] = midLeft[inChannels];
    dest[5] = midLeft[(size_t)2 * inChannels];
    dest[6] = botLeft[0];
    dest[7] = botLeft[inChannels];
    dest[8] = botLeft[(size_t)2 * inChannels];
  }
}

template <typename T>
__global__ static void
im2colNhwcKernel(const T *input, size_t batch, size_t inChannels, size_t h,
                 size_t w, size_t kH, size_t kW, u8 stride, size_t outH,
                 size_t outW, T *colBuffer) {
  size_t outX = (size_t)blockIdx.x;
  size_t outY = (size_t)blockIdx.y;
  size_t batchIdx = (size_t)blockIdx.z;
  size_t kernelArea = kH * kW;
  size_t patchSize = (size_t)inChannels * kernelArea;
  size_t rowBase = (((size_t)batchIdx * outH + outY) * outW + outX) * patchSize;
  size_t inY = outY * stride;
  size_t inX = outX * stride;
  size_t inputBase = ((((size_t)batchIdx * h + inY) * w) + inX) * inChannels;

  for (size_t patchIdx = (size_t)threadIdx.x; patchIdx < patchSize;
       patchIdx += blockDim.x) {
    size_t channel = (size_t)(patchIdx / kernelArea);
    size_t kernelOffset = (size_t)(patchIdx % kernelArea);
    size_t kernelY = kernelOffset / kW;
    size_t kernelX = kernelOffset % kW;
    size_t inputIdx =
        inputBase + (((size_t)kernelY * w + kernelX) * inChannels) + channel;
    colBuffer[rowBase + patchIdx] = input[inputIdx];
  }
}

template <typename T>
__global__ static void im2colKernel(const T *input, size_t batch,
                                    size_t inChannels, size_t h, size_t w,
                                    size_t kH, size_t kW, u8 stride,
                                    size_t outH, size_t outW, T *colBuffer) {
  size_t positions = outH * outW;
  size_t patchSize = inChannels * kH * kW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * positions * patchSize;
  if (idx >= total) {
    return;
  }

  size_t row = idx / patchSize;
  size_t patchIdx = idx % patchSize;
  size_t batchIdx = row / positions;
  size_t posIdx = row % positions;
  size_t outY = posIdx / outW;
  size_t outX = posIdx % outW;
  size_t kernelArea = kH * kW;
  size_t channel = patchIdx / kernelArea;
  size_t kernelOffset = patchIdx % kernelArea;
  size_t kernelY = kernelOffset / kW;
  size_t kernelX = kernelOffset % kW;
  size_t inY = outY * stride + kernelY;
  size_t inX = outX * stride + kernelX;

  size_t inputIdx =
      ((((size_t)batchIdx * h + inY) * w + inX) * inChannels) + channel;
  colBuffer[idx] = input[inputIdx];
}

template <typename T>
__global__ static void col2imKernel(T *dest, const T *colBuffer, size_t batch,
                                    size_t inChannels, size_t h, size_t w,
                                    size_t kH, size_t kW, u8 stride,
                                    size_t outH, size_t outW) {
  size_t positions = outH * outW;
  size_t patchSize = inChannels * kH * kW;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)batch * positions * patchSize;
  if (idx >= total) {
    return;
  }

  size_t row = idx / patchSize;
  size_t patchIdx = idx % patchSize;
  size_t batchIdx = row / positions;
  size_t posIdx = row % positions;
  size_t outY = posIdx / outW;
  size_t outX = posIdx % outW;
  size_t kernelArea = kH * kW;
  size_t channel = patchIdx / kernelArea;
  size_t kernelOffset = patchIdx % kernelArea;
  size_t kernelY = kernelOffset / kW;
  size_t kernelX = kernelOffset % kW;
  size_t inY = outY * stride + kernelY;
  size_t inX = outX * stride + kernelX;

  size_t destIdx =
      ((((size_t)batchIdx * h + inY) * w + inX) * inChannels) + channel;
  atomicAdd(&dest[destIdx], colBuffer[idx]);
}

template <typename T>
static Result
launchIm2colKernel(const void *input, size_t batch, size_t inChannels, size_t h,
                   size_t w, size_t kH, size_t kW, u8 stride, void *colBuffer) {
  size_t outH = (h - kH) / stride + 1;
  size_t outW = (w - kW) / stride + 1;
  dim3 grid((unsigned int)outW, (unsigned int)outH, (unsigned int)batch);
  int threadsPerBlock = 256;
  size_t patchSize = (size_t)inChannels * kH * kW;
  if (patchSize < (size_t)threadsPerBlock) {
    threadsPerBlock = (int)patchSize;
  }
  if (threadsPerBlock < 32) {
    threadsPerBlock = 32;
  }

  im2colNhwcKernel<<<grid, threadsPerBlock>>>((const T *)input, batch,
                                              inChannels, h, w, kH, kW, stride,
                                              outH, outW, (T *)colBuffer);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename T>
static Result launchCol2imKernel(void *dest, const void *colBuffer,
                                 size_t batch, size_t inChannels, size_t h,
                                 size_t w, size_t kH, size_t kW, u8 stride) {
  size_t outH = (h - kH) / stride + 1;
  size_t outW = (w - kW) / stride + 1;
  size_t total = (size_t)batch * outH * outW * inChannels * kH * kW;
  int threadsPerBlock = 256;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  col2imKernel<<<blocks, threadsPerBlock>>>((T *)dest, (const T *)colBuffer,
                                            batch, inChannels, h, w, kH, kW,
                                            stride, outH, outW);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaIm2col(shapes_Dtype dtype, const void *input,
                                size_t batch, size_t inChannels, size_t h,
                                size_t w, size_t kH, size_t kW, u8 stride,
                                void *colBuffer) {
  switch (dtype) {
  case F32: {
    if (kH == 3 && kW == 3) {
      size_t outH = (h - kH) / stride + 1;
      size_t outW = (w - kW) / stride + 1;
      int threadsPerBlock = 256;
      if ((int)inChannels < threadsPerBlock) {
        threadsPerBlock = (int)inChannels;
      }
      if (threadsPerBlock < 32) {
        threadsPerBlock = 32;
      }

      dim3 grid((unsigned int)outW, (unsigned int)outH, (unsigned int)batch);
      im2colNhwcF32Kernel3x3<<<grid, threadsPerBlock>>>(
          (const f32 *)input, batch, inChannels, h, w, stride, outH, outW,
          (f32 *)colBuffer);

      cudaError_t launchError = cudaGetLastError();
      PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                        cudaGetErrorString(launchError));
      return OK;
    }
    return launchIm2colKernel<f32>(input, batch, inChannels, h, w, kH, kW,
                                   stride, colBuffer);
  }
  case F64:
    return launchIm2colKernel<f64>(input, batch, inChannels, h, w, kH, kW,
                                   stride, colBuffer);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaCol2imAccumulate(shapes_Dtype dtype, void *dest,
                                          const void *colBuffer, size_t batch,
                                          size_t inChannels, size_t h, size_t w,
                                          size_t kH, size_t kW, u8 stride) {
  switch (dtype) {
  case F32:
    return launchCol2imKernel<f32>(dest, colBuffer, batch, inChannels, h, w, kH,
                                   kW, stride);
  case F64:
    return launchCol2imKernel<f64>(dest, colBuffer, batch, inChannels, h, w, kH,
                                   kW, stride);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
