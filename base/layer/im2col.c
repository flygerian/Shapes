#include "im2col.h"

// Pack one NCHW input image into a 2D matrix so convolution can be expressed
// as GEMM. Each column is one sliding-window position, and each row selects one
// value within the flattened receptive field (channel, kernelY, kernelX).
void im2colNchwF32(const f32 *input, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                   u8 stride, dim_t outH, dim_t outW, f32 *colBuffer) {
  dim_t positions = outH * outW;

  for (dim_t ic = 0; ic < inChannels; ic++) {
    for (dim_t ky = 0; ky < kH; ky++) {
      for (dim_t kx = 0; kx < kW; kx++) {
        dim_t patchIdx = (ic * kH + ky) * kW + kx;
        dim_t colBase = patchIdx * positions;
        dim_t posIdx = 0;

        for (dim_t oh = 0; oh < outH; oh++) {
          dim_t inY = oh * stride + ky;
          for (dim_t ow = 0; ow < outW; ow++) {
            dim_t inX = ow * stride + kx;
            colBuffer[colBase + posIdx] = input[(ic * h + inY) * w + inX];
            posIdx++;
          }
        }
      }
    }
  }
}

// Same packing as im2colNchwF32, but for F64 inputs.
void im2colNchwF64(const f64 *input, dim_t inChannels, dim_t inputHeight, dim_t inputWidth, dim_t KernelHeight, dim_t kernelWidth,
                   u8 stride, dim_t outputHeight, dim_t outputWidth, f64 *colBuffer) {
  dim_t positions = outputHeight * outputWidth;

  for (dim_t inputChannelIdx = 0; inputChannelIdx < inChannels; inputChannelIdx++) {
    for (dim_t kernelYPosition = 0; kernelYPosition < KernelHeight; kernelYPosition++) {
      for (dim_t kernelXPosition = 0; kernelXPosition < kernelWidth; kernelXPosition++) {
        dim_t patchIdx = (inputChannelIdx * KernelHeight + kernelYPosition) * kernelWidth + kernelXPosition;
        dim_t colBase = patchIdx * positions;
        dim_t posIdx = 0;

        for (dim_t oh = 0; oh < outputHeight; oh++) {
          dim_t inY = oh * stride + kernelYPosition;
          for (dim_t ow = 0; ow < outputWidth; ow++) {
            dim_t inX = ow * stride + kernelXPosition;
            colBuffer[colBase + posIdx] = input[(inputChannelIdx * inputHeight + inY) * inputWidth + inX];
            posIdx++;
          }
        }
      }
    }
  }
}

// Scatter a packed gradient matrix back into NCHW image layout. Multiple
// sliding windows can touch the same input element, so this helper adds into
// dest instead of overwriting it.
void col2imNchwAddF32(const f32 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                      u8 stride, dim_t outH, dim_t outW, f32 *dest) {
  dim_t positions = outH * outW;

  for (dim_t ic = 0; ic < inChannels; ic++) {
    for (dim_t ky = 0; ky < kH; ky++) {
      for (dim_t kx = 0; kx < kW; kx++) {
        dim_t patchIdx = (ic * kH + ky) * kW + kx;
        dim_t colBase = patchIdx * positions;
        dim_t posIdx = 0;

        for (dim_t oh = 0; oh < outH; oh++) {
          dim_t inY = oh * stride + ky;
          for (dim_t ow = 0; ow < outW; ow++) {
            dim_t inX = ow * stride + kx;
            dest[(ic * h + inY) * w + inX] += colBuffer[colBase + posIdx];
            posIdx++;
          }
        }
      }
    }
  }
}

// Same scatter-add as col2imNchwAddF32, but for F64 buffers.
void col2imNchwAddF64(const f64 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                      u8 stride, dim_t outH, dim_t outW, f64 *dest) {
  dim_t positions = outH * outW;

  for (dim_t ic = 0; ic < inChannels; ic++) {
    for (dim_t ky = 0; ky < kH; ky++) {
      for (dim_t kx = 0; kx < kW; kx++) {
        dim_t patchIdx = (ic * kH + ky) * kW + kx;
        dim_t colBase = patchIdx * positions;
        dim_t posIdx = 0;

        for (dim_t oh = 0; oh < outH; oh++) {
          dim_t inY = oh * stride + ky;
          for (dim_t ow = 0; ow < outW; ow++) {
            dim_t inX = ow * stride + kx;
            dest[(ic * h + inY) * w + inX] += colBuffer[colBase + posIdx];
            posIdx++;
          }
        }
      }
    }
  }
}
