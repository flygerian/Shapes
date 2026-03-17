#include "common.h"
#include "im2col.h"
#include "memory.h"
#include <stddef.h>
#include <string.h>

void *im2colF32(Context *ctx, Tensor* t, dim_t kernelHeight, dim_t kernelWidth, u8 stride) {
  dim_t batch = t->shape.dims[0];
  dim_t numInputChannels = t->shape.dims[1];
  dim_t height = t->shape.dims[2];
  dim_t width = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f32 *input = t->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;
  size_t dtypeByteSize = getBytesForDtype(t->dtype);
  size_t colBufferNumChannels = (outputChannelHeight * outputChannelWidth) * numInputChannels;

  size_t colBufferSize = batch * colBufferNumChannels * kernelSize;
  f32* colBuffer = allocate(ctx->memory, dtypeByteSize * colBufferSize);

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f32 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f32* patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f32* rowStart = colBuffer + (currRowNum * numInputChannels * kernelSize);
            f32* currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f32* patchRowStart = patchStart + (kRow * width);
            memcpy(currentRowPosition, patchRowStart, dtypeByteSize * kernelWidth);
          }

        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}

void *im2colF64(Context *ctx, Tensor* t,dim_t kernelHeight, dim_t kernelWidth, u8 stride) {
  dim_t batch = t->shape.dims[0];
  dim_t numInputChannels = t->shape.dims[1];
  dim_t height = t->shape.dims[2];
  dim_t width = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f64 *input = t->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;
  size_t dtypeByteSize = getBytesForDtype(t->dtype);
  size_t colBufferNumChannels = (outputChannelHeight * outputChannelWidth) * numInputChannels;

  size_t colBufferSize = batch * colBufferNumChannels * kernelSize;
  f64* colBuffer = allocate(ctx->memory, dtypeByteSize * colBufferSize);

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f64 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f64* patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f64* rowStart = colBuffer + (currRowNum * numInputChannels * kernelSize);
            f64* currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f64* patchRowStart = patchStart + (kRow * width);
            memcpy(currentRowPosition, patchRowStart, dtypeByteSize * kernelWidth);
          }
        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}
