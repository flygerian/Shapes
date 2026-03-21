#include "common.h"
#include "im2col.h"
#include "tensor/tensor_internal.h"
#include <stddef.h>
#include <string.h>

Tensor *im2colF32(Context *ctx, Tensor *t, dim_t kernelHeight, dim_t kernelWidth, u8 stride) {
  dim_t batch = t->shape.dims[0];
  dim_t numInputChannels = t->shape.dims[1];
  dim_t height = t->shape.dims[2];
  dim_t width = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f32 *input = t->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;
  size_t dtypeByteSize = getBytesForDtype(t->dtype);

  dim_t colBufferDims[2] = {batch * outputChannelHeight * outputChannelWidth,
                            numInputChannels * kernelSize};
  Dim colBufferShape = {.dims = colBufferDims, .numOfDims = 2};
  Tensor *colBuffer = t_Zeros(ctx, colBufferShape, F32);
  f32 *colBufferValues = colBuffer->values;

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f32 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f32 *patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f32 *rowStart = colBufferValues + (currRowNum * numInputChannels * kernelSize);
            f32 *currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f32 *patchRowStart = patchStart + (kRow * width);
            memcpy(currentRowPosition, patchRowStart, dtypeByteSize * kernelWidth);
          }
        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}

Tensor *im2colF64(Context *ctx, Tensor *t, dim_t kernelHeight, dim_t kernelWidth, u8 stride) {
  dim_t batch = t->shape.dims[0];
  dim_t numInputChannels = t->shape.dims[1];
  dim_t height = t->shape.dims[2];
  dim_t width = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f64 *input = t->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;
  size_t dtypeByteSize = getBytesForDtype(t->dtype);

  dim_t colBufferDims[2] = {batch * outputChannelHeight * outputChannelWidth,
                            numInputChannels * kernelSize};
  Dim colBufferShape = {.dims = colBufferDims, .numOfDims = 2};
  Tensor *colBuffer = t_Zeros(ctx, colBufferShape, F64);
  f64 *colBufferValues = colBuffer->values;

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f64 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f64 *patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f64 *rowStart = colBufferValues + (currRowNum * numInputChannels * kernelSize);
            f64 *currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f64 *patchRowStart = patchStart + (kRow * width);
            memcpy(currentRowPosition, patchRowStart, dtypeByteSize * kernelWidth);
          }
        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}

void col2imAccumulateF32(Tensor *dInput, f32 *dColBuffer, dim_t kernelHeight, dim_t kernelWidth,
                         u8 stride) {
  dim_t batch = dInput->shape.dims[0];
  dim_t numInputChannels = dInput->shape.dims[1];
  dim_t height = dInput->shape.dims[2];
  dim_t width = dInput->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f32 *input = dInput->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f32 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f32 *patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f32 *dColBufferRowStart = dColBuffer + (currRowNum * numInputChannels * kernelSize);
            f32 *currentDColRowPosition =
                dColBufferRowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f32 *patchRowStart = patchStart + (kRow * width);
            for (dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              patchRowStart[kCol] += currentDColRowPosition[kCol];
            }
          }
        }
        currRowNum += 1;
      }
    }
  }
}

void col2imAccumulateF64(Tensor *dInput, f64 *dColBuffer, dim_t kernelHeight, dim_t kernelWidth,
                         u8 stride) {
  dim_t batch = dInput->shape.dims[0];
  dim_t numInputChannels = dInput->shape.dims[1];
  dim_t height = dInput->shape.dims[2];
  dim_t width = dInput->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f64 *input = dInput->values;

  dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  size_t currRowNum = 0;
  for (dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (dim_t channel = 0; channel < numInputChannels; channel++) {
          f64 *channelPos =
              input + ((iBatch * numInputChannels * height * width) + channel * height * width);

          dim_t yPos = kernelYStep * stride;
          dim_t xPos = kernelXStep * stride;

          f64 *patchStart = channelPos + ((yPos * width) + xPos);

          for (dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f64 *dColBufferRowStart = dColBuffer + (currRowNum * numInputChannels * kernelSize);
            f64 *currentDColRowPosition =
                dColBufferRowStart + (channel * kernelSize) + (kRow * kernelWidth);
            f64 *patchRowStart = patchStart + (kRow * width);
            for (dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              patchRowStart[kCol] += currentDColRowPosition[kCol];
            }
          }
        }
        currRowNum += 1;
      }
    }
  }
}
