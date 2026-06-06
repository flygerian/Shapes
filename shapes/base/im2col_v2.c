#include "result.h"
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "types.h"
#include "shapes_internal.h"
#include "shapes_common_types.h"

Tensor *im2colF32(shapes_Context *ctx, Tensor *t, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride) {
  shapes_dim_t batch = t->shape.dims[0];
  shapes_dim_t height = t->shape.dims[1];
  shapes_dim_t width = t->shape.dims[2];
  shapes_dim_t numInputChannels = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f32 *input = t->values;

  shapes_dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  shapes_dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  shapes_dim_t colBufferDims[2] = {batch * outputChannelHeight * outputChannelWidth, numInputChannels * kernelSize};
  shapes_Dim colBufferShape = {.dims = colBufferDims, .numOfDims = 2};
  Tensor *colBuffer = olib_Allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(colBuffer == NULL, ALLOCATION_FAILED);
  *colBuffer = t_Zeros(ctx, colBufferShape, F32);
  PANIC_IF((colBuffer == NULL || colBuffer->values == NULL), ALLOCATION_FAILED);

  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    Result result = shapescuda_Im2col(t->dtype, t->values, batch, numInputChannels, height, width, kernelHeight, kernelWidth, stride, colBuffer->values);
    PANIC_IF(result != OK, result);

    return colBuffer;
  }

  f32 *colBufferValues = colBuffer->values;

  size_t currRowNum = 0;
  for (shapes_dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (shapes_dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (shapes_dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (shapes_dim_t channel = 0; channel < numInputChannels; channel++) {
          shapes_dim_t yPos = kernelYStep * stride;
          shapes_dim_t xPos = kernelXStep * stride;

          for (shapes_dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f32 *rowStart = colBufferValues + (currRowNum * numInputChannels * kernelSize);
            f32 *currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            for (shapes_dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              shapes_dim_t inputIdx = (((iBatch * height + (yPos + kRow)) * width + (xPos + kCol)) * numInputChannels) + channel;
              currentRowPosition[kCol] = input[inputIdx];
            }
          }
        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}

Tensor *im2colF64(shapes_Context *ctx, Tensor *t, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride) {
  shapes_dim_t batch = t->shape.dims[0];
  shapes_dim_t height = t->shape.dims[1];
  shapes_dim_t width = t->shape.dims[2];
  shapes_dim_t numInputChannels = t->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f64 *input = t->values;

  shapes_dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  shapes_dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;
  size_t dtypeByteSize = getBytesForDtype(t->dtype);

  shapes_dim_t colBufferDims[2] = {batch * outputChannelHeight * outputChannelWidth, numInputChannels * kernelSize};
  shapes_Dim colBufferShape = {.dims = colBufferDims, .numOfDims = 2};
  Tensor *colBuffer = olib_Allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(colBuffer == NULL, ALLOCATION_FAILED);
  *colBuffer = t_Zeros(ctx, colBufferShape, F64);
  PANIC_IF((colBuffer == NULL || colBuffer->values == NULL), ALLOCATION_FAILED);

  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
    Result result = shapescuda_Im2col(t->dtype, t->values, batch, numInputChannels, height, width, kernelHeight, kernelWidth, stride, colBuffer->values);
    PANIC_IF(result != OK, result);
    return colBuffer;
  }

  f64 *colBufferValues = colBuffer->values;

  size_t currRowNum = 0;
  for (shapes_dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (shapes_dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (shapes_dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (shapes_dim_t channel = 0; channel < numInputChannels; channel++) {
          shapes_dim_t yPos = kernelYStep * stride;
          shapes_dim_t xPos = kernelXStep * stride;

          for (shapes_dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f64 *rowStart = colBufferValues + (currRowNum * numInputChannels * kernelSize);
            f64 *currentRowPosition = rowStart + (channel * kernelSize) + (kRow * kernelWidth);
            for (shapes_dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              shapes_dim_t inputIdx = (((iBatch * height + (yPos + kRow)) * width + (xPos + kCol)) * numInputChannels) + channel;
              currentRowPosition[kCol] = input[inputIdx];
            }
          }
        }

        currRowNum += 1;
      }
    }
  }

  return colBuffer;
}

void col2imAccumulateF32(Tensor *dInput, f32 *dColBuffer, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride) {
  shapes_dim_t batch = dInput->shape.dims[0];
  shapes_dim_t height = dInput->shape.dims[1];
  shapes_dim_t width = dInput->shape.dims[2];
  shapes_dim_t numInputChannels = dInput->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f32 *input = dInput->values;

  if (dInput->context->device->type == CUDA) {
    Result res = shapescuda_Col2imAccumulate(dInput->dtype, dInput->values, dColBuffer, batch, numInputChannels, height, width, kernelHeight, kernelWidth, stride);
    PANIC_IF(res != OK, res);
    return;
  }

  shapes_dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  shapes_dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  size_t currRowNum = 0;
  for (shapes_dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (shapes_dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (shapes_dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (shapes_dim_t channel = 0; channel < numInputChannels; channel++) {
          shapes_dim_t yPos = kernelYStep * stride;
          shapes_dim_t xPos = kernelXStep * stride;

          for (shapes_dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f32 *dColBufferRowStart = dColBuffer + (currRowNum * numInputChannels * kernelSize);
            f32 *currentDColRowPosition = dColBufferRowStart + (channel * kernelSize) + (kRow * kernelWidth);
            for (shapes_dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              shapes_dim_t destIdx = (((iBatch * height + (yPos + kRow)) * width + (xPos + kCol)) * numInputChannels) + channel;
              input[destIdx] += currentDColRowPosition[kCol];
            }
          }
        }
        currRowNum += 1;
      }
    }
  }
}

void col2imAccumulateF64(Tensor *dInput, f64 *dColBuffer, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride) {
  shapes_dim_t batch = dInput->shape.dims[0];
  shapes_dim_t height = dInput->shape.dims[1];
  shapes_dim_t width = dInput->shape.dims[2];
  shapes_dim_t numInputChannels = dInput->shape.dims[3];
  size_t kernelSize = (kernelHeight * kernelWidth);
  f64 *input = dInput->values;

  if (dInput->context->device->type == CUDA) {
    Result res = shapescuda_Col2imAccumulate(dInput->dtype, dInput->values, dColBuffer, batch, numInputChannels, height, width, kernelHeight, kernelWidth, stride);
    PANIC_IF(res != OK, res);
  }

  shapes_dim_t outputChannelHeight = (height - kernelHeight) / stride + 1;
  shapes_dim_t outputChannelWidth = (width - kernelWidth) / stride + 1;

  size_t currRowNum = 0;
  for (shapes_dim_t iBatch = 0; iBatch < batch; iBatch++) {
    for (shapes_dim_t kernelYStep = 0; kernelYStep < outputChannelHeight; kernelYStep++) {
      for (shapes_dim_t kernelXStep = 0; kernelXStep < outputChannelWidth; kernelXStep++) {
        for (shapes_dim_t channel = 0; channel < numInputChannels; channel++) {
          shapes_dim_t yPos = kernelYStep * stride;
          shapes_dim_t xPos = kernelXStep * stride;

          for (shapes_dim_t kRow = 0; kRow < kernelHeight; kRow++) {
            f64 *dColBufferRowStart = dColBuffer + (currRowNum * numInputChannels * kernelSize);
            f64 *currentDColRowPosition = dColBufferRowStart + (channel * kernelSize) + (kRow * kernelWidth);
            for (shapes_dim_t kCol = 0; kCol < kernelWidth; kCol++) {
              shapes_dim_t destIdx = (((iBatch * height + (yPos + kRow)) * width + (xPos + kCol)) * numInputChannels) + channel;
              input[destIdx] += currentDColRowPosition[kCol];
            }
          }
        }
        currRowNum += 1;
      }
    }
  }
}
