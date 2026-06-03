#include "result.h"
#include "shapes_internal.h"
#include "shapes.h"
#include <string.h>
#include <stdlib.h>

static inline dim_t adaptivePoolStart(dim_t outIdx, dim_t inputSize, dim_t outputSize) {
  return (outIdx * inputSize) / outputSize;
}

static inline dim_t adaptivePoolEnd(dim_t outIdx, dim_t inputSize, dim_t outputSize) {
  return ((outIdx + 1) * inputSize + outputSize - 1) / outputSize;
}

static Result clearPoolTarget(Context *ctx, Tensor *target) {
  if (ctx == NULL || target == NULL || target->values == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  return clearTensorValues(target);
}

static Result maxPool2dImpl(Context *ctx, Tensor *x, Dim kernelShape, u8 stride, Tensor *dest, Tensor *indices) {
  if (ctx == NULL || dest == NULL || x == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(x)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (kernelShape.numOfDims != 2 || kernelShape.dims == NULL) {
    return ERR_CONV2D_KERNEL_NOT_2D;
  }

  dim_t kH = kernelShape.dims[0];
  dim_t kW = kernelShape.dims[1];
  dim_t batch = x->shape.dims[0];
  dim_t h = x->shape.dims[1];
  dim_t w = x->shape.dims[2];
  dim_t channels = x->shape.dims[3];

  if (kH == 0 || kW == 0 || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  Tensor *xContig = materializeTensorOnContext(ctx, x);
  Result res = OK;

  Tensor *createdDest = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDest == NULL, ALLOCATION_FAILED);
  *createdDest = t_Zeros(ctx, SHAPE4D(batch, outH, outW, channels), x->dtype);
  PANIC_IF(createdDest == NULL, ERR_OUT_OF_MEMORY);
  *dest = *createdDest;

  if (indices != NULL) {
    Tensor *createdIndices = allocate(ctx->memory, sizeof(Tensor));
    PANIC_IF(createdIndices == NULL, ALLOCATION_FAILED);
    *createdIndices = t_Zeros(ctx, SHAPE4D(batch, outH, outW, channels), U64);
    PANIC_IF(createdIndices == NULL, ERR_OUT_OF_MEMORY);
    *indices = *createdIndices;
  }

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    if (indices != NULL) {
      res = runCudaMaxPool2dWithIndices(x->dtype, xContig->values, batch, channels, h, w, kH, kW, stride, dest->values, indices->values);
    } else {
      res = runCudaMaxPool2d(x->dtype, xContig->values, batch, channels, h, w, kH, kW, stride, dest->values);
    }

    return res;
  }

  if (x->dtype == F64) {
    f64 *input = xContig->values;
    f64 *output = dest->values;
    u64 *argmax = indices != NULL ? indices->values : NULL;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = oh * stride;
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = ow * stride;
          for (dim_t c = 0; c < channels; c++) {
            size_t maxIdx = (((b * h + startY) * w + startX) * channels) + c;
            f64 maxValue = input[maxIdx];
            for (dim_t ky = 0; ky < kH; ky++) {
              for (dim_t kx = 0; kx < kW; kx++) {
                size_t inputIdx = (((b * h + (startY + ky)) * w + (startX + kx)) * channels) + c;
                f64 candidate = input[inputIdx];
                if (candidate > maxValue) {
                  maxValue = candidate;
                  maxIdx = inputIdx;
                }
              }
            }
            size_t outIdx = (((b * outH + oh) * outW + ow) * channels) + c;
            output[outIdx] = maxValue;
            if (argmax != NULL) {
              argmax[outIdx] = (u64)maxIdx;
            }
          }
        }
      }
    }
  } else {
    f32 *input = xContig->values;
    f32 *output = dest->values;
    u64 *argmax = indices != NULL ? indices->values : NULL;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = oh * stride;
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = ow * stride;
          for (dim_t c = 0; c < channels; c++) {
            size_t maxIdx = (((b * h + startY) * w + startX) * channels) + c;
            f32 maxValue = input[maxIdx];
            for (dim_t ky = 0; ky < kH; ky++) {
              for (dim_t kx = 0; kx < kW; kx++) {
                size_t inputIdx = (((b * h + (startY + ky)) * w + (startX + kx)) * channels) + c;
                f32 candidate = input[inputIdx];
                if (candidate > maxValue) {
                  maxValue = candidate;
                  maxIdx = inputIdx;
                }
              }
            }
            size_t outIdx = (((b * outH + oh) * outW + ow) * channels) + c;
            output[outIdx] = maxValue;
            if (argmax != NULL) {
              argmax[outIdx] = (u64)maxIdx;
            }
          }
        }
      }
    }
  }

  return OK;
}

Result shapes_layer_MaxPool2d(Context *ctx, Tensor *x, Dim kernelShape, u8 stride, Tensor *dest) {
  return maxPool2dImpl(ctx, x, kernelShape, stride, dest, NULL);
}

Result shapes_layer_MaxPool2dWithIndices(Context *ctx, Tensor *x, Dim kernelShape, u8 stride, Tensor *dest, Tensor *indices) {
  return maxPool2dImpl(ctx, x, kernelShape, stride, dest, indices);
}

Result shapes_layer_MaxPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, Dim kernelShape, u8 stride, Tensor *dX) {
  if (ctx == NULL || dX == NULL || isInvalidTensor(x) || isInvalidTensor(gradOut)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(x) || isNotFloatType(gradOut)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (kernelShape.numOfDims != 2 || kernelShape.dims == NULL) {
    return ERR_CONV2D_KERNEL_NOT_2D;
  }

  if (x->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t kH = kernelShape.dims[0];
  dim_t kW = kernelShape.dims[1];
  dim_t batch = x->shape.dims[0];
  dim_t h = x->shape.dims[1];
  dim_t w = x->shape.dims[2];
  dim_t channels = x->shape.dims[3];

  if (kH == 0 || kW == 0 || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;
  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outH || gradOut->shape.dims[2] != outW || gradOut->shape.dims[3] != channels) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *xContig = materializeTensorOnContext(ctx, x);
  Tensor *gradContig = materializeTensorOnContext(ctx, gradOut);

  Tensor *createdDX = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *createdDX = t_Zeros(ctx, x->shape, x->dtype);
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *dX = *createdDX;
  Result res = OK;

  res = clearPoolTarget(ctx, dX);
  PANIC_IF(res != OK, res);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    res = runCudaMaxPool2dBackward(x->dtype, xContig->values, gradContig->values, batch, channels, h, w, kH, kW, stride, dX->values);
    return res;
  }

  if (x->dtype == F64) {
    f64 *input = xContig->values;
    f64 *grad = gradContig->values;
    f64 *dx = dX->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = oh * stride;
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = ow * stride;
          for (dim_t c = 0; c < channels; c++) {
            dim_t maxIdx = (((b * h + startY) * w + startX) * channels) + c;
            f64 maxValue = input[maxIdx];
            for (dim_t ky = 0; ky < kH; ky++) {
              for (dim_t kx = 0; kx < kW; kx++) {
                dim_t inputIdx = (((b * h + (startY + ky)) * w + (startX + kx)) * channels) + c;
                f64 candidate = input[inputIdx];
                if (candidate > maxValue) {
                  maxValue = candidate;
                  maxIdx = inputIdx;
                }
              }
            }
            dx[maxIdx] += grad[(((b * outH + oh) * outW + ow) * channels) + c];
          }
        }
      }
    }
  } else {
    f32 *input = xContig->values;
    f32 *grad = gradContig->values;
    f32 *dx = dX->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = oh * stride;
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = ow * stride;
          for (dim_t c = 0; c < channels; c++) {
            dim_t maxIdx = (((b * h + startY) * w + startX) * channels) + c;
            f32 maxValue = input[maxIdx];
            for (dim_t ky = 0; ky < kH; ky++) {
              for (dim_t kx = 0; kx < kW; kx++) {
                dim_t inputIdx = (((b * h + (startY + ky)) * w + (startX + kx)) * channels) + c;
                f32 candidate = input[inputIdx];
                if (candidate > maxValue) {
                  maxValue = candidate;
                  maxIdx = inputIdx;
                }
              }
            }
            dx[maxIdx] += grad[(((b * outH + oh) * outW + ow) * channels) + c];
          }
        }
      }
    }
  }

  return OK;
}

Result shapes_layer_MaxPool2dBackwardWithIndices(Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices, Tensor *dX) {
  if (ctx == NULL || dX == NULL || isInvalidTensor(x) || isInvalidTensor(gradOut) || isInvalidTensor(indices)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isNotFloatType(x) || isNotFloatType(gradOut) || indices->dtype != U64) {
    return ERR_DTYPE_MISMATCH;
  }

  if (x->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4 || indices->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (x->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t channels = x->shape.dims[3];

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[3] != channels || indices->shape.dims[0] != batch || indices->shape.dims[3] != channels ||
      gradOut->shape.dims[1] != indices->shape.dims[1] || gradOut->shape.dims[2] != indices->shape.dims[2]) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *gradContig = materializeTensorOnContext(ctx, gradOut);
  Tensor *indicesContig = materializeTensorOnContext(ctx, indices);

  Tensor *createdDX = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *createdDX = t_Zeros(ctx, x->shape, x->dtype);
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *dX = *createdDX;
  Result res = OK;

  res = clearPoolTarget(ctx, dX);
  PANIC_IF(res != OK, res);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    res = runCudaMaxPool2dBackwardWithIndices(x->dtype, gradContig->values, indicesContig->values, gradContig->size, dX->values);
    return res;
  }

  if (x->dtype == F64) {
    f64 *grad = gradContig->values;
    f64 *dx = dX->values;
    u64 *argmax = indicesContig->values;
    for (tensor_size_t i = 0; i < gradContig->size; i++) {
      dx[argmax[i]] += grad[i];
    }
  } else {
    f32 *grad = gradContig->values;
    f32 *dx = dX->values;
    u64 *argmax = indicesContig->values;
    for (tensor_size_t i = 0; i < gradContig->size; i++) {
      dx[argmax[i]] += grad[i];
    }
  }

  return OK;
}

Result shapes_layer_AdaptiveAvgPool2d(Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor *dest) {
  if (ctx == NULL || dest == NULL || x == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isNotFloatType(x)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (outH == 0 || outW == 0) {
    return ERR_DIM_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t h = x->shape.dims[1];
  dim_t w = x->shape.dims[2];
  dim_t channels = x->shape.dims[3];

  Tensor *xContig = materializeTensorOnContext(ctx, x);
  Result res = OK;

  Tensor *createdDest = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDest == NULL, ALLOCATION_FAILED);
  *createdDest = t_Zeros(ctx, SHAPE4D(batch, outH, outW, channels), x->dtype);
  PANIC_IF(createdDest == NULL, ERR_OUT_OF_MEMORY);
  *dest = *createdDest;

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    res = runCudaAdaptiveAvgPool2d(x->dtype, xContig->values, batch, channels, h, w, outH, outW, dest->values);
    return res;
  }

  if (x->dtype == F64) {
    f64 *input = xContig->values;
    f64 *output = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = adaptivePoolStart(oh, h, outH);
        dim_t endY = adaptivePoolEnd(oh, h, outH);
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = adaptivePoolStart(ow, w, outW);
          dim_t endX = adaptivePoolEnd(ow, w, outW);
          dim_t count = (endY - startY) * (endX - startX);
          for (dim_t c = 0; c < channels; c++) {
            f64 sum = 0.0;
            for (dim_t iy = startY; iy < endY; iy++) {
              for (dim_t ix = startX; ix < endX; ix++) {
                sum += input[(((b * h + iy) * w + ix) * channels) + c];
              }
            }

            output[(((b * outH + oh) * outW + ow) * channels) + c] = sum / (f64)count;
          }
        }
      }
    }
  } else {
    f32 *input = xContig->values;
    f32 *output = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = adaptivePoolStart(oh, h, outH);
        dim_t endY = adaptivePoolEnd(oh, h, outH);
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = adaptivePoolStart(ow, w, outW);
          dim_t endX = adaptivePoolEnd(ow, w, outW);
          dim_t count = (endY - startY) * (endX - startX);
          for (dim_t c = 0; c < channels; c++) {
            f32 sum = 0.0f;
            for (dim_t iy = startY; iy < endY; iy++) {
              for (dim_t ix = startX; ix < endX; ix++) {
                sum += input[(((b * h + iy) * w + ix) * channels) + c];
              }
            }

            output[(((b * outH + oh) * outW + ow) * channels) + c] = sum / (f32)count;
          }
        }
      }
    }
  }

  return OK;
}

Result shapes_layer_AdaptiveAvgPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH, dim_t outW, Tensor *dX) {
  if (ctx == NULL || dX == NULL || isInvalidTensor(x) || isInvalidTensor(gradOut)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (isNotFloatType(x) || isNotFloatType(gradOut)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (x->dtype != gradOut->dtype || outH == 0 || outW == 0) {
    return ERR_DIM_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t h = x->shape.dims[1];
  dim_t w = x->shape.dims[2];
  dim_t channels = x->shape.dims[3];

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outH || gradOut->shape.dims[2] != outW || gradOut->shape.dims[3] != channels) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *gradContig = materializeTensorOnContext(ctx, gradOut);

  Tensor *createdDX = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *createdDX = t_Zeros(ctx, x->shape, x->dtype);
  PANIC_IF(createdDX == NULL, ALLOCATION_FAILED);
  *dX = *createdDX;
  Result res = OK;

  res = clearPoolTarget(ctx, dX);
  PANIC_IF(res != OK, res);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    res = runCudaAdaptiveAvgPool2dBackward(x->dtype, gradContig->values, batch, channels, h, w, outH, outW, dX->values);
    return res;
  }

  if (x->dtype == F64) {
    f64 *grad = gradContig->values;
    f64 *dx = dX->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = adaptivePoolStart(oh, h, outH);
        dim_t endY = adaptivePoolEnd(oh, h, outH);
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = adaptivePoolStart(ow, w, outW);
          dim_t endX = adaptivePoolEnd(ow, w, outW);
          dim_t count = (endY - startY) * (endX - startX);
          for (dim_t c = 0; c < channels; c++) {
            f64 scaledGrad = grad[(((b * outH + oh) * outW + ow) * channels) + c] / (f64)count;
            for (dim_t iy = startY; iy < endY; iy++) {
              for (dim_t ix = startX; ix < endX; ix++) {
                dx[(((b * h + iy) * w + ix) * channels) + c] += scaledGrad;
              }
            }
          }
        }
      }
    }
  } else {
    f32 *grad = gradContig->values;
    f32 *dx = dX->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oh = 0; oh < outH; oh++) {
        dim_t startY = adaptivePoolStart(oh, h, outH);
        dim_t endY = adaptivePoolEnd(oh, h, outH);
        for (dim_t ow = 0; ow < outW; ow++) {
          dim_t startX = adaptivePoolStart(ow, w, outW);
          dim_t endX = adaptivePoolEnd(ow, w, outW);
          dim_t count = (endY - startY) * (endX - startX);
          for (dim_t c = 0; c < channels; c++) {
            f32 scaledGrad = grad[(((b * outH + oh) * outW + ow) * channels) + c] / (f32)count;
            for (dim_t iy = startY; iy < endY; iy++) {
              for (dim_t ix = startX; ix < endX; ix++) {
                dx[(((b * h + iy) * w + ix) * channels) + c] += scaledGrad;
              }
            }
          }
        }
      }
    }
  }

  return OK;
}
