#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include <stddef.h>

static Result init4DTensor(Context *ctx, Tensor *dest, dim_t d0, dim_t d1, dim_t d2, dim_t d3,
                           Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 4);
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * 4);

  dims[0] = d0;
  dims[1] = d1;
  dims[2] = d2;
  dims[3] = d3;

  Dim shape = {.dims = dims, .numOfDims = 4, .multipliers = multipliers};
  tensor_size_t size = calculateNumValuesAndMultipliers(shape, multipliers);

  *dest = (Tensor){.dtype = dtype,
                   .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Dim kernelShape, Tensor *t, Tensor *dest) {
  if (t == NULL || dest == NULL || ctx == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isFloatNotType(t)) {
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

  if (t->dtype == F64) {
    f64 *input = t->values;
    f64 *kernelValues = kernels->values;
    f64 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t oc = 0; oc < outChannels; oc++) {
        f64 *channelKernels = kernelValues + (oc * inChannels * kH * kW);

        for (dim_t ih = 0; ih < outH; ih++) {
          for (dim_t iw = 0; iw < outW; iw++) {
            dim_t currentPos = ih * stride * w + iw * stride;

            f64 valueAtPos = 0;
            for (size_t ic = 0; ic < inChannels; ic++) {
              f64 *values = input + ((b * inChannels + ic) * h * w);
              f64 *icKernel = channelKernels + (ic * kH * kW);
              for (size_t py = 0; py < kH; py++) {
                for (size_t px = 0; px < kW; px++) {
                  f64 val = values[currentPos + py * w + px];
                  f64 kVal = icKernel[py * kW + px];
                  valueAtPos += val * kVal;
                }
              }
            }

            outValues[(((b * outChannels + oc) * outH) + ih) * outW + iw] = valueAtPos;
          }
        }
      }
    }
  } else {
    f32 *input = t->values;
    f32 *kernelValues = kernels->values;
    f32 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t oc = 0; oc < outChannels; oc++) {
        f32 *channelKernels = kernelValues + (oc * inChannels * kH * kW);

        for (dim_t ih = 0; ih < outH; ih++) {
          for (dim_t iw = 0; iw < outW; iw++) {
            dim_t currentPos = ih * stride * w + iw * stride;

            f32 valueAtPos = 0;
            for (size_t ic = 0; ic < inChannels; ic++) {
              f32 *values = input + ((b * inChannels + ic) * h * w);
              f32 *icKernel = channelKernels + (ic * kH * kW);
              for (size_t py = 0; py < kH; py++) {
                for (size_t px = 0; px < kW; px++) {
                  f32 val = values[currentPos + py * w + px];
                  f32 kVal = icKernel[py * kW + px];
                  valueAtPos += val * kVal;
                }
              }
            }

            outValues[(((b * outChannels + oc) * outH) + ih) * outW + iw] = valueAtPos;
          }
        }
      }
    }
  }

  return OK;
}
