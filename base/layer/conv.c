#include "common.h"
#include "memory.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include <stddef.h>
#include <string.h>

Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Dim kernelShape, Tensor *t, Tensor *dest) {
  if (t == NULL || dest == NULL || ctx == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(t)) {
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

Result Conv2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride,
                      Tensor *dX, Tensor *dKernels) {
  if (isInvalidTensor(x) || isInvalidTensor(kernels) || isInvalidTensor(gradOut) || dX == NULL ||
      dKernels == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(x) || isNotFloatType(kernels) || isNotFloatType(gradOut)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4 || kernels->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (x->dtype != kernels->dtype || x->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t inChannels = x->shape.dims[1];
  dim_t h = x->shape.dims[2];
  dim_t w = x->shape.dims[3];
  dim_t outChannels = kernels->shape.dims[0];
  dim_t kernelInChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels || h < kH || w < kW) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - kH) / stride + 1;
  dim_t outW = (w - kW) / stride + 1;

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outChannels ||
      gradOut->shape.dims[2] != outH || gradOut->shape.dims[3] != outW) {
    return ERR_DIM_MISMATCH;
  }

  Result res = initTensorLike(ctx, dX, x, x->dtype);
  if (res != OK) {
    return res;
  }
  res = initTensorLike(ctx, dKernels, kernels, kernels->dtype);
  if (res != OK) {
    return res;
  }

  memset(dX->values, 0, dX->size * getBytesForDtype(dX->dtype));
  memset(dKernels->values, 0, dKernels->size * getBytesForDtype(dKernels->dtype));

  if (x->dtype == F64) {
    f64 *xValues = x->values;
    f64 *kernelValues = kernels->values;
    f64 *gradValues = gradOut->values;
    f64 *dxValues = dX->values;
    f64 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oc = 0; oc < outChannels; oc++) {
        for (dim_t oh = 0; oh < outH; oh++) {
          for (dim_t ow = 0; ow < outW; ow++) {
            dim_t gradIdx = (((b * outChannels + oc) * outH) + oh) * outW + ow;
            f64 grad = gradValues[gradIdx];
            dim_t inY = oh * stride;
            dim_t inX = ow * stride;

            for (dim_t ic = 0; ic < inChannels; ic++) {
              dim_t kernelBase = ((oc * inChannels + ic) * kH) * kW;
              dim_t inputBase = ((b * inChannels + ic) * h) * w;

              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t inputIdx = inputBase + (inY + ky) * w + (inX + kx);
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  } else {
    f32 *xValues = x->values;
    f32 *kernelValues = kernels->values;
    f32 *gradValues = gradOut->values;
    f32 *dxValues = dX->values;
    f32 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t oc = 0; oc < outChannels; oc++) {
        for (dim_t oh = 0; oh < outH; oh++) {
          for (dim_t ow = 0; ow < outW; ow++) {
            dim_t gradIdx = (((b * outChannels + oc) * outH) + oh) * outW + ow;
            f32 grad = gradValues[gradIdx];
            dim_t inY = oh * stride;
            dim_t inX = ow * stride;

            for (dim_t ic = 0; ic < inChannels; ic++) {
              dim_t kernelBase = ((oc * inChannels + ic) * kH) * kW;
              dim_t inputBase = ((b * inChannels + ic) * h) * w;

              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t inputIdx = inputBase + (inY + ky) * w + (inX + kx);
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  }

  return OK;
}

Result ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride,
                       Tensor *kernels, Dim kernelShape, Tensor *t, Tensor *dest) {
  if (t == NULL || dest == NULL || ctx == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(t)) {
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

  if (kH == 0 || kW == 0 || c != inChannels) {
    return ERR_DIM_MISMATCH;
  }

  if (kernels->shape.numOfDims < 4) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->shape.dims[0] != inChannels || kernels->shape.dims[1] != outChannels ||
      kernels->shape.dims[2] != kH || kernels->shape.dims[3] != kW) {
    return ERR_DIM_MISMATCH;
  }
  if (kernels->dtype != t->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  Result res = init4DTensor(ctx, dest, batch, outChannels, outH, outW, t->dtype);
  if (res != OK) {
    return res;
  }

  memset(dest->values, 0, dest->size * getBytesForDtype(dest->dtype));

  if (t->dtype == F64) {
    f64 *input = t->values;
    f64 *kernelValues = kernels->values;
    f64 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            f64 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx = (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  outValues[outIdx] += inputValue * kernelValues[kernelIdx];
                }
              }
            }
          }
        }
      }
    }
  } else {
    f32 *input = t->values;
    f32 *kernelValues = kernels->values;
    f32 *outValues = dest->values;

    for (dim_t b = 0; b < batch; b++) {
      for (size_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            f32 inputValue = input[inputIdx];
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (size_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t outIdx = (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  outValues[outIdx] += inputValue * kernelValues[kernelIdx];
                }
              }
            }
          }
        }
      }
    }
  }

  return OK;
}

Result ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride,
                               Tensor *dX, Tensor *dKernels) {
  if (isInvalidTensor(x) || isInvalidTensor(kernels) || isInvalidTensor(gradOut) || dX == NULL ||
      dKernels == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (stride == 0) {
    return ERR_CONV2D_KERNEL_STRIDE_ZERO;
  }

  if (isNotFloatType(x) || isNotFloatType(kernels) || isNotFloatType(gradOut)) {
    return ERR_CONV2D_KERNEL_NOT_FLOAT;
  }

  if (x->shape.numOfDims != 4 || kernels->shape.numOfDims != 4 || gradOut->shape.numOfDims != 4) {
    return ERR_CONV2D_INVALID_NUM_TENSOR_DIM;
  }

  if (x->dtype != kernels->dtype || x->dtype != gradOut->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t batch = x->shape.dims[0];
  dim_t inChannels = x->shape.dims[1];
  dim_t h = x->shape.dims[2];
  dim_t w = x->shape.dims[3];
  dim_t kernelInChannels = kernels->shape.dims[0];
  dim_t outChannels = kernels->shape.dims[1];
  dim_t kH = kernels->shape.dims[2];
  dim_t kW = kernels->shape.dims[3];

  if (inChannels != kernelInChannels) {
    return ERR_DIM_MISMATCH;
  }

  dim_t outH = (h - 1) * stride + kH;
  dim_t outW = (w - 1) * stride + kW;

  if (gradOut->shape.dims[0] != batch || gradOut->shape.dims[1] != outChannels ||
      gradOut->shape.dims[2] != outH || gradOut->shape.dims[3] != outW) {
    return ERR_DIM_MISMATCH;
  }

  Result res = initTensorLike(ctx, dX, x, x->dtype);
  if (res != OK) {
    return res;
  }
  res = initTensorLike(ctx, dKernels, kernels, kernels->dtype);
  if (res != OK) {
    return res;
  }

  memset(dX->values, 0, dX->size * getBytesForDtype(dX->dtype));
  memset(dKernels->values, 0, dKernels->size * getBytesForDtype(dKernels->dtype));

  if (x->dtype == F64) {
    f64 *xValues = x->values;
    f64 *kernelValues = kernels->values;
    f64 *gradValues = gradOut->values;
    f64 *dxValues = dX->values;
    f64 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  f64 grad = gradValues[gradIdx];
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  } else {
    f32 *xValues = x->values;
    f32 *kernelValues = kernels->values;
    f32 *gradValues = gradOut->values;
    f32 *dxValues = dX->values;
    f32 *dKernelValues = dKernels->values;

    for (dim_t b = 0; b < batch; b++) {
      for (dim_t ic = 0; ic < inChannels; ic++) {
        for (dim_t ih = 0; ih < h; ih++) {
          for (dim_t iw = 0; iw < w; iw++) {
            dim_t inputIdx = (((b * inChannels + ic) * h) + ih) * w + iw;
            dim_t outY = ih * stride;
            dim_t outX = iw * stride;

            for (dim_t oc = 0; oc < outChannels; oc++) {
              dim_t kernelBase = ((ic * outChannels + oc) * kH) * kW;
              for (dim_t ky = 0; ky < kH; ky++) {
                for (dim_t kx = 0; kx < kW; kx++) {
                  dim_t gradIdx =
                      (((b * outChannels + oc) * outH) + (outY + ky)) * outW + outX + kx;
                  dim_t kernelIdx = kernelBase + ky * kW + kx;
                  f32 grad = gradValues[gradIdx];
                  dxValues[inputIdx] += grad * kernelValues[kernelIdx];
                  dKernelValues[kernelIdx] += xValues[inputIdx] * grad;
                }
              }
            }
          }
        }
      }
    }
  }

  return OK;
}
