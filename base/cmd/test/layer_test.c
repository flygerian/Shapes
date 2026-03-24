#include "common.h"
#include "test.h"
#include "../../shapes.h"
#include "../../layer/im2col.h"
#include "../../tensor/tensor_internal.h"
#include "cblas.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool hasCudaDevice(void) {
  int deviceCount = 0;
  return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

static Tensor create1DTensor(Context *ctx, dim_t size, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t));
  dims[0] = size;
  multipliers[0] = 1;

  Tensor t = {.dtype = dtype,
              .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
              .size = size,
              .shape = (Dim){.dims = dims, .numOfDims = 1, .multipliers = multipliers},
              .isView = false,
              .isContigous = true,
              .boundary = NULL};
  return t;
}

static Tensor createScalarTensor(Context *ctx, Dtype dtype) {
  Tensor t = {.context = ctx,
              .metadataMemory = ctx->memory,
              .values = allocate(ctx->memory, getBytesForDtype(dtype)),
              .boundary = NULL,
              .size = 1,
              .shape = (Dim){.dims = NULL, .numOfDims = 0, .multipliers = NULL},
              .dtype = dtype,
              .isView = false,
              .isContigous = true};
  return t;
}

static Tensor create2DTensor(Context *ctx, dim_t rows, dim_t cols, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 2);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * 2);
  dims[0] = rows;
  dims[1] = cols;
  multipliers[0] = cols;
  multipliers[1] = 1;

  tensor_size_t size = rows * cols;
  Tensor t = {.dtype = dtype,
              .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
              .size = size,
              .shape = (Dim){.dims = dims, .numOfDims = 2, .multipliers = multipliers},
              .isView = false,
              .isContigous = true,
              .boundary = NULL};
  return t;
}

static Tensor create3DTensor(Context *ctx, dim_t d0, dim_t d1, dim_t d2, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 3);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * 3);
  dims[0] = d0;
  dims[1] = d1;
  dims[2] = d2;
  multipliers[0] = d1 * d2;
  multipliers[1] = d2;
  multipliers[2] = 1;

  tensor_size_t size = d0 * d1 * d2;
  Tensor t = {.dtype = dtype,
              .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
              .size = size,
              .shape = (Dim){.dims = dims, .numOfDims = 3, .multipliers = multipliers},
              .isView = false,
              .isContigous = true,
              .boundary = NULL};
  return t;
}

static Tensor create4DTensor(Context *ctx, dim_t batch, dim_t channels, dim_t height, dim_t width,
                             Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 4);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * 4);
  dims[0] = batch;
  dims[1] = channels;
  dims[2] = height;
  dims[3] = width;
  multipliers[0] = channels * height * width;
  multipliers[1] = height * width;
  multipliers[2] = width;
  multipliers[3] = 1;

  tensor_size_t size = batch * channels * height * width;
  Tensor t = {.dtype = dtype,
              .values = allocate(ctx->memory, size * getBytesForDtype(dtype)),
              .size = size,
              .shape = (Dim){.dims = dims, .numOfDims = 4, .multipliers = multipliers},
              .isView = false,
              .isContigous = true,
              .boundary = NULL};
  return t;
}

static void assertMovedF32TensorClose(Context *srcCtx, Tensor *tensor, const f32 *expected,
                                      tensor_size_t size, f32 tolerance, const char *label) {
  Context cpuCtx = {.memory = srcCtx->memory};
  Result moveResult = moveTensor(srcCtx, &cpuCtx, tensor);
  ASSERT_EQ(moveResult, OK, label);

  f32 *values = tensor->values;
  for (tensor_size_t i = 0; i < size; i++) {
    ASSERT(fabsf(values[i] - expected[i]) < tolerance, label);
  }
}

static void assertScalarF32Close(Tensor *tensor, f32 expected, f32 tolerance, const char *label) {
  Value value;
  Result result = GetScalar(tensor, &value);
  ASSERT_EQ(result, OK, label);
  ASSERT(fabsf(value.as.f32 - expected) < tolerance, label);
}

static void test_dense_linear_forward_with_bias_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor x = create2DTensor(&ctx, 2, 3, F32);
  Tensor w = create2DTensor(&ctx, 2, 3, F32);
  Tensor b = create1DTensor(&ctx, 2, F32);
  Tensor out;

  f32 *xVals = x.values;
  f32 *wVals = w.values;
  f32 *bVals = b.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;
  xVals[4] = 5.0f;
  xVals[5] = 6.0f;

  wVals[0] = 1.0f;
  wVals[1] = 0.0f;
  wVals[2] = -1.0f;
  wVals[3] = 0.5f;
  wVals[4] = 1.0f;
  wVals[5] = 0.0f;

  bVals[0] = 0.1f;
  bVals[1] = -0.2f;

  Result r = DenseLinear(&ctx, &x, &w, &b, true, &out);
  ASSERT_EQ(r, OK, "DenseLinear should succeed");
  ASSERT_EQ(out.shape.numOfDims, 2, "Dense output should be 2D");
  ASSERT_EQ(out.shape.dims[0], 2, "Dense output rows should match input rows");
  ASSERT_EQ(out.shape.dims[1], 2, "Dense output cols should match weight output size");

  f32 *outVals = out.values;
  ASSERT(fabsf(outVals[0] - -1.9f) < 1e-5f, "Dense out[0,0] mismatch");
  ASSERT(fabsf(outVals[1] - 2.3f) < 1e-5f, "Dense out[0,1] mismatch");
  ASSERT(fabsf(outVals[2] - -1.9f) < 1e-5f, "Dense out[1,0] mismatch");
  ASSERT(fabsf(outVals[3] - 6.8f) < 1e-5f, "Dense out[1,1] mismatch");

  freeMemory(mem);
}

static void test_dense_backward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor x = create2DTensor(&ctx, 2, 3, F32);
  Tensor w = create2DTensor(&ctx, 2, 3, F32);
  Tensor gradOut = create2DTensor(&ctx, 2, 2, F32);
  Tensor dX;
  Tensor dW;
  Tensor dB;

  f32 *xVals = x.values;
  f32 *wVals = w.values;
  f32 *gVals = gradOut.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;
  xVals[4] = 5.0f;
  xVals[5] = 6.0f;

  wVals[0] = 1.0f;
  wVals[1] = 0.0f;
  wVals[2] = -1.0f;
  wVals[3] = 0.5f;
  wVals[4] = 1.0f;
  wVals[5] = 0.0f;

  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  Result r = DenseBackward(&ctx, &x, &w, &gradOut, &dX, &dW, &dB);
  ASSERT_EQ(r, OK, "DenseBackward should succeed");

  f32 *dxVals = dX.values;
  f32 *dwVals = dW.values;
  f32 *dbVals = dB.values;

  ASSERT(fabsf(dxVals[0] - 2.0f) < 1e-5f, "dX[0,0] mismatch");
  ASSERT(fabsf(dxVals[1] - 2.0f) < 1e-5f, "dX[0,1] mismatch");
  ASSERT(fabsf(dxVals[2] - -1.0f) < 1e-5f, "dX[0,2] mismatch");
  ASSERT(fabsf(dxVals[3] - 5.0f) < 1e-5f, "dX[1,0] mismatch");
  ASSERT(fabsf(dxVals[4] - 4.0f) < 1e-5f, "dX[1,1] mismatch");
  ASSERT(fabsf(dxVals[5] - -3.0f) < 1e-5f, "dX[1,2] mismatch");

  ASSERT(fabsf(dwVals[0] - 13.0f) < 1e-5f, "dW[0,0] mismatch");
  ASSERT(fabsf(dwVals[1] - 17.0f) < 1e-5f, "dW[0,1] mismatch");
  ASSERT(fabsf(dwVals[2] - 21.0f) < 1e-5f, "dW[0,2] mismatch");
  ASSERT(fabsf(dwVals[3] - 18.0f) < 1e-5f, "dW[1,0] mismatch");
  ASSERT(fabsf(dwVals[4] - 24.0f) < 1e-5f, "dW[1,1] mismatch");
  ASSERT(fabsf(dwVals[5] - 30.0f) < 1e-5f, "dW[1,2] mismatch");

  ASSERT(fabsf(dbVals[0] - 4.0f) < 1e-5f, "dB[0] mismatch");
  ASSERT(fabsf(dbVals[1] - 6.0f) < 1e-5f, "dB[1] mismatch");

  freeMemory(mem);
}

static void test_batch_norm_forward_training_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor x = create2DTensor(&ctx, 2, 2, F32);
  Tensor gamma = create1DTensor(&ctx, 2, F32);
  Tensor beta = create1DTensor(&ctx, 2, F32);
  Tensor out;
  Tensor mean;
  Tensor variance;

  f32 *xVals = x.values;
  f32 *gammaVals = gamma.values;
  f32 *betaVals = beta.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;
  gammaVals[0] = 1.0f;
  gammaVals[1] = 1.0f;
  betaVals[0] = 0.0f;
  betaVals[1] = 0.0f;

  Result r = BatchNormForwardTraining(&ctx, &x, &gamma, &beta, 0.0f, &out, &mean, &variance);
  ASSERT_EQ(r, OK, "BatchNormForwardTraining should succeed");

  f32 *outVals = out.values;
  f32 *meanVals = mean.values;
  f32 *varVals = variance.values;
  ASSERT(fabsf(meanVals[0] - 2.0f) < 1e-5f, "mean[0] mismatch");
  ASSERT(fabsf(meanVals[1] - 3.0f) < 1e-5f, "mean[1] mismatch");
  ASSERT(fabsf(varVals[0] - 1.0f) < 1e-5f, "variance[0] mismatch");
  ASSERT(fabsf(varVals[1] - 1.0f) < 1e-5f, "variance[1] mismatch");
  ASSERT(fabsf(outVals[0] - -1.0f) < 1e-5f, "out[0,0] mismatch");
  ASSERT(fabsf(outVals[1] - -1.0f) < 1e-5f, "out[0,1] mismatch");
  ASSERT(fabsf(outVals[2] - 1.0f) < 1e-5f, "out[1,0] mismatch");
  ASSERT(fabsf(outVals[3] - 1.0f) < 1e-5f, "out[1,1] mismatch");

  freeMemory(mem);
}

static void test_batch_norm_backward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor x = create2DTensor(&ctx, 2, 2, F32);
  Tensor grad = create2DTensor(&ctx, 2, 2, F32);
  Tensor gamma = create1DTensor(&ctx, 2, F32);
  Tensor dX;
  Tensor dGamma;
  Tensor dBeta;

  f32 *xVals = x.values;
  f32 *gVals = grad.values;
  f32 *gammaVals = gamma.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;
  gVals[0] = 1.0f;
  gVals[1] = 1.0f;
  gVals[2] = 1.0f;
  gVals[3] = 1.0f;
  gammaVals[0] = 1.0f;
  gammaVals[1] = 1.0f;

  Result r = BatchNormBackward(&ctx, &x, &grad, &gamma, 0.0f, &dX, &dGamma, &dBeta);
  ASSERT_EQ(r, OK, "BatchNormBackward should succeed");

  f32 *dxVals = dX.values;
  f32 *dGammaVals = dGamma.values;
  f32 *dBetaVals = dBeta.values;

  ASSERT(fabsf(dBetaVals[0] - 2.0f) < 1e-5f, "dBeta[0] mismatch");
  ASSERT(fabsf(dBetaVals[1] - 2.0f) < 1e-5f, "dBeta[1] mismatch");
  ASSERT(fabsf(dGammaVals[0] - 0.0f) < 1e-5f, "dGamma[0] mismatch");
  ASSERT(fabsf(dGammaVals[1] - 0.0f) < 1e-5f, "dGamma[1] mismatch");
  ASSERT(fabsf(dxVals[0] - 0.0f) < 1e-5f, "dX[0,0] mismatch");
  ASSERT(fabsf(dxVals[1] - 0.0f) < 1e-5f, "dX[0,1] mismatch");
  ASSERT(fabsf(dxVals[2] - 0.0f) < 1e-5f, "dX[1,0] mismatch");
  ASSERT(fabsf(dxVals[3] - 0.0f) < 1e-5f, "dX[1,1] mismatch");

  freeMemory(mem);
}

static void test_conv2d_rejects_non_float_input(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, I32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, NULL, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, ERR_CONV2D_KERNEL_NOT_FLOAT, "Conv2d should reject non-float inputs");
  freeMemory(mem);
}

static void test_conv2d_rejects_tensor_with_too_few_dims(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create2DTensor(&ctx, 3, 3, F32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, NULL, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, ERR_CONV2D_INVALID_NUM_TENSOR_DIM,
            "Conv2d should reject tensors with fewer than 3 dims");
  freeMemory(mem);
}

static void test_conv2d_rejects_zero_in_channels(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 0, 1, 1, NULL, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, ERR_CONV2D_IN_CHANNELS_ZERO, "Conv2d should reject zero input channels");
  freeMemory(mem);
}

static void test_conv2d_rejects_zero_out_channels(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 0, 1, NULL, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, ERR_CONV2D_OUT_CHANNELS_ZERO, "Conv2d should reject zero output channels");
  freeMemory(mem);
}

static void test_conv2d_rejects_non_2d_kernel_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor out;

  Tensor kernels = create2DTensor(&ctx, 1, 1, F32);
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Conv2d should reject kernels without 4 dims");
  freeMemory(mem);
}

static void test_conv2d_forward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 9; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;

  dim_t kernelDimsArr[2] = {2, 2};
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d single-channel forward should succeed");
  ASSERT_EQ(out.shape.numOfDims, 4, "Conv2d output should be 4D");
  ASSERT_EQ(out.shape.dims[0], 1, "Output batch mismatch");
  ASSERT_EQ(out.shape.dims[1], 2, "Output height mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Output width mismatch");
  ASSERT_EQ(out.shape.dims[3], 1, "Output channels mismatch");
  ASSERT(out.isContigous, "Conv2d output should be contiguous");

  f32 *o = out.values;
  ASSERT(fabsf(o[0] - 6.0f) < 1e-5f, "Conv out[0,0,0,0] mismatch");
  ASSERT(fabsf(o[1] - 8.0f) < 1e-5f, "Conv out[0,0,0,1] mismatch");
  ASSERT(fabsf(o[2] - 12.0f) < 1e-5f, "Conv out[0,0,1,0] mismatch");
  ASSERT(fabsf(o[3] - 14.0f) < 1e-5f, "Conv out[0,0,1,1] mismatch");

  freeMemory(mem);
}

static void test_conv2d_forward_f32_single_channel_with_bias(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor bias = create4DTensor(&ctx, 1, 1, 1, 1, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 9; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;

  ((f32 *)bias.values)[0] = 2.5f;

  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &bias, true, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d with bias should succeed");

  f32 *o = out.values;
  ASSERT(fabsf(o[0] - 8.5f) < 1e-5f, "Conv with bias out[0] mismatch");
  ASSERT(fabsf(o[1] - 10.5f) < 1e-5f, "Conv with bias out[1] mismatch");
  ASSERT(fabsf(o[2] - 14.5f) < 1e-5f, "Conv with bias out[2] mismatch");
  ASSERT(fabsf(o[3] - 16.5f) < 1e-5f, "Conv with bias out[3] mismatch");

  freeMemory(mem);
}

static void test_conv2d_returns_col_buffer_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor out;
  Tensor colBuffer;

  f32 *x = t.values;
  for (int i = 0; i < 9; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;

  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, &colBuffer);

  ASSERT_EQ(r, OK, "Conv2d should return a col buffer when requested");
  ASSERT_EQ(colBuffer.shape.numOfDims, 2, "Returned col buffer should be 2D");
  ASSERT_EQ(colBuffer.shape.dims[0], 4, "Returned col buffer row count mismatch");
  ASSERT_EQ(colBuffer.shape.dims[1], 4, "Returned col buffer column count mismatch");
  ASSERT(colBuffer.isContigous, "Returned col buffer should be contiguous");

  f32 *col = colBuffer.values;
  f32 wantCol[16] = {1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f,
                     4.0f, 5.0f, 7.0f, 8.0f, 5.0f, 6.0f, 8.0f, 9.0f};
  for (int i = 0; i < 16; i++) {
    ASSERT(fabsf(col[i] - wantCol[i]) < 1e-5f, "Returned col buffer contents mismatch");
  }

  f32 *o = out.values;
  ASSERT(fabsf(o[0] - 6.0f) < 1e-5f, "Conv out[0,0,0,0] mismatch");
  ASSERT(fabsf(o[1] - 8.0f) < 1e-5f, "Conv out[0,0,0,1] mismatch");
  ASSERT(fabsf(o[2] - 12.0f) < 1e-5f, "Conv out[0,0,1,0] mismatch");
  ASSERT(fabsf(o[3] - 14.0f) < 1e-5f, "Conv out[0,0,1,1] mismatch");

  freeMemory(mem);
}

static void test_conv2d_forward_f32_multi_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 2, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 2, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  f32 input[18] = {1.0f, 10.0f, 2.0f, 11.0f, 3.0f, 12.0f, 4.0f, 13.0f, 5.0f,
                   14.0f, 6.0f, 15.0f, 7.0f, 16.0f, 8.0f, 17.0f, 9.0f, 18.0f};
  for (int i = 0; i < 18; i++) {
    x[i] = input[i];
  }

  f32 *k = kernels.values;
  for (int i = 0; i < 4; i++) {
    k[i] = 1.0f;
  }
  for (int i = 4; i < 8; i++) {
    k[i] = 0.5f;
  }

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 2, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d multi-channel forward should succeed");
  ASSERT_EQ(out.shape.dims[1], 2, "Multi-channel output height mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Multi-channel output width mismatch");
  ASSERT_EQ(out.shape.dims[3], 1, "Multi-channel output channels mismatch");

  f32 *o = out.values;
  ASSERT(fabsf(o[0] - 36.0f) < 1e-5f, "Conv multi out[0] mismatch");
  ASSERT(fabsf(o[1] - 42.0f) < 1e-5f, "Conv multi out[1] mismatch");
  ASSERT(fabsf(o[2] - 54.0f) < 1e-5f, "Conv multi out[2] mismatch");
  ASSERT(fabsf(o[3] - 60.0f) < 1e-5f, "Conv multi out[3] mismatch");

  freeMemory(mem);
}

static void test_conv2d_forward_f32_with_batch_dimension(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 2, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 18; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d batched forward should succeed");
  ASSERT_EQ(out.shape.dims[0], 2, "Batched output batch mismatch");
  ASSERT_EQ(out.shape.dims[1], 2, "Batched output height mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Batched output width mismatch");
  ASSERT_EQ(out.shape.dims[3], 1, "Batched output channels mismatch");

  f32 *o = out.values;
  ASSERT(fabsf(o[0] - 6.0f) < 1e-5f, "Batched conv out[0] mismatch");
  ASSERT(fabsf(o[1] - 8.0f) < 1e-5f, "Batched conv out[1] mismatch");
  ASSERT(fabsf(o[2] - 12.0f) < 1e-5f, "Batched conv out[2] mismatch");
  ASSERT(fabsf(o[3] - 14.0f) < 1e-5f, "Batched conv out[3] mismatch");
  ASSERT(fabsf(o[4] - 24.0f) < 1e-5f, "Batched conv out[4] mismatch");
  ASSERT(fabsf(o[5] - 26.0f) < 1e-5f, "Batched conv out[5] mismatch");
  ASSERT(fabsf(o[6] - 30.0f) < 1e-5f, "Batched conv out[6] mismatch");
  ASSERT(fabsf(o[7] - 32.0f) < 1e-5f, "Batched conv out[7] mismatch");

  freeMemory(mem);
}

static void test_conv2d_restores_openblas_threads_after_local_override(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 9; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;

  int previousThreads = openblas_get_num_threads();
  int maxThreads = openblas_get_num_procs();
  const char *overrideValue = maxThreads > 1 && previousThreads == 1 ? "2" : "1";

  setenv("SHAPES_CONV_THREADS", overrideValue, 1);

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d with a local thread override should succeed");
  ASSERT_EQ(openblas_get_num_threads(), previousThreads,
            "Conv2d should restore the previous OpenBLAS thread count");

  unsetenv("SHAPES_CONV_THREADS");
  freeMemory(mem);
}

static void test_conv2d_forward_f32_stride_two_multi_out_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 5, 5, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 2, 1, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 25; i++) {
    x[i] = (f32)(i + 1);
  }

  f32 *k = kernels.values;
  k[0] = 1.0f;
  k[1] = 0.0f;
  k[2] = 0.0f;
  k[3] = 1.0f;
  k[4] = 1.0f;
  k[5] = 1.0f;
  k[6] = 0.0f;
  k[7] = 0.0f;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 2, 2, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d stride-two forward should succeed");
  ASSERT_EQ(out.shape.dims[1], 2, "Stride-two output height mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Stride-two output width mismatch");
  ASSERT_EQ(out.shape.dims[3], 2, "Stride-two output channel mismatch");

  f32 *o = out.values;
  f32 want[8] = {8.0f, 3.0f, 12.0f, 7.0f, 28.0f, 23.0f, 32.0f, 27.0f};
  for (int i = 0; i < 8; i++) {
    ASSERT(fabsf(o[i] - want[i]) < 1e-5f, "Conv2d stride-two output mismatch");
  }

  freeMemory(mem);
}

static void test_conv2d_forward_f64_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 3, 3, 1, F64);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F64);
  Tensor out;

  f64 *x = t.values;
  for (int i = 0; i < 9; i++) {
    x[i] = (f64)(i + 1);
  }

  f64 *k = kernels.values;
  k[0] = 0.5;
  k[1] = 1.0;
  k[2] = -1.0;
  k[3] = 2.0;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, NULL, false, &t, &out, NULL);

  ASSERT_EQ(r, OK, "Conv2d f64 forward should succeed");

  f64 *o = out.values;
  f64 want[4] = {8.5, 11.0, 16.0, 18.5};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabs(o[i] - want[i]) < 1e-9, "Conv2d f64 output mismatch");
  }

  freeMemory(mem);
}

static void test_conv2d_backward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor dKernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);

  f32 *xVals = x.values;
  for (int i = 0; i < 9; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  f32 *gVals = gradOut.values;
  for (int i = 0; i < 4; i++) {
    gVals[i] = 1.0f;
  }

  memset(dX.values, 0, dX.size * sizeof(f32));
  memset(dKernels.values, 0, dKernels.size * sizeof(f32));

  void *colBuffer = im2colF32(&ctx, &x, 2, 2, 1);
  Result r = Conv2dBackward(&ctx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, NULL, false, 1);

  ASSERT_EQ(r, OK, "Conv2dBackward should succeed");
  ASSERT_EQ(dX.shape.numOfDims, 4, "Conv2dBackward dX should be 4D");
  ASSERT_EQ(dKernels.shape.numOfDims, 4, "Conv2dBackward dKernels should be 4D");

  f32 *dxVals = dX.values;
  f32 wantDX[9] = {1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  for (int i = 0; i < 9; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "Conv2dBackward dX mismatch");
  }

  f32 *dKernelVals = dKernels.values;
  f32 wantDK[4] = {12.0f, 16.0f, 24.0f, 28.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dKernelVals[i] - wantDK[i]) < 1e-5f, "Conv2dBackward dKernels mismatch");
  }

  freeAlloc(ctx.memory, colBuffer);
  freeMemory(mem);
}

static void test_conv2d_backward_f32_bias_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor dKernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor dBias = create4DTensor(&ctx, 1, 1, 1, 1, F32);

  f32 *xVals = x.values;
  for (int i = 0; i < 9; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  f32 *gVals = gradOut.values;
  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  memset(dX.values, 0, dX.size * sizeof(f32));
  memset(dKernels.values, 0, dKernels.size * sizeof(f32));
  memset(dBias.values, 0, dBias.size * sizeof(f32));

  void *colBuffer = im2colF32(&ctx, &x, 2, 2, 1);
  Result r = Conv2dBackward(&ctx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, &dBias, true, 1);

  ASSERT_EQ(r, OK, "Conv2dBackward with bias grad should succeed");
  ASSERT(fabsf(((f32 *)dBias.values)[0] - 10.0f) < 1e-5f, "Conv2dBackward dBias mismatch");

  freeAlloc(ctx.memory, colBuffer);
  freeMemory(mem);
}

static void test_conv2d_backward_uses_provided_col_buffer_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor dKernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);

  f32 *xVals = x.values;
  for (int i = 0; i < 9; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  f32 *gVals = gradOut.values;
  for (int i = 0; i < 4; i++) {
    gVals[i] = 1.0f;
  }

  memset(dX.values, 0, dX.size * sizeof(f32));
  memset(dKernels.values, 0, dKernels.size * sizeof(f32));

  Tensor *colBuffer = im2colF32(&ctx, &x, 2, 2, 1);
  memset(colBuffer->values, 0, colBuffer->size * sizeof(f32));

  Result r = Conv2dBackward(&ctx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, NULL, false, 1);

  ASSERT_EQ(r, OK, "Conv2dBackward should accept an explicitly provided col buffer");

  f32 *dxVals = dX.values;
  f32 wantDX[9] = {1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  for (int i = 0; i < 9; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f,
           "Conv2dBackward dX mismatch with provided col buffer");
  }

  f32 *dKernelVals = dKernels.values;
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dKernelVals[i]) < 1e-5f,
           "Conv2dBackward should use the provided col buffer for dKernels");
  }

  freeMemory(mem);
}

static void test_conv2d_backward_f32_stride_two_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 5, 5, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX = create4DTensor(&ctx, 1, 5, 5, 1, F32);
  Tensor dKernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);

  f32 *xVals = x.values;
  for (int i = 0; i < 25; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 2.0f;
  kVals[2] = 3.0f;
  kVals[3] = 4.0f;

  f32 *gVals = gradOut.values;
  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  memset(dX.values, 0, dX.size * sizeof(f32));
  memset(dKernels.values, 0, dKernels.size * sizeof(f32));

  void *colBuffer = im2colF32(&ctx, &x, 2, 2, 2);
  Result r = Conv2dBackward(&ctx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, NULL, false, 2);
  ASSERT_EQ(r, OK, "Conv2dBackward stride-two should succeed");

  f32 *dxVals = dX.values;
  f32 wantDX[25] = {1.0f, 2.0f, 2.0f, 4.0f,  0.0f,  3.0f,  4.0f, 6.0f, 8.0f, 0.0f, 3.0f, 6.0f, 4.0f,
                    8.0f, 0.0f, 9.0f, 12.0f, 12.0f, 16.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 25; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "Conv2dBackward stride-two dX mismatch");
  }

  f32 *dKernelVals = dKernels.values;
  f32 wantDK[4] = {92.0f, 102.0f, 142.0f, 152.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dKernelVals[i] - wantDK[i]) < 1e-5f,
           "Conv2dBackward stride-two dKernels mismatch");
  }

  freeAlloc(ctx.memory, colBuffer);
  freeMemory(mem);
}

static void test_conv2d_backward_f32_multi_batch_multi_out_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 2, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 2, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 2, 2, 2, 2, F32);
  Tensor dX = create4DTensor(&ctx, 2, 3, 3, 1, F32);
  Tensor dKernels = create4DTensor(&ctx, 2, 1, 2, 2, F32);

  f32 *xVals = x.values;
  for (int i = 0; i < 18; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;
  kVals[4] = 0.0f;
  kVals[5] = 1.0f;
  kVals[6] = 1.0f;
  kVals[7] = 0.0f;

  f32 *gVals = gradOut.values;
  f32 grads[16] = {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f,
                   2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 2.0f, 1.0f, 3.0f};
  for (int i = 0; i < 16; i++) {
    gVals[i] = grads[i];
  }

  memset(dX.values, 0, dX.size * sizeof(f32));
  memset(dKernels.values, 0, dKernels.size * sizeof(f32));

  void *colBuffer = im2colF32(&ctx, &x, 2, 2, 1);
  Result r = Conv2dBackward(&ctx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, NULL, false, 1);
  ASSERT_EQ(r, OK, "Conv2dBackward multi-batch multi-out should succeed");

  f32 *dxVals = dX.values;
  f32 wantDX[18] = {1.0f, 7.0f, 6.0f, 8.0f, 18.0f, 10.0f, 7.0f, 11.0f, 4.0f,
                    2.0f, 2.0f, 0.0f, 1.0f, 5.0f,  4.0f,  2.0f, 3.0f,  1.0f};
  for (int i = 0; i < 18; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "Conv2dBackward multi-batch dX mismatch");
  }

  f32 *dKernelVals = dKernels.values;
  f32 wantDK[8] = {82.0f, 96.0f, 124.0f, 138.0f, 163.0f, 195.0f, 259.0f, 291.0f};
  for (int i = 0; i < 8; i++) {
    ASSERT(fabsf(dKernelVals[i] - wantDK[i]) < 1e-5f,
           "Conv2dBackward multi-batch dKernels mismatch");
  }

  freeAlloc(ctx.memory, colBuffer);
  freeMemory(mem);
}

static void test_conv2d_backward_materializes_cuda_sources_on_cpu_target_context(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};

  Tensor x = create4DTensor(&hostCtx, 1, 3, 3, 1, F32);
  Tensor kernels = create4DTensor(&hostCtx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&hostCtx, 1, 2, 2, 1, F32);
  Tensor dX = create4DTensor(&hostCtx, 1, 3, 3, 1, F32);
  Tensor dKernels = create4DTensor(&hostCtx, 1, 1, 2, 2, F32);

  x.context = &hostCtx;
  x.metadataMemory = hostCtx.memory;
  kernels.context = &hostCtx;
  kernels.metadataMemory = hostCtx.memory;
  gradOut.context = &hostCtx;
  gradOut.metadataMemory = hostCtx.memory;
  dX.context = &hostCtx;
  dX.metadataMemory = hostCtx.memory;
  dKernels.context = &hostCtx;
  dKernels.metadataMemory = hostCtx.memory;

  f32 *xVals = x.values;
  for (int i = 0; i < 9; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  f32 *gVals = gradOut.values;
  for (int i = 0; i < 4; i++) {
    gVals[i] = 1.0f;
  }

  Result moveResult = MoveTensors(&ctx, 3, &x, &kernels, &gradOut);
  ASSERT_EQ(moveResult, OK, "Conv2dBackward mixed-context setup should move tensors to CUDA");

  Tensor *colBuffer = im2colF32(&ctx, &x, 2, 2, 1);
  ASSERT(colBuffer != NULL, "Conv2dBackward mixed-context setup should create CUDA col buffer");

  Result r = Conv2dBackward(&hostCtx, &x, &dX, &kernels, &dKernels, &gradOut, colBuffer, NULL, false, 1);
  ASSERT_EQ(r, OK, "Conv2dBackward should materialize CUDA sources onto CPU target context");

  f32 *dxVals = dX.values;
  f32 wantDX[9] = {1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  for (int i = 0; i < 9; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "Conv2dBackward mixed-context dX mismatch");
  }

  f32 *dKernelVals = dKernels.values;
  f32 wantDK[4] = {12.0f, 16.0f, 24.0f, 28.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dKernelVals[i] - wantDK[i]) < 1e-5f,
           "Conv2dBackward mixed-context dKernels mismatch");
  }

  DestroyContext(&ctx);
}

static void test_conv_transpose2d_forward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor out;

  f32 *xVals = x.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = ConvTranspose2d(&ctx, 1, 1, 1, &kernels, kernel, &x, &out);

  ASSERT_EQ(r, OK, "ConvTranspose2d should succeed");
  ASSERT_EQ(out.shape.dims[1], 3, "ConvTranspose2d output height mismatch");
  ASSERT_EQ(out.shape.dims[2], 3, "ConvTranspose2d output width mismatch");
  ASSERT_EQ(out.shape.dims[3], 1, "ConvTranspose2d output channels mismatch");

  f32 *o = out.values;
  f32 want[9] = {1.0f, 2.0f, 0.0f, 3.0f, 5.0f, 2.0f, 0.0f, 3.0f, 4.0f};
  for (int i = 0; i < 9; i++) {
    ASSERT(fabsf(o[i] - want[i]) < 1e-5f, "ConvTranspose2d output mismatch");
  }

  freeMemory(mem);
}

static void test_conv_transpose2d_backward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 3, 3, 1, F32);
  Tensor dX;
  Tensor dKernels;

  f32 *xVals = x.values;
  xVals[0] = 1.0f;
  xVals[1] = 2.0f;
  xVals[2] = 3.0f;
  xVals[3] = 4.0f;

  f32 *kVals = kernels.values;
  kVals[0] = 1.0f;
  kVals[1] = 0.0f;
  kVals[2] = 0.0f;
  kVals[3] = 1.0f;

  f32 *gVals = gradOut.values;
  for (int i = 0; i < 9; i++) {
    gVals[i] = 1.0f;
  }

  Result r = ConvTranspose2dBackward(&ctx, &x, &kernels, &gradOut, 1, &dX, &dKernels);
  ASSERT_EQ(r, OK, "ConvTranspose2dBackward should succeed");

  f32 *dxVals = dX.values;
  f32 wantDX[4] = {2.0f, 2.0f, 2.0f, 2.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "ConvTranspose2dBackward dX mismatch");
  }

  f32 *dKernelVals = dKernels.values;
  f32 wantDK[4] = {10.0f, 10.0f, 10.0f, 10.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(dKernelVals[i] - wantDK[i]) < 1e-5f, "ConvTranspose2dBackward dKernels mismatch");
  }

  freeMemory(mem);
}

static void test_max_pool2d_forward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 4, 4, 1, F32);
  Tensor out;

  f32 *xVals = x.values;
  f32 input[16] = {1.0f, 3.0f, 2.0f, 1.0f, 4.0f, 6.0f, 5.0f, 2.0f,
                   7.0f, 8.0f, 9.0f, 3.0f, 0.0f, 1.0f, 2.0f, 4.0f};
  for (int i = 0; i < 16; i++) {
    xVals[i] = input[i];
  }

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = MaxPool2d(&ctx, &x, kernel, 2, &out);
  ASSERT_EQ(r, OK, "MaxPool2d should succeed");

  f32 *o = out.values;
  f32 want[4] = {6.0f, 5.0f, 8.0f, 9.0f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(o[i] - want[i]) < 1e-5f, "MaxPool2d output mismatch");
  }

  freeMemory(mem);
}

static void test_max_pool2d_backward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 4, 4, 1, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX;

  f32 *xVals = x.values;
  f32 input[16] = {1.0f, 3.0f, 2.0f, 1.0f, 4.0f, 6.0f, 5.0f, 2.0f,
                   7.0f, 8.0f, 9.0f, 3.0f, 0.0f, 1.0f, 2.0f, 4.0f};
  for (int i = 0; i < 16; i++) {
    xVals[i] = input[i];
  }

  f32 *gVals = gradOut.values;
  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = MaxPool2dBackward(&ctx, &x, &gradOut, kernel, 2, &dX);
  ASSERT_EQ(r, OK, "MaxPool2dBackward should succeed");

  f32 *dxVals = dX.values;
  f32 wantDX[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                    0.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 16; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "MaxPool2dBackward dX mismatch");
  }

  freeMemory(mem);
}

static void test_max_pool2d_backward_materializes_cuda_sources_on_cpu_target_context(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};
  Tensor x = create4DTensor(&hostCtx, 1, 4, 4, 1, F32);
  Tensor gradOut = create4DTensor(&hostCtx, 1, 2, 2, 1, F32);
  Tensor dX;

  x.context = &hostCtx;
  x.metadataMemory = hostCtx.memory;
  gradOut.context = &hostCtx;
  gradOut.metadataMemory = hostCtx.memory;

  f32 *xVals = x.values;
  f32 input[16] = {1.0f, 3.0f, 2.0f, 1.0f, 4.0f, 6.0f, 5.0f, 2.0f,
                   7.0f, 8.0f, 9.0f, 3.0f, 0.0f, 1.0f, 2.0f, 4.0f};
  for (int i = 0; i < 16; i++) {
    xVals[i] = input[i];
  }

  f32 *gVals = gradOut.values;
  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  Result moveResult = MoveTensors(&ctx, 2, &x, &gradOut);
  ASSERT_EQ(moveResult, OK, "MaxPool2dBackward mixed-context setup should move tensors to CUDA");

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = MaxPool2dBackward(&hostCtx, &x, &gradOut, kernel, 2, &dX);
  ASSERT_EQ(r, OK, "MaxPool2dBackward should materialize CUDA sources onto CPU target context");

  f32 *dxVals = dX.values;
  f32 wantDX[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                    0.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 16; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f,
           "MaxPool2dBackward mixed-context dX mismatch");
  }

  DestroyContext(&ctx);
}

static void test_adaptive_avg_pool2d_forward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 4, 4, 1, F32);
  Tensor out;

  f32 *xVals = x.values;
  for (int i = 0; i < 16; i++) {
    xVals[i] = (f32)(i + 1);
  }

  Result r = AdaptiveAvgPool2d(&ctx, &x, 2, 2, &out);
  ASSERT_EQ(r, OK, "AdaptiveAvgPool2d should succeed");

  f32 *o = out.values;
  f32 want[4] = {3.5f, 5.5f, 11.5f, 13.5f};
  for (int i = 0; i < 4; i++) {
    ASSERT(fabsf(o[i] - want[i]) < 1e-5f, "AdaptiveAvgPool2d output mismatch");
  }

  freeMemory(mem);
}

static void test_adaptive_avg_pool2d_backward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 4, 4, 1, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 2, 2, 1, F32);
  Tensor dX;

  f32 *xVals = x.values;
  for (int i = 0; i < 16; i++) {
    xVals[i] = (f32)(i + 1);
  }

  f32 *gVals = gradOut.values;
  gVals[0] = 1.0f;
  gVals[1] = 2.0f;
  gVals[2] = 3.0f;
  gVals[3] = 4.0f;

  Result r = AdaptiveAvgPool2dBackward(&ctx, &x, &gradOut, 2, 2, &dX);
  ASSERT_EQ(r, OK, "AdaptiveAvgPool2dBackward should succeed");

  f32 *dxVals = dX.values;
  f32 wantDX[16] = {0.25f, 0.25f, 0.50f, 0.50f, 0.25f, 0.25f, 0.50f, 0.50f,
                    0.75f, 0.75f, 1.00f, 1.00f, 0.75f, 0.75f, 1.00f, 1.00f};
  for (int i = 0; i < 16; i++) {
    ASSERT(fabsf(dxVals[i] - wantDX[i]) < 1e-5f, "AdaptiveAvgPool2dBackward dX mismatch");
  }

  freeMemory(mem);
}

static void test_cross_entropy_forward_cuda_dispatch_uses_target_context(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};

  Tensor yGround = create2DTensor(&hostCtx, 2, 3, F32);
  Tensor logits = create2DTensor(&hostCtx, 2, 3, F32);
  Tensor loss;
  Tensor probs;

  f32 yValues[6] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  f32 logitValues[6] = {2.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f};
  memcpy(yGround.values, yValues, sizeof(yValues));
  memcpy(logits.values, logitValues, sizeof(logitValues));

  Result result = CrossEntropyForward(&ctx, &yGround, &logits, &loss, &probs);
  ASSERT_EQ(result, OK, "CUDA CrossEntropyForward should succeed");
  ASSERT(loss.context == &ctx, "CUDA CrossEntropyForward loss should live on target context");
  ASSERT(probs.context == &ctx, "CUDA CrossEntropyForward probs should live on target context");

  f32 expectedLoss = 0.40760595f;
  f32 expectedProbs[6] = {0.66524094f, 0.24472848f, 0.09003057f,
                          0.09003057f, 0.24472848f, 0.66524094f};
  assertScalarF32Close(&loss, expectedLoss, 1e-5f, "CUDA CrossEntropyForward loss should match");
  assertMovedF32TensorClose(&ctx, &probs, expectedProbs, 6, 1e-5f,
                            "CUDA CrossEntropyForward probs should match");

  DestroyContext(&ctx);
}

static void test_cross_entropy_backward_cuda_dispatch_uses_target_context(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};

  Tensor yGround = create2DTensor(&hostCtx, 2, 3, F32);
  Tensor probs = create2DTensor(&hostCtx, 2, 3, F32);
  Tensor gradOut = createScalarTensor(&hostCtx, F32);
  Tensor dLogits;

  f32 yValues[6] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  f32 probValues[6] = {0.66524094f, 0.24472848f, 0.09003057f,
                       0.09003057f, 0.24472848f, 0.66524094f};
  memcpy(yGround.values, yValues, sizeof(yValues));
  memcpy(probs.values, probValues, sizeof(probValues));
  ((f32 *)gradOut.values)[0] = 1.0f;

  Result result = CrossEntropyBackward(&ctx, &yGround, &probs, &gradOut, &dLogits);
  ASSERT_EQ(result, OK, "CUDA CrossEntropyBackward should succeed");
  ASSERT(dLogits.context == &ctx, "CUDA CrossEntropyBackward result should live on target context");

  f32 expected[6] = {-0.16737953f, 0.12236424f, 0.04501529f,
                     0.04501529f,  0.12236424f, -0.16737953f};
  assertMovedF32TensorClose(&ctx, &dLogits, expected, 6, 1e-5f,
                            "CUDA CrossEntropyBackward result should match");

  DestroyContext(&ctx);
}

void run_layer_tests(void) {
  test_dense_linear_forward_with_bias_f32();
  test_dense_backward_f32();
  test_batch_norm_forward_training_f32();
  test_batch_norm_backward_f32();
  test_conv2d_rejects_non_float_input();
  test_conv2d_rejects_tensor_with_too_few_dims();
  test_conv2d_rejects_zero_in_channels();
  test_conv2d_rejects_zero_out_channels();
  test_conv2d_rejects_non_2d_kernel_shape();
  test_conv2d_forward_f32_single_channel();
  test_conv2d_forward_f32_single_channel_with_bias();
  test_conv2d_returns_col_buffer_f32();
  test_conv2d_forward_f32_multi_channel();
  test_conv2d_forward_f32_with_batch_dimension();
  test_conv2d_restores_openblas_threads_after_local_override();
  test_conv2d_forward_f32_stride_two_multi_out_channel();
  test_conv2d_forward_f64_single_channel();
  test_conv2d_backward_f32_single_channel();
  test_conv2d_backward_f32_bias_grad();
  test_conv2d_backward_uses_provided_col_buffer_f32();
  test_conv2d_backward_f32_stride_two_single_channel();
  test_conv2d_backward_f32_multi_batch_multi_out_channel();
  test_conv2d_backward_materializes_cuda_sources_on_cpu_target_context();
  test_conv_transpose2d_forward_f32_single_channel();
  test_conv_transpose2d_backward_f32_single_channel();
  test_max_pool2d_forward_f32();
  test_max_pool2d_backward_f32();
  test_max_pool2d_backward_materializes_cuda_sources_on_cpu_target_context();
  test_adaptive_avg_pool2d_forward_f32();
  test_adaptive_avg_pool2d_backward_f32();
  test_cross_entropy_forward_cuda_dispatch_uses_target_context();
  test_cross_entropy_backward_cuda_dispatch_uses_target_context();
}
