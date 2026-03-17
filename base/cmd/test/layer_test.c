#include "common.h"
#include "test.h"
#include "../../shapes.h"
#include "cblas.h"
#include <math.h>
#include <stdlib.h>

static Tensor create1DTensor(Context *ctx, dim_t size, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  u8 *multipliers = allocate(ctx->memory, sizeof(u8));
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

static Tensor create2DTensor(Context *ctx, dim_t rows, dim_t cols, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 2);
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * 2);
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
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * 3);
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
  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * 4);
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
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, I32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, NULL, &t, &out);

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
  Result r = Conv2d(&ctx, 1, 1, 1, NULL, &t, &out);

  ASSERT_EQ(r, ERR_CONV2D_INVALID_NUM_TENSOR_DIM,
            "Conv2d should reject tensors with fewer than 3 dims");
  freeMemory(mem);
}

static void test_conv2d_rejects_zero_in_channels(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 0, 1, 1, NULL, &t, &out);

  ASSERT_EQ(r, ERR_CONV2D_IN_CHANNELS_ZERO, "Conv2d should reject zero input channels");
  freeMemory(mem);
}

static void test_conv2d_rejects_zero_out_channels(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F32);
  Tensor out;

  dim_t kernelDimsArr[2] = {2, 2};
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 0, 1, NULL, &t, &out);

  ASSERT_EQ(r, ERR_CONV2D_OUT_CHANNELS_ZERO, "Conv2d should reject zero output channels");
  freeMemory(mem);
}

static void test_conv2d_rejects_non_2d_kernel_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F32);
  Tensor out;

  Tensor kernels = create2DTensor(&ctx, 1, 1, F32);
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &t, &out);

  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Conv2d should reject kernels without 4 dims");
  freeMemory(mem);
}

static void test_conv2d_forward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F32);
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
  Dim kernel = {.dims = kernelDimsArr, .numOfDims = 2, .multipliers = NULL};
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &t, &out);

  ASSERT_EQ(r, OK, "Conv2d single-channel forward should succeed");
  ASSERT_EQ(out.shape.numOfDims, 4, "Conv2d output should be 4D");
  ASSERT_EQ(out.shape.dims[0], 1, "Output batch mismatch");
  ASSERT_EQ(out.shape.dims[1], 1, "Output channels mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Output height mismatch");
  ASSERT_EQ(out.shape.dims[3], 2, "Output width mismatch");
  ASSERT(out.isContigous, "Conv2d output should be contiguous");

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
  Tensor t = create4DTensor(&ctx, 1, 2, 3, 3, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 2, 2, 2, F32);
  Tensor out;

  f32 *x = t.values;
  for (int i = 0; i < 18; i++) {
    x[i] = (f32)(i + 1);
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
  Result r = Conv2d(&ctx, 2, 1, 1, &kernels, &t, &out);

  ASSERT_EQ(r, OK, "Conv2d multi-channel forward should succeed");
  ASSERT_EQ(out.shape.dims[2], 2, "Multi-channel output height mismatch");
  ASSERT_EQ(out.shape.dims[3], 2, "Multi-channel output width mismatch");

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
  Tensor t = create4DTensor(&ctx, 2, 1, 3, 3, F32);
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
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &t, &out);

  ASSERT_EQ(r, OK, "Conv2d batched forward should succeed");
  ASSERT_EQ(out.shape.dims[0], 2, "Batched output batch mismatch");
  ASSERT_EQ(out.shape.dims[1], 1, "Batched output channels mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Batched output height mismatch");
  ASSERT_EQ(out.shape.dims[3], 2, "Batched output width mismatch");

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
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F32);
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
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &t, &out);

  ASSERT_EQ(r, OK, "Conv2d with a local thread override should succeed");
  ASSERT_EQ(openblas_get_num_threads(), previousThreads,
            "Conv2d should restore the previous OpenBLAS thread count");

  unsetenv("SHAPES_CONV_THREADS");
  freeMemory(mem);
}

static void test_conv2d_forward_f32_stride_two_multi_out_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 5, 5, F32);
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
  Result r = Conv2d(&ctx, 1, 2, 2, &kernels, &t, &out);

  ASSERT_EQ(r, OK, "Conv2d stride-two forward should succeed");
  ASSERT_EQ(out.shape.dims[1], 2, "Stride-two output channel mismatch");
  ASSERT_EQ(out.shape.dims[2], 2, "Stride-two output height mismatch");
  ASSERT_EQ(out.shape.dims[3], 2, "Stride-two output width mismatch");

  f32 *o = out.values;
  f32 want[8] = {8.0f, 12.0f, 28.0f, 32.0f, 3.0f, 7.0f, 23.0f, 27.0f};
  for (int i = 0; i < 8; i++) {
    ASSERT(fabsf(o[i] - want[i]) < 1e-5f, "Conv2d stride-two output mismatch");
  }

  freeMemory(mem);
}

static void test_conv2d_forward_f64_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = create4DTensor(&ctx, 1, 1, 3, 3, F64);
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
  Result r = Conv2d(&ctx, 1, 1, 1, &kernels, &t, &out);

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
  Tensor x = create4DTensor(&ctx, 1, 1, 3, 3, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor dX;
  Tensor dKernels;

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

  Result r = Conv2dBackward(&ctx, &x, &kernels, &gradOut, 1, &dX, &dKernels);

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

  freeMemory(mem);
}

static void test_conv2d_backward_f32_stride_two_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 1, 5, 5, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor dX;
  Tensor dKernels;

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

  Result r = Conv2dBackward(&ctx, &x, &kernels, &gradOut, 2, &dX, &dKernels);
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

  freeMemory(mem);
}

static void test_conv2d_backward_f32_multi_batch_multi_out_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 2, 1, 3, 3, F32);
  Tensor kernels = create4DTensor(&ctx, 2, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 2, 2, 2, 2, F32);
  Tensor dX;
  Tensor dKernels;

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
  f32 grads[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                   2.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 2.0f, 3.0f};
  for (int i = 0; i < 16; i++) {
    gVals[i] = grads[i];
  }

  Result r = Conv2dBackward(&ctx, &x, &kernels, &gradOut, 1, &dX, &dKernels);
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

  freeMemory(mem);
}

static void test_conv_transpose2d_forward_f32_single_channel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 1, 2, 2, F32);
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
  ASSERT_EQ(out.shape.dims[2], 3, "ConvTranspose2d output height mismatch");
  ASSERT_EQ(out.shape.dims[3], 3, "ConvTranspose2d output width mismatch");

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
  Tensor x = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor kernels = create4DTensor(&ctx, 1, 1, 2, 2, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 1, 3, 3, F32);
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
  Tensor x = create4DTensor(&ctx, 1, 1, 4, 4, F32);
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
  Tensor x = create4DTensor(&ctx, 1, 1, 4, 4, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 1, 2, 2, F32);
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

static void test_adaptive_avg_pool2d_forward_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor x = create4DTensor(&ctx, 1, 1, 4, 4, F32);
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
  Tensor x = create4DTensor(&ctx, 1, 1, 4, 4, F32);
  Tensor gradOut = create4DTensor(&ctx, 1, 1, 2, 2, F32);
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
  test_conv2d_forward_f32_multi_channel();
  test_conv2d_forward_f32_with_batch_dimension();
  test_conv2d_restores_openblas_threads_after_local_override();
  test_conv2d_forward_f32_stride_two_multi_out_channel();
  test_conv2d_forward_f64_single_channel();
  test_conv2d_backward_f32_single_channel();
  test_conv2d_backward_f32_stride_two_single_channel();
  test_conv2d_backward_f32_multi_batch_multi_out_channel();
  test_conv_transpose2d_forward_f32_single_channel();
  test_conv_transpose2d_backward_f32_single_channel();
  test_max_pool2d_forward_f32();
  test_max_pool2d_backward_f32();
  test_adaptive_avg_pool2d_forward_f32();
  test_adaptive_avg_pool2d_backward_f32();
}
