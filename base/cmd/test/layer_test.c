#include "test.h"
#include "../../shapes.h"
#include <math.h>

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

void run_layer_tests(void) {
  test_dense_linear_forward_with_bias_f32();
  test_dense_backward_f32();
  test_batch_norm_forward_training_f32();
  test_batch_norm_backward_f32();
}
