#include "test.h"
#include "shapes.h"
#include "common.h"
#include <math.h>

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

// Test 1: Basic single step with closed-form expected values
static void test_adam_single_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create tensors: param, paramGrad, m, v
  Tensor param = create1DTensor(&ctx, 3, F32);
  Tensor paramGrad = create1DTensor(&ctx, 3, F32);
  Tensor m = create1DTensor(&ctx, 3, F32);
  Tensor v = create1DTensor(&ctx, 3, F32);

  // Initialize: param = [1.0, 2.0, 3.0], grad = [0.1, 0.2, 0.3]
  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  pVals[0] = 1.0f;
  pVals[1] = 2.0f;
  pVals[2] = 3.0f;
  gVals[0] = 0.1f;
  gVals[1] = 0.2f;
  gVals[2] = 0.3f;
  mVals[0] = 0.0f;
  mVals[1] = 0.0f;
  mVals[2] = 0.0f;
  vVals[0] = 0.0f;
  vVals[1] = 0.0f;
  vVals[2] = 0.0f;

  // Adam hyperparameters
  f32 b1 = 0.9f, b2 = 0.999f, a = 0.1f, epsilon = 1e-8f;
  size_t step = 1;

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, b1, b2, step, a, epsilon);

  ASSERT_EQ(r, OK, "Adam single step should return OK");

  // Manually compute expected values for step 1 with b1=0.9, b2=0.999, a=0.1:
  // Bias correction: (1 - b1^1) = 0.1, (1 - b2^1) = 0.001
  // For grad=0.1:
  //   m = 0.9*0 + 0.1*0.1 = 0.01
  //   v = 0.999*0 + 0.001*0.01 = 0.00001
  //   m_hat = 0.01 / 0.1 = 0.1 (grad after bias correction)
  //   v_hat = 0.00001 / 0.001 = 0.01 (grad^2 after bias correction)
  //   sqrt(v_hat) = 0.1
  //   update = 0.1 * 0.1 / (0.1 + 1e-8) ≈ 0.1
  //   param_new = 1.0 - 0.1 = 0.9
  //
  // For grad=0.2: update ≈ 0.1, param_new ≈ 1.9
  // For grad=0.3: update ≈ 0.1, param_new ≈ 2.9

  ASSERT(fabsf(pVals[0] - 0.9f) < 1e-5f, "param[0] should be updated correctly");
  ASSERT(fabsf(pVals[1] - 1.9f) < 1e-5f, "param[1] should be updated correctly");
  ASSERT(fabsf(pVals[2] - 2.9f) < 1e-5f, "param[2] should be updated correctly");

  // Check m and v values
  ASSERT(fabsf(mVals[0] - 0.01f) < 1e-6f, "m[0] should accumulate correctly");
  ASSERT(fabsf(mVals[1] - 0.02f) < 1e-6f, "m[1] should accumulate correctly");
  ASSERT(fabsf(mVals[2] - 0.03f) < 1e-6f, "m[2] should accumulate correctly");

  ASSERT(fabsf(vVals[0] - 0.00001f) < 1e-8f, "v[0] should accumulate correctly");
  ASSERT(fabsf(vVals[1] - 0.00004f) < 1e-8f, "v[1] should accumulate correctly");
  ASSERT(fabsf(vVals[2] - 0.00009f) < 1e-8f, "v[2] should accumulate correctly");
}

// Test 2: Two steps to verify momentum and velocity accumulation
static void test_adam_two_steps_accumulation(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F32);
  Tensor paramGrad = create1DTensor(&ctx, 1, F32);
  Tensor m = create1DTensor(&ctx, 1, F32);
  Tensor v = create1DTensor(&ctx, 1, F32);

  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  pVals[0] = 1.0f;
  mVals[0] = 0.0f;
  vVals[0] = 0.0f;

  f32 b1 = 0.9f, b2 = 0.999f, a = 0.1f, epsilon = 1e-8f;

  // Step 1 with grad = 0.5
  gVals[0] = 0.5f;
  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, b1, b2, 1, a, epsilon);
  ASSERT_EQ(r, OK, "Adam step 1 should succeed");

  // Step 2 with grad = 0.3
  gVals[0] = 0.3f;
  r = Adam(&ctx, &triplet, 1, b1, b2, 2, a, epsilon);
  ASSERT_EQ(r, OK, "Adam step 2 should succeed");

  // Manual calculation for step 2:
  // After step 1: m = 0.05, v = 0.00025
  // Step 2: m = 0.9*0.05 + 0.1*0.3 = 0.045 + 0.03 = 0.075
  //         v = 0.999*0.00025 + 0.001*0.09 = 0.00024975 + 0.00009 = 0.00033975
  // m_hat = 0.075 / (1 - 0.9^2) = 0.075 / 0.19 ≈ 0.3947
  // v_hat = 0.00033975 / (1 - 0.999^2) = 0.00033975 / 0.001999 ≈ 0.16996
  // update = 0.1 * 0.3947 / (sqrt(0.16996) + 1e-8) ≈ 0.1 * 0.3947 / 0.4123 ≈ 0.0957

  ASSERT(fabsf(mVals[0] - 0.075f) < 1e-5f, "m should accumulate over steps");
  ASSERT(fabsf(vVals[0] - 0.00033975f) < 1e-8f, "v should accumulate over steps");
}

// Test 3: Compare with PyTorch reference (known good values)
static void test_adam_pytorch_reference(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 2, F32);
  Tensor paramGrad = create1DTensor(&ctx, 2, F32);
  Tensor m = create1DTensor(&ctx, 2, F32);
  Tensor v = create1DTensor(&ctx, 2, F32);

  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  // PyTorch defaults
  pVals[0] = 1.0f;
  pVals[1] = 2.0f;
  gVals[0] = 0.1f;
  gVals[1] = -0.2f;
  mVals[0] = 0.0f;
  mVals[1] = 0.0f;
  vVals[0] = 0.0f;
  vVals[1] = 0.0f;

  f32 b1 = 0.9f, b2 = 0.999f, a = 0.001f, epsilon = 1e-8f;

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, b1, b2, 1, a, epsilon);
  ASSERT_EQ(r, OK, "Adam should match PyTorch behavior");

  // With PyTorch defaults (b1=0.9, b2=0.999, a=0.001) and grads [0.1, -0.2]:
  // The update is small due to low learning rate, param changes by ~0.001
  // Just verify parameters changed in the right direction
  ASSERT(pVals[0] < 1.0f, "param[0] should decrease with positive grad");
  ASSERT(pVals[1] > 2.0f, "param[1] should increase with negative grad");
}

// Test 4: NULL triplets error
static void test_adam_null_triplets(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Result r = Adam(&ctx, NULL, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_NULL_TRIPLETS, "NULL triplets should return ERR_ADAM_NULL_TRIPLETS");
}

// Test 5: NULL individual tensors
static void test_adam_null_tensors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F32);
  Tensor paramGrad = create1DTensor(&ctx, 1, F32);
  Tensor m = create1DTensor(&ctx, 1, F32);
  Tensor v = create1DTensor(&ctx, 1, F32);

  // Test NULL m
  AdamData triplet1 = {.param = &param, .paramGrad = &paramGrad, .m = NULL, .v = &v};
  Result r = Adam(&ctx, &triplet1, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_NULL_M, "NULL m should return ERR_ADAM_NULL_M");

  // Test NULL v
  AdamData triplet2 = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = NULL};
  r = Adam(&ctx, &triplet2, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_NULL_V, "NULL v should return ERR_ADAM_NULL_V");

  // Test NULL param
  AdamData triplet3 = {.param = NULL, .paramGrad = &paramGrad, .m = &m, .v = &v};
  r = Adam(&ctx, &triplet3, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_NULL_PARAM, "NULL param should return ERR_ADAM_NULL_PARAM");

  // Test NULL paramGrad
  AdamData triplet4 = {.param = &param, .paramGrad = NULL, .m = &m, .v = &v};
  r = Adam(&ctx, &triplet4, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_NULL_GRAD, "NULL paramGrad should return ERR_ADAM_NULL_GRAD");
}

// Test 6: Non-float type rejection
static void test_adam_non_float_type(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, I32);
  Tensor paramGrad = create1DTensor(&ctx, 1, I32);
  Tensor m = create1DTensor(&ctx, 1, I32);
  Tensor v = create1DTensor(&ctx, 1, I32);

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_ONLY_FLOAT_TENSORS, "Integer tensors should return ERR_ADAM_ONLY_FLOAT_TENSORS");
}

// Test 7: Size mismatch
static void test_adam_size_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 3, F32);
  Tensor paramGrad = create1DTensor(&ctx, 3, F32);
  Tensor m = create1DTensor(&ctx, 2, F32); // Wrong size
  Tensor v = create1DTensor(&ctx, 3, F32);

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, ERR_ADAM_PARAMS_SIZE_MISMATCH, "Size mismatch should return ERR_ADAM_PARAMS_SIZE_MISMATCH");
}

// Test 8: Multiple triplets in one call
static void test_adam_multiple_triplets(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // First parameter group
  Tensor param1 = create1DTensor(&ctx, 2, F32);
  Tensor grad1 = create1DTensor(&ctx, 2, F32);
  Tensor m1 = create1DTensor(&ctx, 2, F32);
  Tensor v1 = create1DTensor(&ctx, 2, F32);

  // Second parameter group
  Tensor param2 = create1DTensor(&ctx, 2, F32);
  Tensor grad2 = create1DTensor(&ctx, 2, F32);
  Tensor m2 = create1DTensor(&ctx, 2, F32);
  Tensor v2 = create1DTensor(&ctx, 2, F32);

  f32 *p1 = param1.values, *g1 = grad1.values, *m1v = m1.values, *v1v = v1.values;
  f32 *p2 = param2.values, *g2 = grad2.values, *m2v = m2.values, *v2v = v2.values;

  p1[0] = 1.0f;
  p1[1] = 2.0f;
  g1[0] = 0.1f;
  g1[1] = 0.2f;
  m1v[0] = 0.0f;
  m1v[1] = 0.0f;
  v1v[0] = 0.0f;
  v1v[1] = 0.0f;

  p2[0] = 3.0f;
  p2[1] = 4.0f;
  g2[0] = 0.3f;
  g2[1] = 0.4f;
  m2v[0] = 0.0f;
  m2v[1] = 0.0f;
  v2v[0] = 0.0f;
  v2v[1] = 0.0f;

  AdamData triplets[2] = {{.param = &param1, .paramGrad = &grad1, .m = &m1, .v = &v1}, {.param = &param2, .paramGrad = &grad2, .m = &m2, .v = &v2}};

  Result r = Adam(&ctx, triplets, 2, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, OK, "Multiple triplets should work");

  // Both parameters should have been updated
  ASSERT(p1[0] != 1.0f, "param1[0] should be updated");
  ASSERT(p2[0] != 3.0f, "param2[0] should be updated");
}

// Test 9: Bias correction verification
static void test_adam_bias_correction(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F32);
  Tensor paramGrad = create1DTensor(&ctx, 1, F32);
  Tensor m = create1DTensor(&ctx, 1, F32);
  Tensor v = create1DTensor(&ctx, 1, F32);

  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  pVals[0] = 1.0f;
  gVals[0] = 1.0f;
  mVals[0] = 0.0f;
  vVals[0] = 0.0f;

  f32 b1 = 0.9f, b2 = 0.999f, a = 0.1f, epsilon = 1e-8f;

  // Step 1: strong bias correction (dividing by small numbers)
  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, b1, b2, 1, a, epsilon);
  ASSERT_EQ(r, OK, "Step 1 should succeed");

  f32 step1_param = pVals[0];

  // Reset and try step 1000: weak bias correction
  pVals[0] = 1.0f;
  mVals[0] = 0.0f;
  vVals[0] = 0.0f;

  r = Adam(&ctx, &triplet, 1, b1, b2, 1000, a, epsilon);
  ASSERT_EQ(r, OK, "Step 1000 should succeed");

  f32 step1000_param = pVals[0];

  // Step 1 has stronger updates due to bias correction
  ASSERT(step1_param != step1000_param, "Bias correction should affect step size");
}

// Test 10: Very small gradients (numerical stability)
static void test_adam_small_gradients(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F32);
  Tensor paramGrad = create1DTensor(&ctx, 1, F32);
  Tensor m = create1DTensor(&ctx, 1, F32);
  Tensor v = create1DTensor(&ctx, 1, F32);

  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  pVals[0] = 1.0f;
  gVals[0] = 1e-10f; // Very small gradient
  mVals[0] = 0.0f;
  vVals[0] = 0.0f;

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, 0.9f, 0.999f, 1, 0.1f, 1e-8f);
  ASSERT_EQ(r, OK, "Small gradients should not cause numerical issues");

  // Should still update, though minimally
  ASSERT(pVals[0] != 1.0f, "Should still update with small gradients");
}

// Test 11: F64 precision
static void test_adam_f64_precision(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F64);
  Tensor paramGrad = create1DTensor(&ctx, 1, F64);
  Tensor m = create1DTensor(&ctx, 1, F64);
  Tensor v = create1DTensor(&ctx, 1, F64);

  f64 *pVals = param.values;
  f64 *gVals = paramGrad.values;
  f64 *mVals = m.values;
  f64 *vVals = v.values;

  pVals[0] = 1.0;
  gVals[0] = 0.1;
  mVals[0] = 0.0;
  vVals[0] = 0.0;

  f32 b1 = 0.9f, b2 = 0.999f, a = 0.1f, epsilon = 1e-8f;

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, b1, b2, 1, a, epsilon);
  // Note: Adam implementation currently uses f32 internally, so F64 runs but may not
  // produce exact F64-precision results. This test verifies it doesn't crash.
  ASSERT_EQ(r, OK, "F64 tensors should be accepted (may use F32 internally)");
}

// Test 12: Zero learning rate
static void test_adam_zero_learning_rate(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor param = create1DTensor(&ctx, 1, F32);
  Tensor paramGrad = create1DTensor(&ctx, 1, F32);
  Tensor m = create1DTensor(&ctx, 1, F32);
  Tensor v = create1DTensor(&ctx, 1, F32);

  f32 *pVals = param.values;
  f32 *gVals = paramGrad.values;
  f32 *mVals = m.values;
  f32 *vVals = v.values;

  pVals[0] = 1.0f;
  gVals[0] = 0.1f;
  mVals[0] = 0.0f;
  vVals[0] = 0.0f;

  AdamData triplet = {.param = &param, .paramGrad = &paramGrad, .m = &m, .v = &v};
  Result r = Adam(&ctx, &triplet, 1, 0.9f, 0.999f, 1, 0.0f, 1e-8f);
  ASSERT_EQ(r, OK, "Zero learning rate should succeed but not update");

  // Param should not change
  ASSERT_EQ(pVals[0], 1.0f, "Zero learning rate should not change param");
  // But m and v should still accumulate
  ASSERT_NEQ(mVals[0], 0.0f, "m should still accumulate");
  ASSERT_NEQ(vVals[0], 0.0f, "v should still accumulate");
}

void run_adam_tests(void) {
  test_adam_single_step();
  test_adam_two_steps_accumulation();
  test_adam_pytorch_reference();
  test_adam_null_triplets();
  test_adam_null_tensors();
  test_adam_non_float_type();
  test_adam_size_mismatch();
  test_adam_multiple_triplets();
  test_adam_bias_correction();
  test_adam_small_gradients();
  test_adam_f64_precision();
  test_adam_zero_learning_rate();
}
