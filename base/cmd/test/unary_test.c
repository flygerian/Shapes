#include "test.h"
#include "../../shapes.h"
#include "../../tensor/tensor_internal.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static bool hasCudaDevice(void) {
  int deviceCount = 0;
  return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

static void assertMovedF32Values(Context *srcCtx, Tensor *tensor, const f32 *expected,
                                 tensor_size_t size, const char *msg) {
  Context cpuCtx = {.memory = srcCtx->memory};
  Result moveResult = moveTensor(srcCtx, &cpuCtx, tensor);
  ASSERT_EQ(moveResult, OK, msg);

  f32 *values = tensor->values;
  for (tensor_size_t i = 0; i < size; i++) {
    ASSERT(fabsf(values[i] - expected[i]) < 1e-6f, msg);
  }
}

static void assertMovedI32Values(Context *srcCtx, Tensor *tensor, const i32 *expected,
                                 tensor_size_t size, const char *msg) {
  Context cpuCtx = {.memory = srcCtx->memory};
  Result moveResult = moveTensor(srcCtx, &cpuCtx, tensor);
  ASSERT_EQ(moveResult, OK, msg);

  i32 *values = tensor->values;
  for (tensor_size_t i = 0; i < size; i++) {
    ASSERT_EQ(values[i], expected[i], msg);
  }
}

// ===== Pow Tests =====

static void test_pow_scalar_power_of_2(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 3.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 2.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed");
  f32 *output = (f32 *)result.values;
  ASSERT(fabsf(output[0] - 9.0f) < 1e-6, "3^2 should be 9");

  freeMemory(mem);
}

static void test_pow_scalar_power_of_3(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 2.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 3.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed");
  f32 *output = (f32 *)result.values;
  ASSERT(fabsf(output[0] - 8.0f) < 1e-6, "2^3 should be 8");

  freeMemory(mem);
}

static void test_pow_power_of_0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 5.0f;
  values[1] = 10.0f;
  values[2] = -3.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 0.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed");
  f32 *output = (f32 *)result.values;

  // x^0 = 1 for any x
  ASSERT(fabsf(output[0] - 1.0f) < 1e-6, "any number^0 should be 1");
  ASSERT(fabsf(output[1] - 1.0f) < 1e-6, "any number^0 should be 1");
  ASSERT(fabsf(output[2] - 1.0f) < 1e-6, "any number^0 should be 1");

  freeMemory(mem);
}

static void test_pow_power_of_1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 2.5f;
  values[1] = -7.3f;
  values[2] = 0.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 1.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed");
  f32 *output = (f32 *)result.values;

  // x^1 = x (identity)
  ASSERT(fabsf(output[0] - 2.5f) < 1e-6, "x^1 should be x");
  ASSERT(fabsf(output[1] - (-7.3f)) < 1e-6, "x^1 should be x");
  ASSERT(fabsf(output[2] - 0.0f) < 1e-6, "x^1 should be x");

  freeMemory(mem);
}

static void test_pow_negative_power(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 2.0f;
  values[1] = 4.0f;

  Tensor result;
  Result res = Pow(&ctx, t, -1.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed with negative power");
  f32 *output = (f32 *)result.values;

  // x^-1 = 1/x
  ASSERT(fabsf(output[0] - 0.5f) < 1e-6, "2^-1 should be 0.5");
  ASSERT(fabsf(output[1] - 0.25f) < 1e-6, "4^-1 should be 0.25");

  freeMemory(mem);
}

static void test_pow_fractional_power(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 4.0f;
  values[1] = 9.0f;
  values[2] = 16.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 0.5f, &result); // Square root

  ASSERT_EQ(res, OK, "Pow should succeed with fractional power");
  f32 *output = (f32 *)result.values;

  // x^0.5 = sqrt(x)
  ASSERT(fabsf(output[0] - 2.0f) < 1e-6, "4^0.5 should be 2");
  ASSERT(fabsf(output[1] - 3.0f) < 1e-6, "9^0.5 should be 3");
  ASSERT(fabsf(output[2] - 4.0f) < 1e-6, "16^0.5 should be 4");

  freeMemory(mem);
}

static void test_pow_2d_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 1.0f;
  values[1] = 2.0f;
  values[2] = 3.0f;
  values[3] = 4.0f;
  values[4] = 5.0f;
  values[5] = 6.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 2.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed on 2D tensor");
  ASSERT_EQ(result.shape.numOfDims, 2, "output should be 2D");
  ASSERT_EQ(result.shape.dims[0], 2, "first dim should be 2");
  ASSERT_EQ(result.shape.dims[1], 3, "second dim should be 3");

  f32 *output = (f32 *)result.values;
  for (u32 i = 0; i < 6; i++) {
    f32 expected = values[i] * values[i];
    ASSERT(fabsf(output[i] - expected) < 1e-6, "each element should be squared");
  }

  freeMemory(mem);
}

static void test_pow_f64_dtype(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F64);
  f64 *values = (f64 *)t->values;
  values[0] = 2.0;
  values[1] = 3.0;

  Tensor result;
  Result res = Pow(&ctx, t, 3.0f, &result);

  ASSERT_EQ(res, OK, "Pow should succeed on F64");
  f64 *output = (f64 *)result.values;
  ASSERT(fabs(output[0] - 8.0) < 1e-10, "2^3 should be 8 (F64)");
  ASSERT(fabs(output[1] - 27.0) < 1e-10, "3^3 should be 27 (F64)");

  freeMemory(mem);
}

static void test_pow_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor result;
  Result res = Pow(&ctx, NULL, 2.0f, &result);

  ASSERT_NEQ(res, OK, "Pow should fail on NULL tensor");
  ASSERT_EQ(res, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED");

  freeMemory(mem);
}

static void test_pow_invalid_dtype(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 5);

  Tensor result;
  Result res = Pow(&ctx, t, 2.0f, &result);

  ASSERT_NEQ(res, OK, "Pow should fail on non-float dtype");
  ASSERT_EQ(res, ERR_POW_VALUE_NOT_FLOAT, "should return ERR_POW_VALUE_NOT_FLOAT");

  freeMemory(mem);
}

static void test_relu_forward(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = -1.0f;
  values[1] = 0.0f;
  values[2] = 2.5f;

  Tensor result;
  Result res = Relu(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Relu should succeed");
  f32 *output = (f32 *)result.values;
  ASSERT(fabsf(output[0] - 0.0f) < 1e-6, "relu(-1) should be 0");
  ASSERT(fabsf(output[1] - 0.0f) < 1e-6, "relu(0) should be 0");
  ASSERT(fabsf(output[2] - 2.5f) < 1e-6, "relu(2.5) should be 2.5");

  freeMemory(mem);
}

static void test_relu_invalid_dtype(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 5);

  Tensor result;
  Result res = Relu(&ctx, t, &result);

  ASSERT_NEQ(res, OK, "Relu should fail on non-float dtype");
  ASSERT_EQ(res, ERR_RELU_VALUE_NOT_FLOAT, "should return ERR_RELU_VALUE_NOT_FLOAT");

  freeMemory(mem);
}

static void test_pow_cuda_dispatch_materializes_cpu_inputs(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};

  dim_t dims[] = {3};
  Tensor *t = t_Zeros(&hostCtx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 2.0f;
  values[1] = 3.0f;
  values[2] = 4.0f;

  Tensor result;
  Result res = Pow(&ctx, t, 2.0f, &result);

  f32 expected[] = {4.0f, 9.0f, 16.0f};
  ASSERT_EQ(res, OK, "CUDA Pow should succeed");
  ASSERT(result.context == &ctx, "CUDA Pow result should live on the CUDA context");
  assertMovedF32Values(&ctx, &result, expected, 3, "CUDA Pow result should match");

  DestroyContext(&ctx);
}

static void test_negate_cuda_dispatch_i32(void) {
  if (!hasCudaDevice()) {
    return;
  }

  Context ctx = InitializeContext((size_t)1024 * 1024, 1, true);
  Context hostCtx = {.memory = ctx.memory};

  dim_t dims[] = {4};
  Tensor *t = t_Zeros(&hostCtx, (Dim){.dims = dims, .numOfDims = 1}, I32);
  i32 *values = (i32 *)t->values;
  values[0] = 1;
  values[1] = -2;
  values[2] = 0;
  values[3] = 7;

  Tensor result;
  Result res = Negate(&ctx, t, &result);

  i32 expected[] = {-1, 2, 0, -7};
  ASSERT_EQ(res, OK, "CUDA Negate should succeed on I32");
  ASSERT(result.context == &ctx, "CUDA Negate result should live on the CUDA context");
  assertMovedI32Values(&ctx, &result, expected, 4, "CUDA Negate result should match");

  DestroyContext(&ctx);
}

void run_unary_tests(void) {
  printf("=== Unary Operation Tests ===\n");

  // Pow forward pass tests
  test_pow_scalar_power_of_2();
  test_pow_scalar_power_of_3();
  test_pow_power_of_0();
  test_pow_power_of_1();
  test_pow_negative_power();
  test_pow_fractional_power();
  test_pow_2d_tensor();
  test_pow_f64_dtype();
  test_pow_null_tensor();
  test_pow_invalid_dtype();
  test_relu_forward();
  test_relu_invalid_dtype();
  test_pow_cuda_dispatch_materializes_cpu_inputs();
  test_negate_cuda_dispatch_i32();
}
