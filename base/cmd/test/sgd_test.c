#include "test.h"
#include "../../shapes.h"
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

static void test_sgd_updates_f32_parameters(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 3, F32);
  Tensor g = create1DTensor(&ctx, 3, F32);

  f32 *pVals = p.values;
  f32 *gVals = g.values;
  pVals[0] = 1.0f;
  pVals[1] = 2.0f;
  pVals[2] = 3.0f;
  gVals[0] = 0.1f;
  gVals[1] = 0.2f;
  gVals[2] = 0.3f;

  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};
  Result r = Sgd(&ctx, params, grads, 1, 0.5f);

  ASSERT_EQ(r, OK, "SGD should return OK for valid F32 tensors");
  ASSERT(fabsf(pVals[0] - 0.95f) < 1e-6f, "p[0] should be updated");
  ASSERT(fabsf(pVals[1] - 1.9f) < 1e-6f, "p[1] should be updated");
  ASSERT(fabsf(pVals[2] - 2.85f) < 1e-6f, "p[2] should be updated");

  freeMemory(mem);
}

static void test_sgd_updates_f64_parameters(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 2, F64);
  Tensor g = create1DTensor(&ctx, 2, F64);

  f64 *pVals = p.values;
  f64 *gVals = g.values;
  pVals[0] = 10.0;
  pVals[1] = -2.0;
  gVals[0] = 0.5;
  gVals[1] = -1.0;

  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};
  f32 learningRate = 0.1f;
  Result r = Sgd(&ctx, params, grads, 1, learningRate);
  f64 expected0 = 10.0 - (0.5 * (f64)learningRate);
  f64 expected1 = -2.0 - (-1.0 * (f64)learningRate);

  ASSERT_EQ(r, OK, "SGD should return OK for valid F64 tensors");
  ASSERT(fabs(pVals[0] - expected0) < 1e-12, "F64 p[0] should be updated");
  ASSERT(fabs(pVals[1] - expected1) < 1e-12, "F64 p[1] should be updated");

  freeMemory(mem);
}

static void test_sgd_null_inputs(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Result r = Sgd(&ctx, NULL, NULL, 0, 0.1f);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "NULL arrays should return ERR_NULL_TENSOR_PROVIDED");
  freeMemory(mem);
}

static void test_sgd_invalid_learning_rate(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 1, F32);
  Tensor g = create1DTensor(&ctx, 1, F32);
  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};

  Result r = Sgd(&ctx, params, grads, 1, 0.0f);
  ASSERT_EQ(r, ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE,
            "zero learning rate should return dedicated error");

  freeMemory(mem);
}

static void test_sgd_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 2, F32);
  Tensor g = create1DTensor(&ctx, 2, F64);
  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};

  Result r = Sgd(&ctx, params, grads, 1, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH, "dtype mismatch should return SGD dtype error");

  freeMemory(mem);
}

static void test_sgd_size_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 3, F32);
  Tensor g = create1DTensor(&ctx, 2, F32);
  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};

  Result r = Sgd(&ctx, params, grads, 1, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_NUMBER_MISMATCH, "size mismatch should return SGD size error");

  freeMemory(mem);
}

static void test_sgd_materializes_non_contiguous_tensors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 2, F32);
  Tensor g = create1DTensor(&ctx, 2, F32);
  p.isContigous = false;
  ((f32 *)p.values)[0] = 1.0f;
  ((f32 *)p.values)[1] = 2.0f;
  ((f32 *)g.values)[0] = 0.5f;
  ((f32 *)g.values)[1] = 1.0f;

  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};

  Result r = Sgd(&ctx, params, grads, 1, 0.01f);
  ASSERT_EQ(r, OK, "non-contiguous tensors should be materialized");
  ASSERT(fabsf(((f32 *)p.values)[0] - 0.995f) < 1e-6f, "p[0] should be updated after materialization");
  ASSERT(fabsf(((f32 *)p.values)[1] - 1.99f) < 1e-6f, "p[1] should be updated after materialization");

  freeMemory(mem);
}

static void test_sgd_requires_float_tensors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor p = create1DTensor(&ctx, 2, I32);
  Tensor g = create1DTensor(&ctx, 2, I32);
  Tensor *params[] = {&p};
  Tensor *grads[] = {&g};

  Result r = Sgd(&ctx, params, grads, 1, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT, "non-float tensors should be rejected");

  freeMemory(mem);
}

void run_sgd_tests(void) {
  test_sgd_updates_f32_parameters();
  test_sgd_updates_f64_parameters();
  test_sgd_null_inputs();
  test_sgd_invalid_learning_rate();
  test_sgd_dtype_mismatch();
  test_sgd_size_mismatch();
  test_sgd_materializes_non_contiguous_tensors();
  test_sgd_requires_float_tensors();
}
