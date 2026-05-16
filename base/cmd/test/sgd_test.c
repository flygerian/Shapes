#include "common.h"
#include "tensor/tensor_internal.h"
#include "test.h"
#include "../../shapes.h"
#include "utils_lib/array.h"
#include <math.h>

static void test_sgd_updates_f32_parameters(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(3), F32);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(3), F32);

  f32 *pVals = p->values;
  f32 *gVals = g->values;
  pVals[0] = 1.0f;
  pVals[1] = 2.0f;
  pVals[2] = 3.0f;
  gVals[0] = 0.1f;
  gVals[1] = 0.2f;
  gVals[2] = 0.3f;

  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);
  Result r = Sgd(&ctx, params, 0.5f);

  ASSERT_EQ(r, OK, "SGD should return OK for valid F32 tensors");
  ASSERT(fabsf(pVals[0] - 0.95f) < 1e-6f, "p[0] should be updated");
  ASSERT(fabsf(pVals[1] - 1.9f) < 1e-6f, "p[1] should be updated");
  ASSERT(fabsf(pVals[2] - 2.85f) < 1e-6f, "p[2] should be updated");
}

static void test_sgd_updates_f64_parameters(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(3), F64);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(3), F64);

  f64 *pVals = p->values;
  f64 *gVals = g->values;
  pVals[0] = 10.0;
  pVals[1] = -2.0;
  gVals[0] = 0.5;
  gVals[1] = -1.0;

  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);
  f32 learningRate = 0.1f;
  Result r = Sgd(&ctx, params, learningRate);
  f64 expected0 = 10.0 - (0.5 * (f64)learningRate);
  f64 expected1 = -2.0 - (-1.0 * (f64)learningRate);

  ASSERT_EQ(r, OK, "SGD should return OK for valid F64 tensors");
  ASSERT(fabs(pVals[0] - expected0) < 1e-12, "F64 p[0] should be updated");
  ASSERT(fabs(pVals[1] - expected1) < 1e-12, "F64 p[1] should be updated");
}

static void test_sgd_null_inputs(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Result r = Sgd(&ctx, NULL, 0.1f);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "NULL parameter array should return ERR_NULL_TENSOR_PROVIDED");
}

static void test_sgd_invalid_learning_rate(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(1), F32);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(1), F32);
  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);

  Result r = Sgd(&ctx, params, 0.0f);
  ASSERT_EQ(r, ERR_LEARNING_RATE_CANNOT_BE_ZERO_OR_NEGATIVE, "zero learning rate should return dedicated error");
}

static void test_sgd_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(2), F32);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(2), F64);
  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);

  Result r = Sgd(&ctx, params, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_GRAD_DTYPE_MISMATCH, "dtype mismatch should return SGD dtype error");
}

static void test_sgd_size_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(3), F32);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(2), F32);
  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);

  Result r = Sgd(&ctx, params, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_NUMBER_MISMATCH, "size mismatch should return SGD size error");
}

static void test_sgd_requires_float_tensors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *p = t_Zeros(&ctx, SHAPE1D(2), I32);
  Tensor *g = t_Zeros(&ctx, SHAPE1D(2), I32);
  p->grad = g;

  Array *params = MakeArray(mem, sizeof(Tensor *), 1);
  Array_Append(params, &p);

  Result r = Sgd(&ctx, params, 0.01f);
  ASSERT_EQ(r, ERR_SGD_PARAMS_HAVE_TO_BE_FLOAT, "non-float tensors should be rejected");
}

void run_sgd_tests(void) {
  test_sgd_updates_f32_parameters();
  test_sgd_updates_f64_parameters();
  test_sgd_null_inputs();
  test_sgd_invalid_learning_rate();
  test_sgd_dtype_mismatch();
  test_sgd_size_mismatch();
  test_sgd_requires_float_tensors();
}
