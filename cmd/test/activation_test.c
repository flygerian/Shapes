#include "test.h"
#include "../../activation/activation.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_internal.h"
#include <math.h>
#include <stdio.h>

static void test_tanh_scalar_zero(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 0.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;
  ASSERT(fabsf(output[0] - 0.0f) < 1e-6, "tanh(0) should be 0");

  freeMemory(mem);
}

static void test_tanh_scalar_one(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 1.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;
  // tanh(1) ≈ 0.7615941559557649
  ASSERT(fabsf(output[0] - 0.7615941559557649f) < 1e-6, "tanh(1) should be ~0.7616");

  freeMemory(mem);
}

static void test_tanh_scalar_negative(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = -1.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;
  // tanh(-1) ≈ -0.7615941559557649
  ASSERT(fabsf(output[0] - (-0.7615941559557649f)) < 1e-6, "tanh(-1) should be ~-0.7616");

  freeMemory(mem);
}

static void test_tanh_large_positive(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = 10.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;
  // tanh(10) should be very close to 1
  ASSERT(fabsf(output[0] - 1.0f) < 1e-6, "tanh(10) should be ~1.0");

  freeMemory(mem);
}

static void test_tanh_large_negative(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = -10.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;
  // tanh(-10) should be very close to -1
  ASSERT(fabsf(output[0] - (-1.0f)) < 1e-6, "tanh(-10) should be ~-1.0");

  freeMemory(mem);
}

static void test_tanh_1d_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {4};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = -1.0f;
  values[1] = 0.0f;
  values[2] = 1.0f;
  values[3] = 2.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed");
  f32 *output = (f32 *)result.values;

  ASSERT(fabsf(output[0] - tanhf(-1.0f)) < 1e-6, "tanh(-1) should match");
  ASSERT(fabsf(output[1] - tanhf(0.0f)) < 1e-6, "tanh(0) should match");
  ASSERT(fabsf(output[2] - tanhf(1.0f)) < 1e-6, "tanh(1) should match");
  ASSERT(fabsf(output[3] - tanhf(2.0f)) < 1e-6, "tanh(2) should match");

  freeMemory(mem);
}

static void test_tanh_2d_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2}, F32);
  f32 *values = (f32 *)t->values;
  values[0] = -2.0f;
  values[1] = -1.0f;
  values[2] = 0.0f;
  values[3] = 1.0f;
  values[4] = 2.0f;
  values[5] = 3.0f;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed on 2D tensor");
  ASSERT_EQ(result.shape.numOfDims, 2, "output should be 2D");
  ASSERT_EQ(result.shape.dims[0], 2, "first dim should be 2");
  ASSERT_EQ(result.shape.dims[1], 3, "second dim should be 3");

  f32 *output = (f32 *)result.values;
  for (u32 i = 0; i < 6; i++) {
    f32 expected = tanhf(values[i]);
    ASSERT(fabsf(output[i] - expected) < 1e-6, "tanh value should match");
  }

  freeMemory(mem);
}

static void test_tanh_f64_dtype(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2};
  Tensor *t = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F64);
  f64 *values = (f64 *)t->values;
  values[0] = 0.5;
  values[1] = -0.5;

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_EQ(res, OK, "Tanh should succeed on F64");
  f64 *output = (f64 *)result.values;
  ASSERT(fabs(output[0] - tanh(0.5)) < 1e-10, "tanh(0.5) should match (F64)");
  ASSERT(fabs(output[1] - tanh(-0.5)) < 1e-10, "tanh(-0.5) should match (F64)");

  freeMemory(mem);
}

static void test_tanh_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor result;
  Result res = Tanh(&ctx, NULL, &result);

  ASSERT_NEQ(res, OK, "Tanh should fail on NULL tensor");
  ASSERT_EQ(res, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED");

  freeMemory(mem);
}

static void test_tanh_invalid_dtype(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 5);

  Tensor result;
  Result res = Tanh(&ctx, t, &result);

  ASSERT_NEQ(res, OK, "Tanh should fail on non-float dtype");
  ASSERT_EQ(res, ERR_TANH_VALUE_NOT_FLOAT, "should return ERR_TANH_VALUE_NOT_FLOAT");

  freeMemory(mem);
}

// Gradient/Backward Pass Tests

static void test_tanh_backward_scalar(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {1};

  // Create input tensor with a computation node
  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *input_values = (f32 *)input->values;
  input_values[0] = 0.5f; // tanh(0.5) ≈ 0.4621

  // Create a graph node for input so it can receive gradients
  GraphNode *input_node = allocate(ctx.memory, sizeof(GraphNode));
  input_node->grad = t_Zeros(&ctx, input->shape, input->dtype);
  input->computation = input_node;

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  f32 *output_values = (f32 *)output.values;
  f32 tanh_output = output_values[0];

  // Set gradient on output (simulating backprop from next layer)
  f32 *grad_output = (f32 *)output.computation->grad->values;
  grad_output[0] = 1.0f; // gradient from loss

  // Backward pass
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Tanh backward should succeed");

  // Check gradient: d(tanh(x))/dx = 1 - tanh^2(x)
  f32 expected_grad = 1.0f - tanh_output * tanh_output;
  f32 *input_grad = (f32 *)input_node->grad->values;

  ASSERT(fabsf(input_grad[0] - expected_grad) < 1e-6,
         "Gradient should be 1 - tanh^2(output)");

  freeMemory(mem);
}

static void test_tanh_backward_vector(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {4};

  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *input_values = (f32 *)input->values;
  input_values[0] = -1.0f;
  input_values[1] = 0.0f;
  input_values[2] = 0.5f;
  input_values[3] = 1.0f;

  GraphNode *input_node = allocate(ctx.memory, sizeof(GraphNode));
  input_node->grad = t_Zeros(&ctx, input->shape, input->dtype);
  input->computation = input_node;

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  // Set uniform gradient on output
  f32 *grad_output = (f32 *)output.computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    grad_output[i] = 1.0f;
  }

  // Backward pass
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Tanh backward should succeed");

  // Verify gradients
  f32 *output_values = (f32 *)output.values;
  f32 *input_grad = (f32 *)input_node->grad->values;

  for (u32 i = 0; i < 4; i++) {
    f32 expected = 1.0f - output_values[i] * output_values[i];
    ASSERT(fabsf(input_grad[i] - expected) < 1e-6,
           "Gradient should match formula for each element");
  }

  freeMemory(mem);
}

static void test_tanh_backward_with_scaled_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {3};

  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *input_values = (f32 *)input->values;
  input_values[0] = 0.0f;
  input_values[1] = 1.0f;
  input_values[2] = -0.5f;

  GraphNode *input_node = allocate(ctx.memory, sizeof(GraphNode));
  input_node->grad = t_Zeros(&ctx, input->shape, input->dtype);
  input->computation = input_node;

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  // Set different gradients on each output
  f32 *grad_output = (f32 *)output.computation->grad->values;
  grad_output[0] = 2.0f;
  grad_output[1] = 0.5f;
  grad_output[2] = 3.0f;

  // Backward pass
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Tanh backward should succeed");

  // Verify: grad_input = grad_output * (1 - tanh^2(output))
  f32 *output_values = (f32 *)output.values;
  f32 *input_grad = (f32 *)input_node->grad->values;

  for (u32 i = 0; i < 3; i++) {
    f32 local_grad = 1.0f - output_values[i] * output_values[i];
    f32 expected = grad_output[i] * local_grad;
    ASSERT(fabsf(input_grad[i] - expected) < 1e-6,
           "Gradient should be grad_output * local_gradient");
  }

  freeMemory(mem);
}

static void test_tanh_backward_f64(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {2};

  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F64);
  f64 *input_values = (f64 *)input->values;
  input_values[0] = 0.3;
  input_values[1] = -0.7;

  GraphNode *input_node = allocate(ctx.memory, sizeof(GraphNode));
  input_node->grad = t_Zeros(&ctx, input->shape, input->dtype);
  input->computation = input_node;

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  // Set gradients
  f64 *grad_output = (f64 *)output.computation->grad->values;
  grad_output[0] = 1.0;
  grad_output[1] = 2.0;

  // Backward pass
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Tanh backward should succeed on F64");

  // Verify gradients
  f64 *output_values = (f64 *)output.values;
  f64 *input_grad = (f64 *)input_node->grad->values;

  for (u32 i = 0; i < 2; i++) {
    f64 local_grad = 1.0 - output_values[i] * output_values[i];
    f64 expected = grad_output[i] * local_grad;
    ASSERT(fabs(input_grad[i] - expected) < 1e-10,
           "F64 gradient should match with high precision");
  }

  freeMemory(mem);
}

static void test_tanh_backward_no_grad_input(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {2};

  // Input WITHOUT computation node (no gradients needed)
  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *input_values = (f32 *)input->values;
  input_values[0] = 1.0f;
  input_values[1] = -1.0f;

  // No input->computation set, so input doesn't need gradients

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  // Set gradient on output
  f32 *grad_output = (f32 *)output.computation->grad->values;
  grad_output[0] = 1.0f;
  grad_output[1] = 1.0f;

  // Backward pass should succeed but not crash (input has no grad)
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Backward should succeed even if input needs no grad");

  freeMemory(mem);
}

static void test_tanh_backward_accumulation(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  u32 dims[] = {2};

  Tensor *input = t_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1}, F32);
  f32 *input_values = (f32 *)input->values;
  input_values[0] = 0.5f;
  input_values[1] = -0.5f;

  GraphNode *input_node = allocate(ctx.memory, sizeof(GraphNode));
  input_node->grad = t_Zeros(&ctx, input->shape, input->dtype);
  input->computation = input_node;

  // Pre-set some gradient (simulating accumulation from another operation)
  f32 *input_grad = (f32 *)input_node->grad->values;
  input_grad[0] = 0.5f;
  input_grad[1] = 1.0f;

  // Forward pass
  Tensor output;
  Result res = Tanh(&ctx, input, &output);
  ASSERT_EQ(res, OK, "Tanh forward should succeed");

  // Set gradient on output
  f32 *grad_output = (f32 *)output.computation->grad->values;
  grad_output[0] = 1.0f;
  grad_output[1] = 1.0f;

  // Backward pass
  res = output.computation->backward(&ctx, output.computation);
  ASSERT_EQ(res, OK, "Tanh backward should succeed");

  // Verify gradients are ACCUMULATED (not replaced)
  f32 *output_values = (f32 *)output.values;

  for (u32 i = 0; i < 2; i++) {
    f32 local_grad = 1.0f - output_values[i] * output_values[i];
    f32 new_grad = grad_output[i] * local_grad;
    f32 expected = (i == 0 ? 0.5f : 1.0f) + new_grad; // previous + new
    ASSERT(fabsf(input_grad[i] - expected) < 1e-6,
           "Gradients should accumulate (+=), not replace");
  }

  freeMemory(mem);
}

void run_activation_tests(void) {
  printf("=== Activation Tests ===\n");
  // Forward pass tests
  test_tanh_scalar_zero();
  test_tanh_scalar_one();
  test_tanh_scalar_negative();
  test_tanh_large_positive();
  test_tanh_large_negative();
  test_tanh_1d_tensor();
  test_tanh_2d_tensor();
  test_tanh_f64_dtype();
  test_tanh_null_tensor();
  test_tanh_invalid_dtype();

  // Backward pass tests
  test_tanh_backward_scalar();
  test_tanh_backward_vector();
  test_tanh_backward_with_scaled_grad();
  test_tanh_backward_f64();
  test_tanh_backward_no_grad_input();
  test_tanh_backward_accumulation();
}
