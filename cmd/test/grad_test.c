#include "test.h"
#include "../../grad/grad.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_internal.h"
#include <math.h>
#include <stdio.h>

// Test basic addition backward without broadcasting
static void test_add_backward_no_broadcast(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create two [2, 3] tensors
  u32 dims[] = {2, 3};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set some values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  for (u32 i = 0; i < 6; i++) {
    a_vals[i] = (f32)i;
    b_vals[i] = (f32)i * 2;
  }

  // Forward pass: c = a + b
  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");

  // Set gradient on output
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 6; i++) {
    c_grad_vals[i] = 1.0f;  // All ones gradient
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "addBackward should succeed");

  // Check gradients accumulated to inputs
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;

  for (u32 i = 0; i < 6; i++) {
    ASSERT(fabsf(a_grad[i] - 1.0f) < 1e-6, "a gradient should be 1.0");
    ASSERT(fabsf(b_grad[i] - 1.0f) < 1e-6, "b gradient should be 1.0");
  }

  freeMemory(mem);
}

// Test addition backward with size-1 broadcasting: [3, 1] + [3, 5] -> [3, 5]
static void test_add_backward_broadcast_size1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create a: [3, 1] and b: [3, 5]
  u32 dims_a[] = {3, 1};
  u32 dims_b[] = {3, 5};
  Dim shape_a = {.dims = dims_a, .numOfDims = 2};
  Dim shape_b = {.dims = dims_b, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape_a, F32);
  Tensor *b = t_Zeros(&ctx, shape_b, F32);

  // Set values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  for (u32 i = 0; i < 3; i++) {
    a_vals[i] = (f32)i;
  }
  for (u32 i = 0; i < 15; i++) {
    b_vals[i] = (f32)i;
  }

  // Forward pass: c = a + b (result is [3, 5])
  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");
  ASSERT_EQ(c.shape.numOfDims, 2, "Output should be 2D");
  ASSERT_EQ(c.shape.dims[0], 3, "Output dim 0 should be 3");
  ASSERT_EQ(c.shape.dims[1], 5, "Output dim 1 should be 5");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "addBackward should succeed");

  // Check gradients
  // For a: gradient should be summed along dim 1: each element should be 5.0
  f32 *a_grad = (f32 *)a->computation->grad->values;
  for (u32 i = 0; i < 3; i++) {
    ASSERT(fabsf(a_grad[i] - 5.0f) < 1e-5, "a gradient should be 5.0 (summed across 5 elements)");
  }

  // For b: gradient should be 1.0 for each element (no broadcasting)
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    ASSERT(fabsf(b_grad[i] - 1.0f) < 1e-6, "b gradient should be 1.0");
  }

  freeMemory(mem);
}

// Test addition backward with dimension mismatch: [5] + [3, 5] -> [3, 5]
static void test_add_backward_broadcast_dim_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create a: [5] (1D) and b: [3, 5] (2D)
  u32 dims_a[] = {5};
  u32 dims_b[] = {3, 5};
  Dim shape_a = {.dims = dims_a, .numOfDims = 1};
  Dim shape_b = {.dims = dims_b, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape_a, F32);
  Tensor *b = t_Zeros(&ctx, shape_b, F32);

  // Set values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  for (u32 i = 0; i < 5; i++) {
    a_vals[i] = (f32)i;
  }
  for (u32 i = 0; i < 15; i++) {
    b_vals[i] = (f32)i * 0.5f;
  }

  // Forward pass: c = a + b (result is [3, 5])
  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");
  ASSERT_EQ(c.shape.numOfDims, 2, "Output should be 2D");
  ASSERT_EQ(c.shape.dims[0], 3, "Output dim 0 should be 3");
  ASSERT_EQ(c.shape.dims[1], 5, "Output dim 1 should be 5");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "addBackward should succeed");

  // Check gradients
  // For a: gradient should be summed along dim 0: each element should be 3.0
  f32 *a_grad = (f32 *)a->computation->grad->values;
  for (u32 i = 0; i < 5; i++) {
    ASSERT(fabsf(a_grad[i] - 3.0f) < 1e-5, "a gradient should be 3.0 (summed across 3 rows)");
  }

  // For b: gradient should be 1.0 for each element
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    ASSERT(fabsf(b_grad[i] - 1.0f) < 1e-6, "b gradient should be 1.0");
  }

  freeMemory(mem);
}

// Test addition backward with scalar broadcast: [1] + [3, 5] -> [3, 5]
static void test_add_backward_broadcast_scalar(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create a: [1] (scalar) and b: [3, 5]
  u32 dims_a[] = {1};
  u32 dims_b[] = {3, 5};
  Dim shape_a = {.dims = dims_a, .numOfDims = 1};
  Dim shape_b = {.dims = dims_b, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape_a, F32);
  Tensor *b = t_Zeros(&ctx, shape_b, F32);

  // Set values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 5.0f;
  for (u32 i = 0; i < 15; i++) {
    b_vals[i] = (f32)i;
  }

  // Forward pass: c = a + b
  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < c.size; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "addBackward should succeed");

  // Check gradients
  // For a: gradient should be summed over all 15 elements
  f32 *a_grad = (f32 *)a->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 15.0f) < 1e-5, "a gradient should be 15.0 (summed over all elements)");

  // For b: gradient should be 1.0 for each element
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    ASSERT(fabsf(b_grad[i] - 1.0f) < 1e-6, "b gradient should be 1.0");
  }

  freeMemory(mem);
}

// Test gradient accumulation across multiple operations
static void test_add_backward_gradient_accumulation(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create tensors
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  for (u32 i = 0; i < 4; i++) {
    a_vals[i] = (f32)i;
    b_vals[i] = (f32)i * 2;
  }

  // First operation: c = a + b
  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "First add should succeed");

  // Second operation: d = a + c (a is used again)
  Tensor d;
  res = Add(&ctx, a, &c, &d);
  ASSERT_EQ(res, OK, "Second add should succeed");

  // Set gradient on d
  f32 *d_grad_vals = (f32 *)d.computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    d_grad_vals[i] = 1.0f;
  }

  // Backward pass on d
  res = d.computation->backward(&ctx, d.computation);
  ASSERT_EQ(res, OK, "d backward should succeed");

  // Now backward pass on c
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    c_grad_vals[i] = 1.0f;  // Set gradient from d
  }

  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "c backward should succeed");

  // Check that a's gradient accumulated from both operations
  // From d: 1.0, from c: 1.0, total: 2.0
  f32 *a_grad = (f32 *)a->computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    ASSERT(fabsf(a_grad[i] - 2.0f) < 1e-5, "a gradient should accumulate to 2.0");
  }

  freeMemory(mem);
}

void run_grad_tests(void) {
  test_add_backward_no_broadcast();
  test_add_backward_broadcast_size1();
  test_add_backward_broadcast_dim_mismatch();
  test_add_backward_broadcast_scalar();
  test_add_backward_gradient_accumulation();
}
