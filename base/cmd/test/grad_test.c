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

// Test basic multiplication backward without broadcasting
static void test_multiply_backward_no_broadcast(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create two [2, 2] tensors
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values: a = [[2, 3], [4, 5]], b = [[1, 2], [3, 4]]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 2.0f; a_vals[1] = 3.0f; a_vals[2] = 4.0f; a_vals[3] = 5.0f;
  b_vals[0] = 1.0f; b_vals[1] = 2.0f; b_vals[2] = 3.0f; b_vals[3] = 4.0f;

  // Forward pass: c = a * b
  Tensor c;
  Result res = Multiply(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "multiplyBackward should succeed");

  // Check gradients: grad_a = grad_c * b, grad_b = grad_c * a
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;

  // grad_a should be [1*1, 1*2, 1*3, 1*4] = [1, 2, 3, 4]
  ASSERT(fabsf(a_grad[0] - 1.0f) < 1e-6, "a grad[0] should be 1.0");
  ASSERT(fabsf(a_grad[1] - 2.0f) < 1e-6, "a grad[1] should be 2.0");
  ASSERT(fabsf(a_grad[2] - 3.0f) < 1e-6, "a grad[2] should be 3.0");
  ASSERT(fabsf(a_grad[3] - 4.0f) < 1e-6, "a grad[3] should be 4.0");

  // grad_b should be [1*2, 1*3, 1*4, 1*5] = [2, 3, 4, 5]
  ASSERT(fabsf(b_grad[0] - 2.0f) < 1e-6, "b grad[0] should be 2.0");
  ASSERT(fabsf(b_grad[1] - 3.0f) < 1e-6, "b grad[1] should be 3.0");
  ASSERT(fabsf(b_grad[2] - 4.0f) < 1e-6, "b grad[2] should be 4.0");
  ASSERT(fabsf(b_grad[3] - 5.0f) < 1e-6, "b grad[3] should be 5.0");

  freeMemory(mem);
}

// Test multiplication backward with broadcasting: [3, 1] * [3, 5] -> [3, 5]
static void test_multiply_backward_broadcast_size1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create a: [3, 1] and b: [3, 5]
  u32 dims_a[] = {3, 1};
  u32 dims_b[] = {3, 5};
  Dim shape_a = {.dims = dims_a, .numOfDims = 2};
  Dim shape_b = {.dims = dims_b, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape_a, F32);
  Tensor *b = t_Zeros(&ctx, shape_b, F32);

  // Set values: a = [[2], [3], [4]], b = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 2.0f; a_vals[1] = 3.0f; a_vals[2] = 4.0f;
  for (u32 i = 0; i < 15; i++) {
    b_vals[i] = (f32)(i / 5 + 1);  // Row-wise: 1s, then 2s, then 3s
  }

  // Forward pass: c = a * b
  Tensor c;
  Result res = Multiply(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "multiplyBackward should succeed");

  // Check gradients
  // For a: grad should be summed across columns
  // Row 0: sum of b[0,:] = 1+1+1+1+1 = 5
  // Row 1: sum of b[1,:] = 2+2+2+2+2 = 10
  // Row 2: sum of b[2,:] = 3+3+3+3+3 = 15
  f32 *a_grad = (f32 *)a->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 5.0f) < 1e-5, "a grad[0] should be 5.0");
  ASSERT(fabsf(a_grad[1] - 10.0f) < 1e-5, "a grad[1] should be 10.0");
  ASSERT(fabsf(a_grad[2] - 15.0f) < 1e-5, "a grad[2] should be 15.0");

  // For b: grad = output_grad * a (broadcasted)
  // Each element in row i gets multiplied by a[i]
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 5; i++) {
    ASSERT(fabsf(b_grad[i] - 2.0f) < 1e-6, "b grad row 0 should be 2.0");
  }
  for (u32 i = 5; i < 10; i++) {
    ASSERT(fabsf(b_grad[i] - 3.0f) < 1e-6, "b grad row 1 should be 3.0");
  }
  for (u32 i = 10; i < 15; i++) {
    ASSERT(fabsf(b_grad[i] - 4.0f) < 1e-6, "b grad row 2 should be 4.0");
  }

  freeMemory(mem);
}

// Test multiplication backward with scalar: [1] * [3, 5] -> [3, 5]
static void test_multiply_backward_scalar(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create a: [1] and b: [3, 5]
  u32 dims_a[] = {1};
  u32 dims_b[] = {3, 5};
  Dim shape_a = {.dims = dims_a, .numOfDims = 1};
  Dim shape_b = {.dims = dims_b, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape_a, F32);
  Tensor *b = t_Zeros(&ctx, shape_b, F32);

  // Set values: a = [2.0], b = [1, 2, 3, ..., 15]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 2.0f;
  for (u32 i = 0; i < 15; i++) {
    b_vals[i] = (f32)(i + 1);
  }

  // Forward pass: c = a * b
  Tensor c;
  Result res = Multiply(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Set gradient on output (all ones)
  f32 *c_grad_vals = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    c_grad_vals[i] = 1.0f;
  }

  // Backward pass
  res = c.computation->backward(&ctx, c.computation);
  ASSERT_EQ(res, OK, "multiplyBackward should succeed");

  // Check gradients
  // For a: grad = sum(output_grad * b) = sum([1*1, 1*2, ..., 1*15]) = 1+2+...+15 = 120
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 expected_a_grad = (15.0f * 16.0f) / 2.0f;  // Sum of 1 to 15
  ASSERT(fabsf(a_grad[0] - expected_a_grad) < 1e-4, "a grad should be sum of 1 to 15 (120)");

  // For b: grad = output_grad * a = [1*2, 1*2, ...] = all 2.0
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 15; i++) {
    ASSERT(fabsf(b_grad[i] - 2.0f) < 1e-6, "b grad should all be 2.0");
  }

  freeMemory(mem);
}

// Test topological sort creates correct computation graph
static void test_init_computation_graph_simple(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create simple computation: c = a + b
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");

  // Build computation graph
  ComputationGraph *graph = InitComputationGraph(&ctx, &c);
  ASSERT_NOT_NULL(graph, "Computation graph should be created");
  ASSERT_NOT_NULL(graph->nodes, "Graph nodes should be allocated");

  // Graph should have 3 nodes: a, b, c (in topological order)
  ASSERT_EQ(graph->size, 3, "Graph should have 3 nodes");

  // Verify topological order: inputs before outputs
  // The order should be: a's computation, b's computation, c's computation
  ASSERT_EQ(graph->nodes[2], c.computation, "Last node should be output (c)");

  freeMemory(mem);
}

// Test topological sort with multi-level computation graph
static void test_init_computation_graph_multilevel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: d = (a + b) * c
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);
  Tensor *c = t_Zeros(&ctx, shape, F32);

  Tensor temp;
  Result res = Add(&ctx, a, b, &temp);
  ASSERT_EQ(res, OK, "Add should succeed");

  Tensor d;
  res = Multiply(&ctx, &temp, c, &d);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Build computation graph
  ComputationGraph *graph = InitComputationGraph(&ctx, &d);
  ASSERT_NOT_NULL(graph, "Computation graph should be created");

  // Graph should have 5 nodes: a, b, c, temp, d
  ASSERT_EQ(graph->size, 5, "Graph should have 5 nodes");

  // Last node should be the output
  ASSERT_EQ(graph->nodes[4], d.computation, "Last node should be output (d)");

  // temp should come before d
  int temp_idx = -1;
  int d_idx = -1;
  for (size_t i = 0; i < graph->size; i++) {
    if (graph->nodes[i] == temp.computation) temp_idx = i;
    if (graph->nodes[i] == d.computation) d_idx = i;
  }
  ASSERT(temp_idx >= 0, "temp should be in graph");
  ASSERT(d_idx >= 0, "d should be in graph");
  ASSERT(temp_idx < d_idx, "temp should come before d in topological order");

  freeMemory(mem);
}

// Test topological sort handles shared inputs (diamond dependency)
static void test_init_computation_graph_diamond(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create diamond computation: d = a + a (a is used twice)
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);

  Tensor d;
  Result res = Add(&ctx, a, a, &d);
  ASSERT_EQ(res, OK, "Add should succeed");

  // Build computation graph
  ComputationGraph *graph = InitComputationGraph(&ctx, &d);
  ASSERT_NOT_NULL(graph, "Computation graph should be created");

  // Graph should have 2 nodes: a (once), d
  // The 'a' node should only appear once even though it's used twice
  ASSERT_EQ(graph->size, 2, "Graph should have 2 nodes (a should appear only once)");

  freeMemory(mem);
}

// Test full backward pass on simple computation
static void test_backward_simple_add(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: c = a + b
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  for (u32 i = 0; i < 4; i++) {
    a_vals[i] = (f32)i;
    b_vals[i] = (f32)(i + 1);
  }

  Tensor c;
  Result res = Add(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Add should succeed");

  // Build computation graph
  ComputationGraph *graph = InitComputationGraph(&ctx, &c);
  ASSERT_NOT_NULL(graph, "Computation graph should be created");

  // Run backward pass (this sets c's grad to 1.0 and propagates)
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check that c's gradient was set to 1.0
  f32 *c_grad = (f32 *)c.computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    ASSERT(fabsf(c_grad[i] - 1.0f) < 1e-6, "c gradient should be 1.0");
  }

  // Check that gradients propagated to a and b
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    ASSERT(fabsf(a_grad[i] - 1.0f) < 1e-6, "a gradient should be 1.0");
    ASSERT(fabsf(b_grad[i] - 1.0f) < 1e-6, "b gradient should be 1.0");
  }

  freeMemory(mem);
}

// Test full backward pass on multi-operation computation
static void test_backward_multilevel(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: d = (a + b) * c
  // Expected gradients:
  //   grad_d = 1.0
  //   grad_c = grad_d * (a + b)
  //   grad_temp = grad_d * c
  //   grad_a = grad_temp * 1 = grad_d * c
  //   grad_b = grad_temp * 1 = grad_d * c

  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);
  Tensor *c = t_Zeros(&ctx, shape, F32);

  // Set values: a = [1, 2, 3, 4], b = [1, 1, 1, 1], c = [2, 2, 2, 2]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  f32 *c_vals = (f32 *)c->values;
  for (u32 i = 0; i < 4; i++) {
    a_vals[i] = (f32)(i + 1);
    b_vals[i] = 1.0f;
    c_vals[i] = 2.0f;
  }

  Tensor temp;
  Result res = Add(&ctx, a, b, &temp);
  ASSERT_EQ(res, OK, "Add should succeed");

  Tensor d;
  res = Multiply(&ctx, &temp, c, &d);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &d);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check gradients
  // grad_a = grad_b = c = [2, 2, 2, 2]
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    ASSERT(fabsf(a_grad[i] - 2.0f) < 1e-5, "a gradient should be 2.0 (value of c)");
    ASSERT(fabsf(b_grad[i] - 2.0f) < 1e-5, "b gradient should be 2.0 (value of c)");
  }

  // grad_c = (a + b) = [2, 3, 4, 5]
  f32 *c_grad = (f32 *)c->computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    f32 expected = a_vals[i] + b_vals[i];
    ASSERT(fabsf(c_grad[i] - expected) < 1e-5, "c gradient should be a + b");
  }

  freeMemory(mem);
}

// Test backward pass with shared input (gradient accumulation)
static void test_backward_shared_input(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: d = a * a (a is used twice)
  // grad_a should accumulate from both uses: grad_a = a + a = 2*a
  u32 dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};

  Tensor *a = t_Zeros(&ctx, shape, F32);

  // Set values: a = [1, 2, 3, 4]
  f32 *a_vals = (f32 *)a->values;
  for (u32 i = 0; i < 4; i++) {
    a_vals[i] = (f32)(i + 1);
  }

  Tensor d;
  Result res = Multiply(&ctx, a, a, &d);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &d);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // For f(a) = a * a, df/da = 2*a
  // grad_a = grad_d * a + grad_d * a = 1*a + 1*a = 2*a
  f32 *a_grad = (f32 *)a->computation->grad->values;
  for (u32 i = 0; i < 4; i++) {
    f32 expected = 2.0f * a_vals[i];
    ASSERT(fabsf(a_grad[i] - expected) < 1e-5, "a gradient should be 2*a");
  }

  freeMemory(mem);
}

// Test backward pass with complex computation graph
static void test_backward_complex_graph(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: f = (a + b) * (a + c)
  // This creates a diamond-shaped graph where 'a' is used in both branches
  u32 dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);
  Tensor *c = t_Zeros(&ctx, shape, F32);

  // Set values: a = [2, 3], b = [1, 1], c = [1, 2]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  f32 *c_vals = (f32 *)c->values;
  a_vals[0] = 2.0f; a_vals[1] = 3.0f;
  b_vals[0] = 1.0f; b_vals[1] = 1.0f;
  c_vals[0] = 1.0f; c_vals[1] = 2.0f;

  Tensor left;
  Result res = Add(&ctx, a, b, &left);  // left = a + b = [3, 4]
  ASSERT_EQ(res, OK, "Add should succeed");

  Tensor right;
  res = Add(&ctx, a, c, &right);  // right = a + c = [3, 5]
  ASSERT_EQ(res, OK, "Add should succeed");

  Tensor f;
  res = Multiply(&ctx, &left, &right, &f);  // f = left * right = [9, 20]
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &f);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Verify gradients:
  // df/da = df/dleft * dleft/da + df/dright * dright/da
  //       = right * 1 + left * 1 = right + left
  //       = [3, 5] + [3, 4] = [6, 9]
  f32 *a_grad = (f32 *)a->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 6.0f) < 1e-5, "a gradient[0] should be 6.0");
  ASSERT(fabsf(a_grad[1] - 9.0f) < 1e-5, "a gradient[1] should be 9.0");

  // df/db = df/dleft * dleft/db = right * 1 = [3, 5]
  f32 *b_grad = (f32 *)b->computation->grad->values;
  ASSERT(fabsf(b_grad[0] - 3.0f) < 1e-5, "b gradient[0] should be 3.0");
  ASSERT(fabsf(b_grad[1] - 5.0f) < 1e-5, "b gradient[1] should be 5.0");

  // df/dc = df/dright * dright/dc = left * 1 = [3, 4]
  f32 *c_grad = (f32 *)c->computation->grad->values;
  ASSERT(fabsf(c_grad[0] - 3.0f) < 1e-5, "c gradient[0] should be 3.0");
  ASSERT(fabsf(c_grad[1] - 4.0f) < 1e-5, "c gradient[1] should be 4.0");

  freeMemory(mem);
}

// Test full backward pass with Pow operation
static void test_backward_with_pow(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: y = (a + b)^2
  // Expected gradients:
  //   dy/da = 2 * (a + b)
  //   dy/db = 2 * (a + b)

  u32 dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values: a = [1, 2, 3], b = [1, 1, 1]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 1.0f; a_vals[1] = 2.0f; a_vals[2] = 3.0f;
  b_vals[0] = 1.0f; b_vals[1] = 1.0f; b_vals[2] = 1.0f;

  // temp = a + b = [2, 3, 4]
  Tensor temp;
  Result res = Add(&ctx, a, b, &temp);
  ASSERT_EQ(res, OK, "Add should succeed");

  // y = temp^2 = [4, 9, 16]
  Tensor y;
  res = Pow(&ctx, &temp, 2.0f, &y);
  ASSERT_EQ(res, OK, "Pow should succeed");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &y);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check gradients
  // dy/dtemp = 2 * temp = [4, 6, 8]
  // dy/da = dy/dtemp * dtemp/da = [4, 6, 8] * 1 = [4, 6, 8]
  // dy/db = dy/dtemp * dtemp/db = [4, 6, 8] * 1 = [4, 6, 8]

  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;

  ASSERT(fabsf(a_grad[0] - 4.0f) < 1e-5, "a gradient[0] should be 4.0");
  ASSERT(fabsf(a_grad[1] - 6.0f) < 1e-5, "a gradient[1] should be 6.0");
  ASSERT(fabsf(a_grad[2] - 8.0f) < 1e-5, "a gradient[2] should be 8.0");

  ASSERT(fabsf(b_grad[0] - 4.0f) < 1e-5, "b gradient[0] should be 4.0");
  ASSERT(fabsf(b_grad[1] - 6.0f) < 1e-5, "b gradient[1] should be 6.0");
  ASSERT(fabsf(b_grad[2] - 8.0f) < 1e-5, "b gradient[2] should be 8.0");

  freeMemory(mem);
}

// Test Pow in complex computation graph
static void test_backward_complex_with_pow(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Create computation: f = a^2 * b
  // Expected gradients:
  //   df/da = 2*a*b
  //   df/db = a^2

  u32 dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values: a = [2, 3], b = [5, 4]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 2.0f; a_vals[1] = 3.0f;
  b_vals[0] = 5.0f; b_vals[1] = 4.0f;

  // a_squared = a^2 = [4, 9]
  Tensor a_squared;
  Result res = Pow(&ctx, a, 2.0f, &a_squared);
  ASSERT_EQ(res, OK, "Pow should succeed");

  // f = a_squared * b = [20, 36]
  Tensor f;
  res = Multiply(&ctx, &a_squared, b, &f);
  ASSERT_EQ(res, OK, "Multiply should succeed");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &f);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check gradients
  // df/da_squared = b = [5, 4]
  // da_squared/da = 2*a = [4, 6]
  // df/da = df/da_squared * da_squared/da = [5*4, 4*6] = [20, 24]
  f32 *a_grad = (f32 *)a->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 20.0f) < 1e-4, "a gradient[0] should be 20.0");
  ASSERT(fabsf(a_grad[1] - 24.0f) < 1e-4, "a gradient[1] should be 24.0");

  // df/db = a_squared = [4, 9]
  f32 *b_grad = (f32 *)b->computation->grad->values;
  ASSERT(fabsf(b_grad[0] - 4.0f) < 1e-5, "b gradient[0] should be 4.0");
  ASSERT(fabsf(b_grad[1] - 9.0f) < 1e-5, "b gradient[1] should be 9.0");

  freeMemory(mem);
}

// Test that Divide (implemented as a * b^-1) has correct gradients
static void test_divide_backward_composed(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Test: c = a / b
  // Implemented as: c = a * (b^-1)
  // Expected gradients:
  //   dc/da = 1/b
  //   dc/db = -a/b^2

  u32 dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);

  // Set values: a = [6, 8, 10], b = [2, 4, 5]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  a_vals[0] = 6.0f; a_vals[1] = 8.0f; a_vals[2] = 10.0f;
  b_vals[0] = 2.0f; b_vals[1] = 4.0f; b_vals[2] = 5.0f;

  // Forward: c = a / b = [3, 2, 2]
  Tensor c;
  Result res = Divide(&ctx, a, b, &c);
  ASSERT_EQ(res, OK, "Divide should succeed");

  // Verify forward pass results
  f32 *c_vals = (f32 *)c.values;
  ASSERT(fabsf(c_vals[0] - 3.0f) < 1e-5, "6/2 should be 3");
  ASSERT(fabsf(c_vals[1] - 2.0f) < 1e-5, "8/4 should be 2");
  ASSERT(fabsf(c_vals[2] - 2.0f) < 1e-5, "10/5 should be 2");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &c);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check gradients
  // dc/da = 1/b = [1/2, 1/4, 1/5] = [0.5, 0.25, 0.2]
  f32 *a_grad = (f32 *)a->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 0.5f) < 1e-5, "a gradient[0] should be 1/2 = 0.5");
  ASSERT(fabsf(a_grad[1] - 0.25f) < 1e-5, "a gradient[1] should be 1/4 = 0.25");
  ASSERT(fabsf(a_grad[2] - 0.2f) < 1e-5, "a gradient[2] should be 1/5 = 0.2");

  // dc/db = -a/b^2 = [-6/4, -8/16, -10/25] = [-1.5, -0.5, -0.4]
  f32 *b_grad = (f32 *)b->computation->grad->values;
  ASSERT(fabsf(b_grad[0] - (-1.5f)) < 1e-4, "b gradient[0] should be -6/4 = -1.5");
  ASSERT(fabsf(b_grad[1] - (-0.5f)) < 1e-4, "b gradient[1] should be -8/16 = -0.5");
  ASSERT(fabsf(b_grad[2] - (-0.4f)) < 1e-4, "b gradient[2] should be -10/25 = -0.4");

  freeMemory(mem);
}

// Test division in a complex computation graph
static void test_divide_in_complex_graph(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  // Test: f = (a + b) / c
  // Expected gradients:
  //   df/da = 1/c
  //   df/db = 1/c
  //   df/dc = -(a+b)/c^2

  u32 dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};

  Tensor *a = t_Zeros(&ctx, shape, F32);
  Tensor *b = t_Zeros(&ctx, shape, F32);
  Tensor *c = t_Zeros(&ctx, shape, F32);

  // Set values: a = [3, 5], b = [1, 3], c = [2, 4]
  f32 *a_vals = (f32 *)a->values;
  f32 *b_vals = (f32 *)b->values;
  f32 *c_vals = (f32 *)c->values;
  a_vals[0] = 3.0f; a_vals[1] = 5.0f;
  b_vals[0] = 1.0f; b_vals[1] = 3.0f;
  c_vals[0] = 2.0f; c_vals[1] = 4.0f;

  // sum = a + b = [4, 8]
  Tensor sum;
  Result res = Add(&ctx, a, b, &sum);
  ASSERT_EQ(res, OK, "Add should succeed");

  // f = sum / c = [4/2, 8/4] = [2, 2]
  Tensor f;
  res = Divide(&ctx, &sum, c, &f);
  ASSERT_EQ(res, OK, "Divide should succeed");

  // Verify forward pass
  f32 *f_vals = (f32 *)f.values;
  ASSERT(fabsf(f_vals[0] - 2.0f) < 1e-5, "(3+1)/2 should be 2");
  ASSERT(fabsf(f_vals[1] - 2.0f) < 1e-5, "(5+3)/4 should be 2");

  // Build computation graph and run backward
  ComputationGraph *graph = InitComputationGraph(&ctx, &f);
  res = Backward(&ctx, graph);
  ASSERT_EQ(res, OK, "Backward should succeed");

  // Check gradients
  // df/da = df/db = 1/c = [1/2, 1/4] = [0.5, 0.25]
  f32 *a_grad = (f32 *)a->computation->grad->values;
  f32 *b_grad = (f32 *)b->computation->grad->values;
  ASSERT(fabsf(a_grad[0] - 0.5f) < 1e-5, "a gradient[0] should be 0.5");
  ASSERT(fabsf(a_grad[1] - 0.25f) < 1e-5, "a gradient[1] should be 0.25");
  ASSERT(fabsf(b_grad[0] - 0.5f) < 1e-5, "b gradient[0] should be 0.5");
  ASSERT(fabsf(b_grad[1] - 0.25f) < 1e-5, "b gradient[1] should be 0.25");

  // df/dc = -(a+b)/c^2 = [-4/4, -8/16] = [-1.0, -0.5]
  f32 *c_grad = (f32 *)c->computation->grad->values;
  ASSERT(fabsf(c_grad[0] - (-1.0f)) < 1e-4, "c gradient[0] should be -1.0");
  ASSERT(fabsf(c_grad[1] - (-0.5f)) < 1e-4, "c gradient[1] should be -0.5");

  freeMemory(mem);
}

void run_grad_tests(void) {
  // Individual backward operation tests
  test_add_backward_no_broadcast();
  test_add_backward_broadcast_size1();
  test_add_backward_broadcast_dim_mismatch();
  test_add_backward_broadcast_scalar();
  test_add_backward_gradient_accumulation();

  test_multiply_backward_no_broadcast();
  test_multiply_backward_broadcast_size1();
  test_multiply_backward_scalar();

  // Topological sort tests
  test_init_computation_graph_simple();
  test_init_computation_graph_multilevel();
  test_init_computation_graph_diamond();

  // Full backward pass tests
  test_backward_simple_add();
  test_backward_multilevel();
  test_backward_shared_input();
  test_backward_complex_graph();

  // Backward pass tests with Pow
  test_backward_with_pow();
  test_backward_complex_with_pow();

  // Backward pass tests with composed Divide (a * b^-1)
  test_divide_backward_composed();
  test_divide_in_complex_graph();
}
