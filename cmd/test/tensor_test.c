#include "test.h"
#include "../../tensor/tensor.h"

typedef struct {
  Tensor tensor;
  Memory *mem;
} TestTensor;

static TestTensor createZerosTensor(u32 *dims, u8 numOfDims) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = numOfDims});
  return (TestTensor){.tensor = t, .mem = mem};
}

static void test_zeros_creates_tensor_with_correct_shape(void) {
  u32 dims[] = {2, 3};
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 2, "tensor should have 2 dimensions");
  ASSERT_EQ(tt.tensor.dtype, U8, "T_Zeros should create U8 tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "tensor values should be allocated");

  freeMemory(tt.mem);
}

static void test_zeros_values_are_zero(void) {
  u32 dims[] = {4};
  TestTensor tt = createZerosTensor(dims, 1);

  u8 *values = (u8 *)tt.tensor.values;
  int all_zero = 1;
  for (u32 i = 0; i < 4; i++) {
    if (values[i] != 0) {
      all_zero = 0;
      break;
    }
  }
  ASSERT(all_zero, "all tensor values should be zero");

  freeMemory(tt.mem);
}

static void test_zeros_1d_tensor(void) {
  u32 dims[] = {5};
  TestTensor tt = createZerosTensor(dims, 1);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 1, "should be 1D tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "values should be allocated");

  freeMemory(tt.mem);
}

static void test_zeros_3d_tensor(void) {
  u32 dims[] = {2, 3, 4};
  TestTensor tt = createZerosTensor(dims, 3);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 3, "should be 3D tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "values should be allocated");

  u8 *values = (u8 *)tt.tensor.values;
  int all_zero = 1;
  for (u32 i = 0; i < 2 * 3 * 4; i++) {
    if (values[i] != 0) {
      all_zero = 0;
      break;
    }
  }
  ASSERT(all_zero, "all 3D tensor values should be zero");

  freeMemory(tt.mem);
}

// For shape [rows, cols], multipliers should be [cols, 1]
static void test_multipliers_2d_tensor(void) {
  u32 dims[] = {3, 4};  // 3 rows, 4 cols
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 4, "2D: multiplier[0] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 1, "2D: multiplier[1] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2], multipliers should be [d1*d2, d2, 1]
static void test_multipliers_3d_tensor(void) {
  u32 dims[] = {2, 3, 4};  // shape: 2x3x4
  TestTensor tt = createZerosTensor(dims, 3);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 12, "3D: multiplier[0] should be 12");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 4, "3D: multiplier[1] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[2], 1, "3D: multiplier[2] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2, d3], multipliers should be [d1*d2*d3, d2*d3, d3, 1]
static void test_multipliers_4d_tensor(void) {
  u32 dims[] = {2, 3, 4, 5};  // shape: 2x3x4x5
  TestTensor tt = createZerosTensor(dims, 4);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 60, "4D: multiplier[0] should be 60");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 20, "4D: multiplier[1] should be 20");
  ASSERT_EQ(tt.tensor.shape.multipliers[2], 5, "4D: multiplier[2] should be 5");
  ASSERT_EQ(tt.tensor.shape.multipliers[3], 1, "4D: multiplier[3] should be 1");

  freeMemory(tt.mem);
}

// AssignValue tests
static void test_assign_value_success(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {1, 2};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U8, .as.u8 = 42};

  Result r = AssignValue(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, OK, "AssignValue should return OK");

  u8 *values = (u8 *)tt.tensor.values;
  ASSERT_EQ(values[1 * 4 + 2], 42, "value at [1,2] should be 42");

  freeMemory(mem);
}

static void test_assign_value_dtype_mismatch(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U32, .as.u32 = 100};

  Result r = AssignValue(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "should return ERR_DTYPE_MISMATCH");

  freeMemory(mem);
}

static void test_assign_value_dim_mismatch(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {0, 0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 3};
  Value val = {.dtype = U8, .as.u8 = 10};

  Result r = AssignValue(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH");

  freeMemory(mem);
}

static void test_assign_value_out_of_bounds(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {3, 0};  // 3 >= 3, out of bounds
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U8, .as.u8 = 10};

  Result r = AssignValue(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "should return ERR_OUT_OF_BOUNDS");

  freeMemory(mem);
}

static void test_assign_value_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U8, .as.u8 = 10};

  Result r = AssignValue(&ctx, NULL, idx, val);
  ASSERT_EQ(r, ERR_NULL_PTR, "should return ERR_NULL_PTR for null tensor");

  freeMemory(mem);
}

static void test_assign_value_only_modifies_target_index(void) {
  u32 dims[] = {3, 4};  // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {1, 2};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U8, .as.u8 = 77};
  AssignValue(&ctx, &tt.tensor, idx, val);

  u8 *values = (u8 *)tt.tensor.values;
  int target_idx = 1 * 4 + 2;  // = 6

  int only_target_modified = 1;
  for (int i = 0; i < 12; i++) {
    if (i == target_idx) {
      if (values[i] != 77) {
        only_target_modified = 0;
        break;
      }
    } else {
      if (values[i] != 0) {
        only_target_modified = 0;
        break;
      }
    }
  }
  ASSERT(only_target_modified, "only index [1,2] should be modified");

  freeMemory(mem);
}

static void test_assign_value_multiple_indices(void) {
  u32 dims[] = {2, 3};  // 6 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx0[] = {0, 0};
  u32 idx1[] = {0, 2};
  u32 idx2[] = {1, 1};

  AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx0, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 10});
  AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx1, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 20});
  AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx2, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 30});

  u8 *values = (u8 *)tt.tensor.values;
  ASSERT_EQ(values[0 * 3 + 0], 10, "[0,0] should be 10");
  ASSERT_EQ(values[0 * 3 + 1], 0, "[0,1] should remain 0");
  ASSERT_EQ(values[0 * 3 + 2], 20, "[0,2] should be 20");
  ASSERT_EQ(values[1 * 3 + 0], 0, "[1,0] should remain 0");
  ASSERT_EQ(values[1 * 3 + 1], 30, "[1,1] should be 30");
  ASSERT_EQ(values[1 * 3 + 2], 0, "[1,2] should remain 0");

  freeMemory(mem);
}

// GetAt tests
static void test_get_at_success(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {1, 2};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U8, .as.u8 = 99};
  AssignValue(&ctx, &tt.tensor, idx, val);

  Value result;
  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, OK, "GetAt should return OK");
  ASSERT_EQ(result.dtype, U8, "result dtype should be U8");
  ASSERT_EQ(result.as.u8, 99, "result value should be 99");

  freeMemory(mem);
}

static void test_get_at_dim_mismatch(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  u32 idx_dims[] = {0};
  Dim idx = {.dims = idx_dims, .numOfDims = 1};
  Value result;

  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH");

  freeMemory(tt.mem);
}

static void test_get_at_out_of_bounds(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  u32 idx_dims[] = {0, 5};  // 5 >= 4, out of bounds
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value result;

  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "should return ERR_OUT_OF_BOUNDS");

  freeMemory(tt.mem);
}

static void test_get_at_null_tensor(void) {
  u32 idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value result;

  Result r = GetAt(NULL, idx, &result);
  ASSERT_EQ(r, ERR_NULL_PTR, "should return ERR_NULL_PTR for null tensor");
}

static void test_get_at_null_result(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  u32 idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};

  Result r = GetAt(&tt.tensor, idx, NULL);
  ASSERT_EQ(r, ERR_NULL_PTR, "should return ERR_NULL_PTR for null result");

  freeMemory(tt.mem);
}

void run_tensor_tests(void) {
  printf("=== Tensor Tests ===\n");
  test_zeros_creates_tensor_with_correct_shape();
  test_zeros_values_are_zero();
  test_zeros_1d_tensor();
  test_zeros_3d_tensor();
  test_multipliers_2d_tensor();
  test_multipliers_3d_tensor();
  test_multipliers_4d_tensor();
  test_assign_value_success();
  test_assign_value_dtype_mismatch();
  test_assign_value_dim_mismatch();
  test_assign_value_out_of_bounds();
  test_assign_value_null_tensor();
  test_assign_value_only_modifies_target_index();
  test_assign_value_multiple_indices();
  test_get_at_success();
  test_get_at_dim_mismatch();
  test_get_at_out_of_bounds();
  test_get_at_null_tensor();
  test_get_at_null_result();
}
