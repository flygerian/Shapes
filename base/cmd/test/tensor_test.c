#include "test.h"
#include "../../tensor/tensor.h"
#include <string.h>

typedef struct {
  Tensor tensor;
  Memory *mem;
} TestTensor;

static TestTensor createZerosTensor(u32 *dims, u8 numOfDims) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = numOfDims});
  return (TestTensor){.tensor = *t, .mem = mem};
}

static void test_zeros_creates_tensor_with_correct_shape(void) {
  u32 dims[] = {2, 3};
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 2, "tensor should have 2 dimensions");
  ASSERT_EQ(tt.tensor.dtype, F32, "T_Zeros should create F32 tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "tensor values should be allocated");

  freeMemory(tt.mem);
}

static void test_zeros_values_are_zero(void) {
  u32 dims[] = {4};
  TestTensor tt = createZerosTensor(dims, 1);

  f32 *values = (f32 *)tt.tensor.values;
  int all_zero = 1;
  for (u32 i = 0; i < 4; i++) {
    if (values[i] != 0.0f) {
      all_zero = 0;
      break;
    }
  }
  ASSERT(all_zero, "all tensor values should be zero");

  freeMemory(tt.mem);
}

static void test_int_creates_tensor_with_value(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 5);

  ASSERT_EQ(t->shape.numOfDims, 2, "tensor should have 2 dimensions");
  ASSERT_EQ(t->dtype, I8, "T_Int should create I8 tensor");

  i8 *values = (i8 *)t->values;
  int all_match = 1;
  for (u32 i = 0; i < 6; i++) {
    if (values[i] != 5) {
      all_match = 0;
      break;
    }
  }
  ASSERT(all_match, "all tensor values should be 5");

  freeMemory(mem);
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

  f32 *values = (f32 *)tt.tensor.values;
  int all_zero = 1;
  for (u32 i = 0; i < 2 * 3 * 4; i++) {
    if (values[i] != 0.0f) {
      all_zero = 0;
      break;
    }
  }
  ASSERT(all_zero, "all 3D tensor values should be zero");

  freeMemory(tt.mem);
}

// For shape [rows, cols], multipliers should be [cols, 1]
static void test_multipliers_2d_tensor(void) {
  u32 dims[] = {3, 4}; // 3 rows, 4 cols
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 4, "2D: multiplier[0] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 1, "2D: multiplier[1] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2], multipliers should be [d1*d2, d2, 1]
static void test_multipliers_3d_tensor(void) {
  u32 dims[] = {2, 3, 4}; // shape: 2x3x4
  TestTensor tt = createZerosTensor(dims, 3);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 12, "3D: multiplier[0] should be 12");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 4, "3D: multiplier[1] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[2], 1, "3D: multiplier[2] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2, d3], multipliers should be [d1*d2*d3, d2*d3, d3, 1]
static void test_multipliers_4d_tensor(void) {
  u32 dims[] = {2, 3, 4, 5}; // shape: 2x3x4x5
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
  Value val = {.dtype = F32, .as.f32 = 42.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, OK, "AssignValue should return OK");

  f32 *values = (f32 *)tt.tensor.values;
  ASSERT_EQ(values[1 * 4 + 2], 42.0f, "value at [1,2] should be 42");

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

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
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
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH");

  freeMemory(mem);
}

static void test_assign_value_out_of_bounds(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {3, 0}; // 3 >= 3, out of bounds
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "should return ERR_OUT_OF_BOUNDS");

  freeMemory(mem);
}

static void test_assign_value_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, NULL, idx, val);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED for null tensor");

  freeMemory(mem);
}

static void test_assign_value_only_modifies_target_index(void) {
  u32 dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx_dims[] = {1, 2};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 77.0f};
  AssignValueAt(&ctx, &tt.tensor, idx, val);

  f32 *values = (f32 *)tt.tensor.values;
  int target_idx = 1 * 4 + 2; // = 6

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
  u32 dims[] = {2, 3}; // 6 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 idx0[] = {0, 0};
  u32 idx1[] = {0, 2};
  u32 idx2[] = {1, 1};

  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx0, .numOfDims = 2},
                (Value){.dtype = F32, .as.f32 = 10.0f});
  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx1, .numOfDims = 2},
                (Value){.dtype = F32, .as.f32 = 20.0f});
  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx2, .numOfDims = 2},
                (Value){.dtype = F32, .as.f32 = 30.0f});

  f32 *values = (f32 *)tt.tensor.values;
  ASSERT_EQ(values[0 * 3 + 0], 10.0f, "[0,0] should be 10");
  ASSERT_EQ(values[0 * 3 + 1], 0.0f, "[0,1] should remain 0");
  ASSERT_EQ(values[0 * 3 + 2], 20.0f, "[0,2] should be 20");
  ASSERT_EQ(values[1 * 3 + 0], 0.0f, "[1,0] should remain 0");
  ASSERT_EQ(values[1 * 3 + 1], 30.0f, "[1,1] should be 30");
  ASSERT_EQ(values[1 * 3 + 2], 0.0f, "[1,2] should remain 0");

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
  Value val = {.dtype = F32, .as.f32 = 99.0f};
  AssignValueAt(&ctx, &tt.tensor, idx, val);

  Value result;
  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, OK, "GetAt should return OK");
  ASSERT_EQ(result.dtype, F32, "result dtype should be F32");
  ASSERT_EQ(result.as.f32, 99.0f, "result value should be 99");

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

  u32 idx_dims[] = {0, 5}; // 5 >= 4, out of bounds
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

// Slice tests
static void test_slice_basic_2d(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate tensor with values for testing
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 5 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 4});
  ASSERT_EQ(r, OK, "Slice should return OK");
  ASSERT(slice.isView, "slice should be a view");
  ASSERT_EQ(slice.shape.numOfDims, 2, "slice should have 2 dimensions");
  ASSERT_EQ(slice.shape.dims[0], 2, "slice dim[0] should be 2");
  ASSERT_EQ(slice.shape.dims[1], 3, "slice dim[1] should be 3");

  freeMemory(mem);
}

static void test_slice_shares_data_with_source(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Set a value in source
  u32 idx_dims[] = {1, 2};
  Value val = {.dtype = F32, .as.f32 = 42.0f};
  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);

  Tensor slice;
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 3}, (Range){.start = 0, .end = 4});
  ASSERT_EQ(r, OK, "Slice should return OK");
  ASSERT_EQ(slice.values, tt.tensor.values, "slice should share values pointer with source");

  freeMemory(mem);
}

static void test_slice_get_at_correct_values(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j] = i*5 + j
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 5 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  // Slice rows 1-3 (exclusive), cols 2-5 (exclusive) -> should get [1,2], [1,3], [1,4], [2,2],
  // [2,3], [2,4]
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 2, .end = 5});

  // Access slice[0,0] should be source[1,2] = 1*5+2 = 7
  u32 slice_idx[] = {0, 0};
  Value result;
  Result r = GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.f32, 7.0f, "slice[0,0] should be 7 (source[1,2])");

  // Access slice[1,2] should be source[2,4] = 2*5+4 = 14
  u32 slice_idx2[] = {1, 2};
  r = GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.f32, 14.0f, "slice[1,2] should be 14 (source[2,4])");

  freeMemory(mem);
}

static void test_slice_invalid_range_end_before_start(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 1}, (Range){.start = 0, .end = 4});
  ASSERT_EQ(r, ERR_INVALID_RANGE, "should return ERR_INVALID_RANGE when end < start");

  freeMemory(mem);
}

static void test_slice_range_out_of_bounds(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  // Range end exceeds dim size (5 > 4 for first dim)
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 5}, (Range){.start = 0, .end = 4});
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH when range exceeds bounds");

  freeMemory(mem);
}

static void test_slice_single_element_range(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Set value at [2,3]
  u32 idx_dims[] = {2, 3};
  Value val = {.dtype = F32, .as.f32 = 99.0f};
  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);

  Tensor slice;
  // Single element slice at [2,3] (exclusive end: 2:3 gives 1 element, 3:4 gives 1 element)
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 3}, (Range){.start = 3, .end = 4});
  ASSERT_EQ(r, OK, "single element slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 1, "slice dim[0] should be 1");
  ASSERT_EQ(slice.shape.dims[1], 1, "slice dim[1] should be 1");

  u32 slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 99.0f, "single element slice value should be 99");

  freeMemory(mem);
}

static void test_slice_full_range(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  // Full range slice (exclusive end: 0:3 gives 3 elements, 0:4 gives 4 elements)
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 3}, (Range){.start = 0, .end = 4});
  ASSERT_EQ(r, OK, "full range slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 3, "slice dim[0] should match source");
  ASSERT_EQ(slice.shape.dims[1], 4, "slice dim[1] should match source");

  freeMemory(mem);
}

static void test_slice_1d_tensor(void) {
  u32 dims[] = {10};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with values 0-9
  for (u32 i = 0; i < 10; i++) {
    u32 idx_dims[] = {i};
    Value val = {.dtype = F32, .as.f32 = (f32)i};
    AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 1}, val);
  }

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 3, .end = 8});
  ASSERT_EQ(r, OK, "1D slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 5, "1D slice should have 5 elements");

  // slice[0] should be source[3] = 3
  u32 slice_idx[] = {0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 3.0f, "slice[0] should be 3");

  // slice[4] should be source[7] = 7
  u32 slice_idx2[] = {4};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 7.0f, "slice[4] should be 7");

  freeMemory(mem);
}

static void test_slice_modify_reflects_in_source(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 4});

  // Modify slice[0,1] which maps to source[1,2]
  u32 slice_idx[] = {0, 1};
  Value val = {.dtype = F32, .as.f32 = 77.0f};
  AssignValueAt(&ctx, &slice, (Dim){.dims = slice_idx, .numOfDims = 2}, val);

  // Check source[1,2]
  u32 src_idx[] = {1, 2};
  Value result;
  GetAt(&tt.tensor, (Dim){.dims = src_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 77.0f, "modifying slice should reflect in source");

  freeMemory(mem);
}

static void test_slice_of_slice(void) {
  u32 dims[] = {6, 6};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j] = i*6 + j
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 6; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 6 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  // First slice: rows 1-5 (exclusive), cols 1-5 (exclusive) -> 4x4 region
  Tensor slice1;
  Slice(&ctx, &tt.tensor, &slice1, (Range){.start = 1, .end = 5}, (Range){.start = 1, .end = 5});

  // Second slice of first slice: rows 1-3 (exclusive), cols 1-3 (exclusive) -> 2x2 region
  // This maps to source rows 2-3, cols 2-3
  Tensor slice2;
  Result r =
      Slice(&ctx, &slice1, &slice2, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 3});
  ASSERT_EQ(r, OK, "slice of slice should return OK");
  ASSERT_EQ(slice2.shape.dims[0], 2, "nested slice dim[0] should be 2");
  ASSERT_EQ(slice2.shape.dims[1], 2, "nested slice dim[1] should be 2");

  // slice2[0,0] should be source[2,2] = 2*6+2 = 14
  u32 slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice2, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 14.0f, "nested slice[0,0] should be 14 (source[2,2])");

  freeMemory(mem);
}

static void test_slice_large_4d_tensor(void) {
  u32 dims[] = {8, 10, 12, 6}; // 8x10x12x6 = 5760 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j,k,l] = (i*10*12*6 + j*12*6 + k*6 + l) % 256
  for (u32 i = 0; i < 8; i++) {
    for (u32 j = 0; j < 10; j++) {
      for (u32 k = 0; k < 12; k++) {
        for (u32 l = 0; l < 6; l++) {
          u32 idx_dims[] = {i, j, k, l};
          u8 val_num = (u8)((i * 10 * 12 * 6 + j * 12 * 6 + k * 6 + l) % 256);
          Value val = {.dtype = F32, .as.f32 = (f32)val_num};
          AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 4}, val);
        }
      }
    }
  }

  Tensor slice;
  // Slice: [2:6, 3:8, 4:10, 1:5] (exclusive) -> 4x5x6x4 = 480 elements
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 6}, (Range){.start = 3, .end = 8},
            (Range){.start = 4, .end = 10}, (Range){.start = 1, .end = 5});

  ASSERT_EQ(r, OK, "4D slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 4, "4D slice dim[0] should be 4");
  ASSERT_EQ(slice.shape.dims[1], 5, "4D slice dim[1] should be 5");
  ASSERT_EQ(slice.shape.dims[2], 6, "4D slice dim[2] should be 6");
  ASSERT_EQ(slice.shape.dims[3], 4, "4D slice dim[3] should be 4");

  // Test slice[0,0,0,0] = source[2,3,4,1]
  u32 slice_idx[] = {0, 0, 0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 4}, &result);
  f32 expected = (f32)((2 * 10 * 12 * 6 + 3 * 12 * 6 + 4 * 6 + 1) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice[0,0,0,0] should match source[2,3,4,1]");

  // Test slice[3,4,5,3] = source[5,7,9,4]
  u32 slice_idx2[] = {3, 4, 5, 3};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 4}, &result);
  expected = (f32)((5 * 10 * 12 * 6 + 7 * 12 * 6 + 9 * 6 + 4) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice[3,4,5,3] should match source[5,7,9,4]");

  // Test middle element: slice[2,2,3,2] = source[4,5,7,3]
  u32 slice_idx3[] = {2, 2, 3, 2};
  GetAt(&slice, (Dim){.dims = slice_idx3, .numOfDims = 4}, &result);
  expected = (f32)((4 * 10 * 12 * 6 + 5 * 12 * 6 + 7 * 6 + 3) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice middle element should be correct");

  // Verify all elements in slice match expected source values
  int all_correct = 1;
  for (u32 i = 0; i < 4 && all_correct; i++) {
    for (u32 j = 0; j < 5 && all_correct; j++) {
      for (u32 k = 0; k < 6 && all_correct; k++) {
        for (u32 l = 0; l < 4 && all_correct; l++) {
          u32 s_idx[] = {i, j, k, l};
          GetAt(&slice, (Dim){.dims = s_idx, .numOfDims = 4}, &result);

          u32 src_i = i + 2, src_j = j + 3, src_k = k + 4, src_l = l + 1;
          f32 exp = (f32)((src_i * 10 * 12 * 6 + src_j * 12 * 6 + src_k * 6 + src_l) % 256);
          if (result.as.f32 != exp) {
            all_correct = 0;
          }
        }
      }
    }
  }
  ASSERT(all_correct, "all 480 elements in 4D slice should be correct");

  freeMemory(mem);
}

// Reshape tests
static void test_reshape_basic_2d_to_1d(void) {
  u32 dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 new_dims[] = {12};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 2D to 1D should return OK");
  ASSERT_EQ(reshaped.shape.numOfDims, 1, "reshaped should have 1 dimension");
  ASSERT_EQ(reshaped.shape.dims[0], 12, "reshaped dim[0] should be 12");

  freeMemory(mem);
}

static void test_reshape_1d_to_2d(void) {
  u32 dims[] = {24};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 new_dims[] = {4, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 1D to 2D should return OK");
  ASSERT_EQ(reshaped.shape.numOfDims, 2, "reshaped should have 2 dimensions");
  ASSERT_EQ(reshaped.shape.dims[0], 4, "reshaped dim[0] should be 4");
  ASSERT_EQ(reshaped.shape.dims[1], 6, "reshaped dim[1] should be 6");

  freeMemory(mem);
}

static void test_reshape_preserves_data(void) {
  u32 dims[] = {2, 3}; // 6 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with sequential values
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 3 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  u32 new_dims[] = {6};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Reshape(&ctx, &tt.tensor, &reshaped, newShape);

  // Verify all values preserved in row-major order
  for (u32 i = 0; i < 6; i++) {
    u32 idx[] = {i};
    Value result;
    GetAt(&reshaped, (Dim){.dims = idx, .numOfDims = 1}, &result);
    ASSERT_EQ(result.as.f32, i, "reshaped data should be preserved");
  }

  freeMemory(mem);
}

static void test_reshape_shares_data_with_source(void) {
  u32 dims[] = {4, 3};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 new_dims[] = {2, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Reshape(&ctx, &tt.tensor, &reshaped, newShape);

  ASSERT_EQ(reshaped.values, tt.tensor.values, "reshaped should share values pointer");

  // Modify via reshaped, check source
  u32 r_idx[] = {0, 0};
  Value val = {.dtype = F32, .as.f32 = 55.0f};
  AssignValueAt(&ctx, &reshaped, (Dim){.dims = r_idx, .numOfDims = 2}, val);

  u32 s_idx[] = {0, 0};
  Value result;
  GetAt(&tt.tensor, (Dim){.dims = s_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 55.0f, "modification via reshaped should reflect in source");

  freeMemory(mem);
}

static void test_reshape_invalid_size_mismatch(void) {
  u32 dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u32 new_dims[] = {10}; // 10 != 12
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, ERR_RESHAPE_DIM_MISMATCH,
            "should return ERR_RESHAPE_DIM_MISMATCH for size mismatch");

  freeMemory(mem);
}

static void test_reshape_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 new_dims[] = {6};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, NULL, &reshaped, newShape);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED for null tensor");

  freeMemory(mem);
}

static void test_reshape_null_shape(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Dim newShape = {.dims = NULL, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, ERR_NULL_SHAPE_PROVIDED,
            "should return ERR_NULL_SHAPE_PROVIDED for null shape dims");

  freeMemory(mem);
}

static void test_reshape_3d_to_2d(void) {
  u32 dims[] = {2, 3, 4}; // 24 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        u32 idx_dims[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)(i * 12 + j * 4 + k)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 3}, val);
      }
    }
  }

  u32 new_dims[] = {6, 4};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 3D to 2D should return OK");
  ASSERT_EQ(reshaped.shape.dims[0], 6, "reshaped dim[0] should be 6");
  ASSERT_EQ(reshaped.shape.dims[1], 4, "reshaped dim[1] should be 4");

  // Check reshaped[0,0] = 0, reshaped[5,3] = 23
  u32 idx1[] = {0, 0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0,0] should be 0");

  u32 idx2[] = {5, 3};
  GetAt(&reshaped, (Dim){.dims = idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 23.0f, "reshaped[5,3] should be 23");

  freeMemory(mem);
}

static void test_reshape_view(void) {
  u32 dims[] = {6, 6}; // 36 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 6; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 6 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  // Create a slice: rows 1-4 (exclusive), cols 0-6 (exclusive) -> 3x6 = 18 elements
  // Note: This slice is contiguous in memory
  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 4}, (Range){.start = 0, .end = 6});

  // Reshape the slice to 1D (18 elements)
  u32 new_dims[] = {18};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape of view should return OK");
  ASSERT(!reshaped.isView, "reshaped view should be copied to contiguous array");
  ASSERT_EQ(reshaped.shape.dims[0], 18, "reshaped should have 18 elements");

  freeMemory(mem);
}

static void test_reshape_3d_view(void) {
  u32 dims[] = {4, 5, 6}; // 120 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      for (u32 k = 0; k < 6; k++) {
        u32 idx_dims[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)((i * 30 + j * 6 + k) % 256)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 3}, val);
      }
    }
  }

  // Slice: [1:3, 0:5, 0:6] (exclusive) -> 2x5x6 = 60 elements
  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 0, .end = 5},
        (Range){.start = 0, .end = 6});

  ASSERT_EQ(slice.shape.dims[0], 2, "3D slice dim[0] should be 2");
  ASSERT_EQ(slice.shape.dims[1], 5, "3D slice dim[1] should be 5");
  ASSERT_EQ(slice.shape.dims[2], 6, "3D slice dim[2] should be 6");

  // Reshape slice to 2D: 10x6 = 60 elements
  u32 new_dims[] = {10, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 3D view to 2D should return OK");
  ASSERT(!reshaped.isView, "reshaped 3D view should be copied to contiguous array");
  ASSERT_EQ(reshaped.shape.dims[0], 10, "reshaped dim[0] should be 10");
  ASSERT_EQ(reshaped.shape.dims[1], 6, "reshaped dim[1] should be 6");

  // Verify reshaped[0,0] = slice[0,0,0] = source[1,0,0] = 1*30 = 30
  u32 r_idx[] = {0, 0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = r_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 30.0f, "reshaped[0,0] should be 30");

  freeMemory(mem);
}

static void test_reshape_4d_view(void) {
  u32 dims[] = {3, 4, 5, 6}; // 360 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with pattern
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      for (u32 k = 0; k < 5; k++) {
        for (u32 l = 0; l < 6; l++) {
          u32 idx_dims[] = {i, j, k, l};
          u8 val_num = (u8)((i * 120 + j * 30 + k * 6 + l) % 256);
          Value val = {.dtype = F32, .as.f32 = (f32)val_num};
          AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 4}, val);
        }
      }
    }
  }

  // Slice: [0:2, 1:4, 0:5, 0:6] (exclusive) -> 2x3x5x6 = 180 elements
  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 2}, (Range){.start = 1, .end = 4},
        (Range){.start = 0, .end = 5}, (Range){.start = 0, .end = 6});

  ASSERT_EQ(slice.shape.dims[0], 2, "4D slice dim[0] should be 2");
  ASSERT_EQ(slice.shape.dims[1], 3, "4D slice dim[1] should be 3");
  ASSERT_EQ(slice.shape.dims[2], 5, "4D slice dim[2] should be 5");
  ASSERT_EQ(slice.shape.dims[3], 6, "4D slice dim[3] should be 6");

  // Reshape to 3D: 6x5x6 = 180 elements
  u32 new_dims[] = {6, 5, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 3};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 4D view to 3D should return OK");
  ASSERT(!reshaped.isView, "reshaped 4D view should be copied to contiguous array");
  ASSERT_EQ(reshaped.shape.numOfDims, 3, "reshaped should have 3 dimensions");
  ASSERT_EQ(reshaped.shape.dims[0], 6, "reshaped dim[0] should be 6");
  ASSERT_EQ(reshaped.shape.dims[1], 5, "reshaped dim[1] should be 5");
  ASSERT_EQ(reshaped.shape.dims[2], 6, "reshaped dim[2] should be 6");

  freeMemory(mem);
}

static void test_reshape_4d_view_to_1d(void) {
  u32 dims[] = {2, 3, 4, 5}; // 120 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate sequentially
  u8 counter = 0;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        for (u32 l = 0; l < 5; l++) {
          u32 idx_dims[] = {i, j, k, l};
          Value val = {.dtype = F32, .as.f32 = (f32)(counter++)};
          AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 4}, val);
        }
      }
    }
  }

  // Slice: [0:1, 0:3, 0:4, 0:5] (exclusive) -> 1x3x4x5 = 60 elements
  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 1}, (Range){.start = 0, .end = 3},
        (Range){.start = 0, .end = 4}, (Range){.start = 0, .end = 5});

  // Reshape to 1D: 60 elements
  u32 new_dims[] = {60};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 4D view to 1D should return OK");
  ASSERT_EQ(reshaped.shape.numOfDims, 1, "reshaped should have 1 dimension");
  ASSERT_EQ(reshaped.shape.dims[0], 60, "reshaped should have 60 elements");

  // Check first element: reshaped[0] = slice[0,0,0,0] = source[0,0,0,0] = 0
  u32 idx1[] = {0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0] should be 0");

  freeMemory(mem);
}

static void test_reshape_then_access_elements(void) {
  u32 dims[] = {2, 2, 3}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with values 0-11
  u8 counter = 0;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 3; k++) {
        u32 idx_dims[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)(counter++)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 3}, val);
      }
    }
  }

  // Reshape to 4x3
  u32 new_dims[] = {4, 3};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Reshape(&ctx, &tt.tensor, &reshaped, newShape);

  // Verify all elements accessible with new indexing
  // reshaped[0,0] = 0, reshaped[0,1] = 1, reshaped[0,2] = 2
  // reshaped[1,0] = 3, reshaped[1,1] = 4, reshaped[1,2] = 5
  // etc.
  int all_correct = 1;
  for (u32 i = 0; i < 4 && all_correct; i++) {
    for (u32 j = 0; j < 3 && all_correct; j++) {
      u32 idx[] = {i, j};
      Value result;
      GetAt(&reshaped, (Dim){.dims = idx, .numOfDims = 2}, &result);
      f32 expected = (f32)(i * 3 + j);
      if (result.as.f32 != expected) {
        all_correct = 0;
      }
    }
  }
  ASSERT(all_correct, "all reshaped elements should be accessible with correct values");

  freeMemory(mem);
}

// Transpose tests
static void test_transpose_basic_2d(void) {
  u32 dims[] = {3, 4}; // 3 rows, 4 cols
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);
  ASSERT_EQ(r, OK, "Transpose should return OK");
  ASSERT(transposed.isView, "transposed should be a view");
  ASSERT(!transposed.isContigous, "transposed should not be contiguous");
  ASSERT_EQ(transposed.shape.dims[0], 4, "transposed dim[0] should be 4");
  ASSERT_EQ(transposed.shape.dims[1], 3, "transposed dim[1] should be 3");

  freeMemory(mem);
}

static void test_transpose_swaps_dims_and_multipliers(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  u8 orig_mult_0 = tt.tensor.shape.multipliers[0];
  u8 orig_mult_1 = tt.tensor.shape.multipliers[1];

  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  ASSERT_EQ(transposed.shape.multipliers[0], orig_mult_1, "multiplier[0] should be swapped");
  ASSERT_EQ(transposed.shape.multipliers[1], orig_mult_0, "multiplier[1] should be swapped");

  freeMemory(mem);
}

static void test_transpose_shares_data(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  ASSERT_EQ(transposed.values, tt.tensor.values, "transposed should share values pointer");

  freeMemory(mem);
}

static void test_transpose_access_elements(void) {
  u32 dims[] = {2, 3};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: source[i,j] = i*3 + j
  // source[0,0]=0, source[0,1]=1, source[0,2]=2
  // source[1,0]=3, source[1,1]=4, source[1,2]=5
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 3 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx, .numOfDims = 2}, val);
    }
  }

  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  // transposed[j,i] should equal source[i,j]
  // transposed[0,0] = source[0,0] = 0
  // transposed[0,1] = source[1,0] = 3
  // transposed[1,0] = source[0,1] = 1
  // transposed[2,1] = source[1,2] = 5
  Value result;

  u32 idx1[] = {0, 0};
  GetAt(&transposed, (Dim){.dims = idx1, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "transposed[0,0] should be 0");

  u32 idx2[] = {0, 1};
  GetAt(&transposed, (Dim){.dims = idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 3.0f, "transposed[0,1] should be 3");

  u32 idx3[] = {1, 0};
  GetAt(&transposed, (Dim){.dims = idx3, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "transposed[1,0] should be 1");

  u32 idx4[] = {2, 1};
  GetAt(&transposed, (Dim){.dims = idx4, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 5.0f, "transposed[2,1] should be 5");

  freeMemory(mem);
}

static void test_transpose_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, NULL, &transposed, (dim_t)0, (dim_t)1);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED");

  freeMemory(mem);
}

static void test_transpose_dim_out_of_bounds(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH for out of bounds dim");

  freeMemory(mem);
}

static void test_transpose_size_less_than_2(void) {
  u32 dims[] = {1};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)0);
  ASSERT_EQ(r, ERR_NO_OP, "should return ERR_NO_OP for tensor with size < 2");

  freeMemory(mem);
}

static void test_transpose_3d(void) {
  u32 dims[] = {2, 3, 4}; // 2x3x4
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: source[i,j,k] = i*12 + j*4 + k
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        u32 idx[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)(i * 12 + j * 4 + k)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx, .numOfDims = 3}, val);
      }
    }
  }

  // Transpose dims 0 and 2: shape becomes 4x3x2
  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)2);
  ASSERT_EQ(r, OK, "Transpose 3D should return OK");
  ASSERT_EQ(transposed.shape.dims[0], 4, "transposed dim[0] should be 4");
  ASSERT_EQ(transposed.shape.dims[1], 3, "transposed dim[1] should be 3");
  ASSERT_EQ(transposed.shape.dims[2], 2, "transposed dim[2] should be 2");

  // transposed[k,j,i] = source[i,j,k]
  // transposed[0,0,0] = source[0,0,0] = 0
  // transposed[3,2,1] = source[1,2,3] = 1*12 + 2*4 + 3 = 23
  Value result;

  u32 idx1[] = {0, 0, 0};
  GetAt(&transposed, (Dim){.dims = idx1, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "transposed[0,0,0] should be 0");

  u32 idx2[] = {3, 2, 1};
  GetAt(&transposed, (Dim){.dims = idx2, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 23.0f, "transposed[3,2,1] should be 23");

  freeMemory(mem);
}

static void test_reshape_after_transpose_copies(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 4 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx, .numOfDims = 2}, val);
    }
  }

  // Transpose: 3x4 -> 4x3
  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  // Reshape transposed to 1D: 12 elements
  u32 new_dims[] = {12};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &transposed, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape after transpose should return OK");
  ASSERT(!reshaped.isContigous || reshaped.values != transposed.values,
         "reshape should copy non-contiguous data");

  // Verify data is correctly copied in transposed order
  // Original source: 3x4, source[i,j] = i*4+j
  // Transposed: 4x3, transposed[j,i] = source[i,j]
  // Row-major iteration of transposed:
  //   transposed[0,0]=source[0,0]=0, transposed[0,1]=source[1,0]=4, transposed[0,2]=source[2,0]=8
  //   transposed[1,0]=source[0,1]=1, transposed[1,1]=source[1,1]=5, transposed[1,2]=source[2,1]=9
  //   ...
  // So reshaped = [0,4,8, 1,5,9, 2,6,10, 3,7,11]
  Value result;
  u32 idx0[] = {0};
  GetAt(&reshaped, (Dim){.dims = idx0, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0] should be 0");

  u32 idx1[] = {1};
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "reshaped[1] should be 4 (transposed[0,1])");

  u32 idx3[] = {3};
  GetAt(&reshaped, (Dim){.dims = idx3, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "reshaped[3] should be 1 (transposed[1,0])");

  freeMemory(mem);
}

// Add tests
static void test_add_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  // b = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value va = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      Value vb = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, va);
      AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, vb);
    }
  }

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add should return OK");

  // Verify: result = [[11,22,33], [44,55,66]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0,0] should be 11");
  ASSERT_EQ(values[1], 22.0f, "result[0,1] should be 22");
  ASSERT_EQ(values[5], 66.0f, "result[1,2] should be 66");

  freeMemory(mem);
}

static void test_add_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 2};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  b->dtype = U32; // Force dtype mismatch
  Tensor dest;

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "Add should return ERR_DTYPE_MISMATCH");

  freeMemory(mem);
}

static void test_add_broadcast_row_vector(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // a: [2, 3], b: [1, 3] -> broadcast b across rows
  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10, 20, 30]]
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {0, j};
    Value v = {.dtype = F32, .as.f32 = (f32)((j + 1) * 10)};
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, v);
  }

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add with broadcast should return OK");

  // result = [[11,22,33], [14,25,36]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0,0] should be 11");
  ASSERT_EQ(values[1], 22.0f, "result[0,1] should be 22");
  ASSERT_EQ(values[2], 33.0f, "result[0,2] should be 33");
  ASSERT_EQ(values[3], 14.0f, "result[1,0] should be 14");
  ASSERT_EQ(values[4], 25.0f, "result[1,1] should be 25");
  ASSERT_EQ(values[5], 36.0f, "result[1,2] should be 36");

  freeMemory(mem);
}

static void test_add_broadcast_col_vector(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // a: [2, 3], b: [2, 1] -> broadcast b across cols
  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {2, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10], [20]]
  for (u32 i = 0; i < 2; i++) {
    u32 idx[] = {i, 0};
    Value v = {.dtype = F32, .as.f32 = (f32)((i + 1) * 10)};
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, v);
  }

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add with col broadcast should return OK");

  // result = [[11,12,13], [24,25,26]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0,0] should be 11");
  ASSERT_EQ(values[1], 12.0f, "result[0,1] should be 12");
  ASSERT_EQ(values[2], 13.0f, "result[0,2] should be 13");
  ASSERT_EQ(values[3], 24.0f, "result[1,0] should be 24");
  ASSERT_EQ(values[4], 25.0f, "result[1,1] should be 25");
  ASSERT_EQ(values[5], 26.0f, "result[1,2] should be 26");

  freeMemory(mem);
}

static void test_add_broadcast_scalar(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // a: [2, 3], b: [1, 1] -> broadcast scalar b to all elements
  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {1, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[100]]
  u32 idx_b[] = {0, 0};
  Value vb = {.dtype = F32, .as.f32 = 100.0f};
  AssignValueAt(&ctx, b, (Dim){.dims = idx_b, .numOfDims = 2}, vb);

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add with scalar broadcast should return OK");

  // result = [[101,102,103], [104,105,106]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 101.0f, "result[0,0] should be 101");
  ASSERT_EQ(values[5], 106.0f, "result[1,2] should be 106");

  freeMemory(mem);
}

static void test_add_1d_tensors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {4};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});
  Tensor dest;

  // a = [1, 2, 3, 4], b = [10, 20, 30, 40]
  for (u32 i = 0; i < 4; i++) {
    u32 idx[] = {i};
    Value va = {.dtype = F32, .as.f32 = (f32)(i + 1)};
    Value vb = {.dtype = F32, .as.f32 = (f32)((i + 1) * 10)};
    AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 1}, va);
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 1}, vb);
  }

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add 1D should return OK");

  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0] should be 11");
  ASSERT_EQ(values[1], 22.0f, "result[1] should be 22");
  ASSERT_EQ(values[2], 33.0f, "result[2] should be 33");
  ASSERT_EQ(values[3], 44.0f, "result[3] should be 44");

  freeMemory(mem);
}

// Subtract tests
static void test_subtract_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  // b = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value va = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      Value vb = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, va);
      AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, vb);
    }
  }

  Result r = Subtract(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Subtract should return OK");

  // result = [[9,18,27], [36,45,54]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 9.0f, "result[0,0] should be 9");
  ASSERT_EQ(values[1], 18.0f, "result[0,1] should be 18");
  ASSERT_EQ(values[5], 54.0f, "result[1,2] should be 54");

  freeMemory(mem);
}

static void test_subtract_broadcast(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[1, 2, 3]]
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {0, j};
    Value v = {.dtype = F32, .as.f32 = (f32)(j + 1)};
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, v);
  }

  Result r = Subtract(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Subtract with broadcast should return OK");

  // result = [[9,18,27], [39,48,57]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 9.0f, "result[0,0] should be 9");
  ASSERT_EQ(values[3], 39.0f, "result[1,0] should be 39");

  freeMemory(mem);
}

// Multiply tests
static void test_multiply_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  // b = [[2,2,2], [3,3,3]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value va = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      Value vb = {.dtype = F32, .as.f32 = (f32)(i + 2)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, va);
      AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, vb);
    }
  }

  Result r = Multiply(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Multiply should return OK");

  // result = [[2,4,6], [12,15,18]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 2.0f, "result[0,0] should be 2");
  ASSERT_EQ(values[2], 6.0f, "result[0,2] should be 6");
  ASSERT_EQ(values[3], 12.0f, "result[1,0] should be 12");
  ASSERT_EQ(values[5], 18.0f, "result[1,2] should be 18");

  freeMemory(mem);
}

static void test_multiply_broadcast_scalar(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {1, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[5]]
  u32 idx_b[] = {0, 0};
  Value vb = {.dtype = F32, .as.f32 = 5.0f};
  AssignValueAt(&ctx, b, (Dim){.dims = idx_b, .numOfDims = 2}, vb);

  Result r = Multiply(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Multiply with scalar should return OK");

  // result = [[5,10,15], [20,25,30]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 5.0f, "result[0,0] should be 5");
  ASSERT_EQ(values[2], 15.0f, "result[0,2] should be 15");
  ASSERT_EQ(values[5], 30.0f, "result[1,2] should be 30");

  freeMemory(mem);
}

// Divide tests
static void test_divide_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  // b = [[2,4,5], [8,10,12]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value va = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      Value vb = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 2)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, va);
      AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, vb);
    }
  }

  Result r = Divide(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Divide should return OK");

  // result = [[5,5,5], [5,5,5]] (integer division)
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 5.0f, "result[0,0] should be 5");
  ASSERT_EQ(values[5], 5.0f, "result[1,2] should be 5");

  freeMemory(mem);
}

static void test_divide_broadcast(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[2, 5, 10]]
  u8 divisors[] = {2, 5, 10};
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {0, j};
    Value v = {.dtype = F32, .as.f32 = (f32)divisors[j]};
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, v);
  }

  Result r = Divide(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Divide with broadcast should return OK");

  // result = [[5,4,3], [20,10,6]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 5.0f, "result[0,0] should be 5");
  ASSERT_EQ(values[1], 4.0f, "result[0,1] should be 4");
  ASSERT_EQ(values[2], 3.0f, "result[0,2] should be 3");
  ASSERT_EQ(values[3], 20.0f, "result[1,0] should be 20");
  ASSERT_EQ(values[4], 10.0f, "result[1,1] should be 10");
  ASSERT_EQ(values[5], 6.0f, "result[1,2] should be 6");

  freeMemory(mem);
}

static void test_add_non_contiguous_transposed(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // a: [2, 3], b: [3, 2] transposed to [2, 3]
  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {3, 2};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10,20], [30,40], [50,60]] (3x2)
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 2; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)((i * 2 + j + 1) * 10)};
      AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Transpose b: [3, 2] -> [2, 3]
  // transposed b = [[10,30,50], [20,40,60]]
  Tensor bTransposed;
  Transpose(&ctx, b, &bTransposed, (dim_t)0, (dim_t)1);
  ASSERT(!bTransposed.isContigous, "transposed tensor should be non-contiguous");

  Result r = Add(&ctx, a, &bTransposed, &dest);
  ASSERT_EQ(r, OK, "Add with transposed tensor should return OK");

  // result = [[1+10, 2+30, 3+50], [4+20, 5+40, 6+60]]
  //        = [[11, 32, 53], [24, 45, 66]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0,0] should be 11");
  ASSERT_EQ(values[1], 32.0f, "result[0,1] should be 32");
  ASSERT_EQ(values[2], 53.0f, "result[0,2] should be 53");
  ASSERT_EQ(values[3], 24.0f, "result[1,0] should be 24");
  ASSERT_EQ(values[4], 45.0f, "result[1,1] should be 45");
  ASSERT_EQ(values[5], 66.0f, "result[1,2] should be 66");

  freeMemory(mem);
}

static void test_add_2d_plus_1d_broadcast(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // a: [2, 3], b: [3] -> broadcast 1D across rows
  u32 dims_a[] = {2, 3};
  u32 dims_b[] = {3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 1});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [10, 20, 30]
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {j};
    Value v = {.dtype = F32, .as.f32 = (f32)((j + 1) * 10)};
    AssignValueAt(&ctx, b, (Dim){.dims = idx, .numOfDims = 1}, v);
  }

  Result r = Add(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "Add 2D + 1D broadcast should return OK");

  // result = [[11,22,33], [14,25,36]]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 11.0f, "result[0,0] should be 11");
  ASSERT_EQ(values[1], 22.0f, "result[0,1] should be 22");
  ASSERT_EQ(values[2], 33.0f, "result[0,2] should be 33");
  ASSERT_EQ(values[3], 14.0f, "result[1,0] should be 14");
  ASSERT_EQ(values[4], 25.0f, "result[1,1] should be 25");
  ASSERT_EQ(values[5], 36.0f, "result[1,2] should be 36");

  freeMemory(mem);
}

// Sum tests
static void test_sum_dim0_2d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3 tensor: [[1,2,3], [4,5,6]]
  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "Sum dim 0 should return OK");

  // Sum along dim 0: [1+4, 2+5, 3+6] = [5, 7, 9], shape [1, 3]
  ASSERT_EQ(dest.shape.numOfDims, 2, "result should have 2 dims");
  ASSERT_EQ(dest.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(dest.shape.dims[1], 3, "dim 1 should be 3");

  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 5.0f, "result[0] should be 5");
  ASSERT_EQ(values[1], 7.0f, "result[1] should be 7");
  ASSERT_EQ(values[2], 9.0f, "result[2] should be 9");

  freeMemory(mem);
}

static void test_sum_dim1_2d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3 tensor: [[1,2,3], [4,5,6]]
  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "Sum dim 1 should return OK");

  // Sum along dim 1: [1+2+3, 4+5+6] = [6, 15], shape [2, 1]
  ASSERT_EQ(dest.shape.numOfDims, 2, "result should have 2 dims");
  ASSERT_EQ(dest.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "dim 1 should be 1");

  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 6.0f, "result[0] should be 6");
  ASSERT_EQ(values[1], 15.0f, "result[1] should be 15");

  freeMemory(mem);
}

static void test_sum_3d_middle_dim(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3x2 tensor
  u32 dims[] = {2, 3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  // Fill with sequential values 1-12
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 2; k++) {
        u32 idx[] = {i, j, k};
        Value v = {.dtype = F32, .as.f32 = (f32)(i * 6 + j * 2 + k + 1)};
        AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 3}, v);
      }
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "Sum 3D middle dim should return OK");

  // Shape should be [2, 1, 2]
  ASSERT_EQ(dest.shape.numOfDims, 3, "result should have 3 dims");
  ASSERT_EQ(dest.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "dim 1 should be 1");
  ASSERT_EQ(dest.shape.dims[2], 2, "dim 2 should be 2");

  // For batch 0: sum rows [1,2], [3,4], [5,6] along dim 1 = [1+3+5, 2+4+6] = [9, 12]
  // For batch 1: sum rows [7,8], [9,10], [11,12] along dim 1 = [7+9+11, 8+10+12] = [27, 30]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 9.0f, "result[0,0,0] should be 9");
  ASSERT_EQ(values[1], 12.0f, "result[0,0,1] should be 12");
  ASSERT_EQ(values[2], 27.0f, "result[1,0,0] should be 27");
  ASSERT_EQ(values[3], 30.0f, "result[1,0,1] should be 30");

  freeMemory(mem);
}

static void test_sum_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  Result r = Sum(&ctx, t, &dest, 2);
  ASSERT_EQ(r, ERR_SUM_DIM_OUT_OF_BOUNDS, "Sum with dim >= numOfDims should fail");

  freeMemory(mem);
}

static void test_sum_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor dest;

  Result r = Sum(&ctx, NULL, &dest, 0);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Sum with null tensor should fail");

  freeMemory(mem);
}

static void test_sum_non_contiguous(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create 3x2, transpose to 2x3, then sum
  u32 dims[] = {3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // t = [[1,2], [3,4], [5,6]]
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 2; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 2 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Transpose to 2x3: [[1,3,5], [2,4,6]]
  Tensor transposed;
  Transpose(&ctx, t, &transposed, (dim_t)0, (dim_t)1);
  ASSERT(!transposed.isContigous, "transposed should be non-contiguous");

  Tensor dest;
  Result r = Sum(&ctx, &transposed, &dest, 1);
  ASSERT_EQ(r, OK, "Sum on transposed tensor should return OK");

  // Sum along dim 1: [1+3+5, 2+4+6] = [9, 12], shape [2, 1]
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 9.0f, "result[0] should be 9");
  ASSERT_EQ(values[1], 12.0f, "result[1] should be 12");

  freeMemory(mem);
}

static void test_sum_1d_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {5};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  // t = [1, 2, 3, 4, 5]
  for (u32 i = 0; i < 5; i++) {
    u32 idx[] = {i};
    Value v = {.dtype = F32, .as.f32 = (f32)(i + 1)};
    AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 1}, v);
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "Sum 1D should return OK");

  // Sum = 15, shape [1]
  ASSERT_EQ(dest.shape.numOfDims, 1, "result should have 1 dim");
  ASSERT_EQ(dest.shape.dims[0], 1, "dim 0 should be 1");

  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 15.0f, "result should be 15");

  freeMemory(mem);
}

static void test_sum_4d_dim0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x2x2x3 tensor
  u32 dims[] = {2, 2, 2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 3; l++) {
          u32 idx[] = {i, j, k, l};
          Value v = {.dtype = F32, .as.f32 = (f32)(val++)};
          AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 4}, v);
        }
      }
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "Sum 4D dim 0 should return OK");

  // Shape should be [1, 2, 2, 3]
  ASSERT_EQ(dest.shape.numOfDims, 4, "result should have 4 dims");
  ASSERT_EQ(dest.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(dest.shape.dims[1], 2, "dim 1 should be 2");
  ASSERT_EQ(dest.shape.dims[2], 2, "dim 2 should be 2");
  ASSERT_EQ(dest.shape.dims[3], 3, "dim 3 should be 3");

  // First batch [0,:,:,:] has values 1-12, second [1,:,:,:] has 13-24
  // Sum along dim 0: element-wise 1+13=14, 2+14=16, ..., 12+24=36
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 14.0f, "result[0,0,0,0] should be 14");
  ASSERT_EQ(values[1], 16.0f, "result[0,0,0,1] should be 16");
  ASSERT_EQ(values[2], 18.0f, "result[0,0,0,2] should be 18");
  ASSERT_EQ(values[11], 36.0f, "result[0,1,1,2] should be 36");

  freeMemory(mem);
}

static void test_sum_4d_dim1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3x2x2 tensor
  u32 dims[] = {2, 3, 2, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 2; l++) {
          u32 idx[] = {i, j, k, l};
          Value v = {.dtype = F32, .as.f32 = (f32)(val++)};
          AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 4}, v);
        }
      }
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "Sum 4D dim 1 should return OK");

  // Shape should be [2, 1, 2, 2]
  ASSERT_EQ(dest.shape.numOfDims, 4, "result should have 4 dims");
  ASSERT_EQ(dest.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "dim 1 should be 1");
  ASSERT_EQ(dest.shape.dims[2], 2, "dim 2 should be 2");
  ASSERT_EQ(dest.shape.dims[3], 2, "dim 3 should be 2");

  // Batch 0: sum j=0,1,2 for each (k,l)
  // [0,0,0,0]: 1+5+9=15, [0,0,0,1]: 2+6+10=18, [0,0,1,0]: 3+7+11=21, [0,0,1,1]: 4+8+12=24
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 15.0f, "result[0,0,0,0] should be 15");
  ASSERT_EQ(values[1], 18.0f, "result[0,0,0,1] should be 18");
  ASSERT_EQ(values[2], 21.0f, "result[0,0,1,0] should be 21");
  ASSERT_EQ(values[3], 24.0f, "result[0,0,1,1] should be 24");

  freeMemory(mem);
}

static void test_sum_4d_dim3(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x2x2x4 tensor
  u32 dims[] = {2, 2, 2, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-32
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 4; l++) {
          u32 idx[] = {i, j, k, l};
          Value v = {.dtype = F32, .as.f32 = (f32)(val++)};
          AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 4}, v);
        }
      }
    }
  }

  Tensor dest;
  Result r = Sum(&ctx, t, &dest, 3);
  ASSERT_EQ(r, OK, "Sum 4D dim 3 should return OK");

  // Shape should be [2, 2, 2, 1]
  ASSERT_EQ(dest.shape.numOfDims, 4, "result should have 4 dims");
  ASSERT_EQ(dest.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 2, "dim 1 should be 2");
  ASSERT_EQ(dest.shape.dims[2], 2, "dim 2 should be 2");
  ASSERT_EQ(dest.shape.dims[3], 1, "dim 3 should be 1");

  // [0,0,0,:] = 1+2+3+4 = 10
  // [0,0,1,:] = 5+6+7+8 = 26
  // [0,1,0,:] = 9+10+11+12 = 42
  f32 *values = (f32 *)dest.values;
  ASSERT_EQ(values[0], 10.0f, "result[0,0,0,0] should be 10");
  ASSERT_EQ(values[1], 26.0f, "result[0,0,1,0] should be 26");
  ASSERT_EQ(values[2], 42.0f, "result[0,1,0,0] should be 42");

  freeMemory(mem);
}

static void test_sum_multiple_reduces_3d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3x4 tensor
  u32 dims[] = {2, 3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        u32 idx[] = {i, j, k};
        Value v = {.dtype = F32, .as.f32 = (f32)(val++)};
        AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 3}, v);
      }
    }
  }

  // First reduce dim 2: [2,3,4] -> [2,3,1]
  Tensor after_dim2;
  Result r = Sum(&ctx, t, &after_dim2, 2);
  ASSERT_EQ(r, OK, "First reduce should return OK");
  ASSERT_EQ(after_dim2.shape.dims[0], 2, "after dim2: dim 0 should be 2");
  ASSERT_EQ(after_dim2.shape.dims[1], 3, "after dim2: dim 1 should be 3");
  ASSERT_EQ(after_dim2.shape.dims[2], 1, "after dim2: dim 2 should be 1");

  // Then reduce dim 1: [2,3,1] -> [2,1,1]
  Tensor after_dim1;
  r = Sum(&ctx, &after_dim2, &after_dim1, 1);
  ASSERT_EQ(r, OK, "Second reduce should return OK");
  ASSERT_EQ(after_dim1.shape.dims[0], 2, "after dim1: dim 0 should be 2");
  ASSERT_EQ(after_dim1.shape.dims[1], 1, "after dim1: dim 1 should be 1");
  ASSERT_EQ(after_dim1.shape.dims[2], 1, "after dim1: dim 2 should be 1");

  // Finally reduce dim 0: [2,1,1] -> [1,1,1]
  Tensor after_dim0;
  r = Sum(&ctx, &after_dim1, &after_dim0, 0);
  ASSERT_EQ(r, OK, "Third reduce should return OK");
  ASSERT_EQ(after_dim0.shape.dims[0], 1, "after dim0: dim 0 should be 1");
  ASSERT_EQ(after_dim0.shape.dims[1], 1, "after dim0: dim 1 should be 1");
  ASSERT_EQ(after_dim0.shape.dims[2], 1, "after dim0: dim 2 should be 1");

  // Total sum of 1+2+...+24 = 300
  f32 *values = (f32 *)after_dim0.values;
  ASSERT_EQ(values[0], 300.0f, "final sum should be 300");

  freeMemory(mem);
}

static void test_sum_multiple_reduces_4d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x2x3x2 tensor (24 elements)
  u32 dims[] = {2, 2, 3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 3; k++) {
        for (u32 l = 0; l < 2; l++) {
          u32 idx[] = {i, j, k, l};
          Value v = {.dtype = F32, .as.f32 = (f32)(val++)};
          AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 4}, v);
        }
      }
    }
  }

  // Reduce dim 3: [2,2,3,2] -> [2,2,3,1]
  Tensor r1;
  Result res = Sum(&ctx, t, &r1, 3);
  ASSERT_EQ(res, OK, "Reduce dim 3 should return OK");

  // Reduce dim 0: [2,2,3,1] -> [1,2,3,1]
  Tensor r2;
  res = Sum(&ctx, &r1, &r2, 0);
  ASSERT_EQ(res, OK, "Reduce dim 0 should return OK");

  ASSERT_EQ(r2.shape.dims[0], 1, "r2: dim 0 should be 1");
  ASSERT_EQ(r2.shape.dims[1], 2, "r2: dim 1 should be 2");
  ASSERT_EQ(r2.shape.dims[2], 3, "r2: dim 2 should be 3");
  ASSERT_EQ(r2.shape.dims[3], 1, "r2: dim 3 should be 1");

  // Verify size
  ASSERT_EQ(r2.size, 6, "r2 size should be 6");

  freeMemory(mem);
}

static void test_sum_reduce_to_scalar_2d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 3x4 tensor
  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // Fill with 1-12
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 4 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Reduce dim 1: [3,4] -> [3,1]
  Tensor r1;
  Sum(&ctx, t, &r1, 1);

  // Reduce dim 0: [3,1] -> [1,1]
  Tensor r2;
  Sum(&ctx, &r1, &r2, 0);

  ASSERT_EQ(r2.shape.dims[0], 1, "final dim 0 should be 1");
  ASSERT_EQ(r2.shape.dims[1], 1, "final dim 1 should be 1");

  // Sum of 1-12 = 78
  f32 *values = (f32 *)r2.values;
  ASSERT_EQ(values[0], 78.0f, "total sum should be 78");

  freeMemory(mem);
}

// Squeeze tests
static void test_squeeze_removes_single_dims(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [1, 3, 1, 4] -> [3, 4]
  u32 dims[] = {1, 3, 1, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  Tensor squeezed;
  Result r = Squeeze(&ctx, t, &squeezed);
  ASSERT_EQ(r, OK, "Squeeze should return OK");

  ASSERT_EQ(squeezed.shape.numOfDims, 2, "squeezed should have 2 dims");
  ASSERT_EQ(squeezed.shape.dims[0], 3, "dim 0 should be 3");
  ASSERT_EQ(squeezed.shape.dims[1], 4, "dim 1 should be 4");
  ASSERT_EQ(squeezed.size, 12, "size should remain 12");

  freeMemory(mem);
}

static void test_squeeze_middle_dim(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [2, 1, 3] -> [2, 3]
  u32 dims[] = {2, 1, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  Tensor squeezed;
  Result r = Squeeze(&ctx, t, &squeezed);
  ASSERT_EQ(r, OK, "Squeeze should return OK");

  ASSERT_EQ(squeezed.shape.numOfDims, 2, "squeezed should have 2 dims");
  ASSERT_EQ(squeezed.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(squeezed.shape.dims[1], 3, "dim 1 should be 3");

  freeMemory(mem);
}

static void test_squeeze_no_single_dims(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [2, 3, 4] -> [2, 3, 4] (unchanged)
  u32 dims[] = {2, 3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  Tensor squeezed;
  Result r = Squeeze(&ctx, t, &squeezed);
  ASSERT_EQ(r, OK, "Squeeze should return OK");

  ASSERT_EQ(squeezed.shape.numOfDims, 3, "squeezed should have 3 dims");
  ASSERT_EQ(squeezed.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(squeezed.shape.dims[1], 3, "dim 1 should be 3");
  ASSERT_EQ(squeezed.shape.dims[2], 4, "dim 2 should be 4");

  freeMemory(mem);
}

static void test_squeeze_all_ones(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [1, 1, 1] -> [1]
  u32 dims[] = {1, 1, 1};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  Tensor squeezed;
  Result r = Squeeze(&ctx, t, &squeezed);
  ASSERT_EQ(r, OK, "Squeeze should return OK");

  ASSERT_EQ(squeezed.shape.numOfDims, 1, "squeezed should have 1 dim");
  ASSERT_EQ(squeezed.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(squeezed.size, 1, "size should be 1");

  freeMemory(mem);
}

static void test_squeeze_shares_data(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {1, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // Set a value
  u32 idx[] = {0, 1};
  Value v = {.dtype = F32, .as.f32 = 42.0f};
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);

  Tensor squeezed;
  Squeeze(&ctx, t, &squeezed);

  // Check value is accessible in squeezed tensor
  u32 sq_idx[] = {1};
  Value result;
  GetAt(&squeezed, (Dim){.dims = sq_idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 42.0f, "squeezed should share data with source");

  freeMemory(mem);
}

static void test_squeeze_dim_specific(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [1, 3, 1, 4] squeeze dim 0 -> [3, 1, 4]
  u32 dims[] = {1, 3, 1, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  Tensor squeezed;
  Result r = SqueezeDim(&ctx, t, &squeezed, 0);
  ASSERT_EQ(r, OK, "SqueezeDim should return OK");
  ASSERT_EQ(squeezed.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(squeezed.shape.dims[0], 3, "dim 0 should be 3");
  ASSERT_EQ(squeezed.shape.dims[1], 1, "dim 1 should be 1");
  ASSERT_EQ(squeezed.shape.dims[2], 4, "dim 2 should be 4");

  // [1, 3, 1, 4] squeeze dim 2 -> [1, 3, 4]
  Tensor squeezed2;
  r = SqueezeDim(&ctx, t, &squeezed2, 2);
  ASSERT_EQ(r, OK, "SqueezeDim dim 2 should return OK");
  ASSERT_EQ(squeezed2.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(squeezed2.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(squeezed2.shape.dims[1], 3, "dim 1 should be 3");
  ASSERT_EQ(squeezed2.shape.dims[2], 4, "dim 2 should be 4");

  // Squeezing a non-1 dim should fail
  Tensor squeezed3;
  r = SqueezeDim(&ctx, t, &squeezed3, 1);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "SqueezeDim non-1 dim should fail");

  // Out of bounds dim should fail
  Tensor squeezed4;
  r = SqueezeDim(&ctx, t, &squeezed4, 5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "SqueezeDim out of bounds should fail");

  freeMemory(mem);
}

static void test_squeeze_after_sum(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3 tensor
  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Sum dim 1: [2, 3] -> [2, 1]
  Tensor summed;
  Sum(&ctx, t, &summed, 1);

  // Squeeze: [2, 1] -> [2]
  Tensor squeezed;
  Squeeze(&ctx, &summed, &squeezed);

  ASSERT_EQ(squeezed.shape.numOfDims, 1, "squeezed should have 1 dim");
  ASSERT_EQ(squeezed.shape.dims[0], 2, "dim 0 should be 2");

  f32 *values = (f32 *)squeezed.values;
  ASSERT_EQ(values[0], 6.0f, "result[0] should be 6");
  ASSERT_EQ(values[1], 15.0f, "result[1] should be 15");

  freeMemory(mem);
}

// UnSqueeze tests
static void test_unsqueeze_dim0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [3, 4] -> [1, 3, 4]
  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  Tensor unsqueezed;
  Result r = UnSqueeze(&ctx, t, &unsqueezed, 0);
  ASSERT_EQ(r, OK, "UnSqueeze should return OK");

  ASSERT_EQ(unsqueezed.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(unsqueezed.shape.dims[1], 3, "dim 1 should be 3");
  ASSERT_EQ(unsqueezed.shape.dims[2], 4, "dim 2 should be 4");
  ASSERT_EQ(unsqueezed.size, 12, "size should remain 12");

  freeMemory(mem);
}

static void test_unsqueeze_middle(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [3, 4] -> [3, 1, 4]
  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  Tensor unsqueezed;
  Result r = UnSqueeze(&ctx, t, &unsqueezed, 1);
  ASSERT_EQ(r, OK, "UnSqueeze should return OK");

  ASSERT_EQ(unsqueezed.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 3, "dim 0 should be 3");
  ASSERT_EQ(unsqueezed.shape.dims[1], 1, "dim 1 should be 1");
  ASSERT_EQ(unsqueezed.shape.dims[2], 4, "dim 2 should be 4");

  freeMemory(mem);
}

static void test_unsqueeze_end(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [3, 4] -> [3, 4, 1]
  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  Tensor unsqueezed;
  Result r = UnSqueeze(&ctx, t, &unsqueezed, 2);
  ASSERT_EQ(r, OK, "UnSqueeze should return OK");

  ASSERT_EQ(unsqueezed.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 3, "dim 0 should be 3");
  ASSERT_EQ(unsqueezed.shape.dims[1], 4, "dim 1 should be 4");
  ASSERT_EQ(unsqueezed.shape.dims[2], 1, "dim 2 should be 1");

  freeMemory(mem);
}

static void test_unsqueeze_1d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [5] -> [1, 5]
  u32 dims[] = {5};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  Tensor unsqueezed;
  Result r = UnSqueeze(&ctx, t, &unsqueezed, 0);
  ASSERT_EQ(r, OK, "UnSqueeze should return OK");

  ASSERT_EQ(unsqueezed.shape.numOfDims, 2, "should have 2 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(unsqueezed.shape.dims[1], 5, "dim 1 should be 5");

  freeMemory(mem);
}

static void test_unsqueeze_shares_data(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  u32 idx[] = {1};
  Value v = {.dtype = F32, .as.f32 = 42.0f};
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 1}, v);

  Tensor unsqueezed;
  UnSqueeze(&ctx, t, &unsqueezed, 0);

  // Access via [0, 1]
  u32 new_idx[] = {0, 1};
  Value result;
  GetAt(&unsqueezed, (Dim){.dims = new_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 42.0f, "unsqueezed should share data");

  freeMemory(mem);
}

static void test_unsqueeze_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  Result r = UnSqueeze(&ctx, t, &dest, 3);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "UnSqueeze with dim > numOfDims should fail");

  freeMemory(mem);
}

static void test_unsqueeze_non_contiguous(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create 2x3, transpose to 3x2, then unsqueeze
  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Transpose to 3x2: [[1,4], [2,5], [3,6]]
  Tensor transposed;
  Transpose(&ctx, t, &transposed, (dim_t)0, (dim_t)1);
  ASSERT(!transposed.isContigous, "transposed should be non-contiguous");

  // Unsqueeze to [1, 3, 2]
  Tensor unsqueezed;
  Result r = UnSqueeze(&ctx, &transposed, &unsqueezed, 0);
  ASSERT_EQ(r, OK, "UnSqueeze non-contiguous should return OK");

  ASSERT_EQ(unsqueezed.shape.numOfDims, 3, "should have 3 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 1, "dim 0 should be 1");
  ASSERT_EQ(unsqueezed.shape.dims[1], 3, "dim 1 should be 3");
  ASSERT_EQ(unsqueezed.shape.dims[2], 2, "dim 2 should be 2");

  // Verify data access: [0, 0, 0] should be 1, [0, 0, 1] should be 4
  Value result;
  u32 idx1[] = {0, 0, 0};
  GetAt(&unsqueezed, (Dim){.dims = idx1, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "unsqueezed[0,0,0] should be 1");

  u32 idx2[] = {0, 0, 1};
  GetAt(&unsqueezed, (Dim){.dims = idx2, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "unsqueezed[0,0,1] should be 4");

  u32 idx3[] = {0, 1, 0};
  GetAt(&unsqueezed, (Dim){.dims = idx3, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 2.0f, "unsqueezed[0,1,0] should be 2");

  u32 idx4[] = {0, 2, 1};
  GetAt(&unsqueezed, (Dim){.dims = idx4, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 6.0f, "unsqueezed[0,2,1] should be 6");

  freeMemory(mem);
}

static void test_squeeze_unsqueeze_roundtrip(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [2, 3] -> squeeze (no change) -> unsqueeze dim 1 -> [2, 1, 3] -> squeeze -> [2, 3]
  u32 dims[] = {2, 1, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  for (u32 i = 0; i < 2; i++) {
    for (u32 k = 0; k < 3; k++) {
      u32 idx[] = {i, 0, k};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + k + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 3}, v);
    }
  }

  Tensor squeezed;
  Squeeze(&ctx, t, &squeezed);
  ASSERT_EQ(squeezed.shape.numOfDims, 2, "squeezed should have 2 dims");

  Tensor unsqueezed;
  UnSqueeze(&ctx, &squeezed, &unsqueezed, 1);
  ASSERT_EQ(unsqueezed.shape.numOfDims, 3, "unsqueezed should have 3 dims");
  ASSERT_EQ(unsqueezed.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(unsqueezed.shape.dims[1], 1, "dim 1 should be 1");
  ASSERT_EQ(unsqueezed.shape.dims[2], 3, "dim 2 should be 3");

  // Verify data
  u32 idx[] = {1, 0, 2};
  Value result;
  GetAt(&unsqueezed, (Dim){.dims = idx, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 6.0f, "data should be preserved");

  freeMemory(mem);
}

// Clone tests
static void test_clone_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  Tensor cloned;
  Result r = Clone(&ctx, t, &cloned);
  ASSERT_EQ(r, OK, "Clone should return OK");

  ASSERT_EQ(cloned.shape.numOfDims, 2, "cloned should have 2 dims");
  ASSERT_EQ(cloned.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(cloned.shape.dims[1], 3, "dim 1 should be 3");
  ASSERT_EQ(cloned.size, 6, "size should be 6");
  ASSERT(!cloned.isView, "clone should not be a view");

  f32 *values = (f32 *)cloned.values;
  ASSERT_EQ(values[0], 1.0f, "cloned[0] should be 1");
  ASSERT_EQ(values[5], 6.0f, "cloned[5] should be 6");

  freeMemory(mem);
}

static void test_clone_independent_data(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  u32 idx[] = {1};
  Value v = {.dtype = F32, .as.f32 = 10.0f};
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 1}, v);

  Tensor cloned;
  Clone(&ctx, t, &cloned);

  // Modify original
  v.as.u8 = 99;
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 1}, v);

  // Clone should be unchanged
  Value result;
  GetAt(&cloned, (Dim){.dims = idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 10.0f, "clone should be independent from source");

  freeMemory(mem);
}

static void test_clone_slice(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {4, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 4 + j)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // Slice [1:3, 1:3] -> 2x2 region
  Tensor slice;
  Slice(&ctx, t, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 3});

  Tensor cloned;
  Result r = Clone(&ctx, &slice, &cloned);
  ASSERT_EQ(r, OK, "Clone slice should return OK");

  ASSERT_EQ(cloned.shape.numOfDims, 2, "cloned should have 2 dims");
  ASSERT_EQ(cloned.shape.dims[0], 2, "dim 0 should be 2");
  ASSERT_EQ(cloned.shape.dims[1], 2, "dim 1 should be 2");
  ASSERT(!cloned.isView, "clone should not be a view");
  ASSERT(cloned.isContigous, "clone should be contiguous");

  // slice[0,0] = source[1,1] = 5
  // slice[0,1] = source[1,2] = 6
  // slice[1,0] = source[2,1] = 9
  // slice[1,1] = source[2,2] = 10
  f32 *values = (f32 *)cloned.values;
  ASSERT_EQ(values[0], 5.0f, "cloned[0,0] should be 5");
  ASSERT_EQ(values[1], 6.0f, "cloned[0,1] should be 6");
  ASSERT_EQ(values[2], 9.0f, "cloned[1,0] should be 9");
  ASSERT_EQ(values[3], 10.0f, "cloned[1,1] should be 10");

  freeMemory(mem);
}

static void test_clone_transposed(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  Tensor transposed;
  Transpose(&ctx, t, &transposed, (dim_t)0, (dim_t)1);

  Tensor cloned;
  Result r = Clone(&ctx, &transposed, &cloned);
  ASSERT_EQ(r, OK, "Clone transposed should return OK");

  ASSERT_EQ(cloned.shape.dims[0], 3, "dim 0 should be 3");
  ASSERT_EQ(cloned.shape.dims[1], 2, "dim 1 should be 2");
  ASSERT(cloned.isContigous, "clone should be contiguous");

  // Transposed: [[1,4], [2,5], [3,6]]
  f32 *values = (f32 *)cloned.values;
  ASSERT_EQ(values[0], 1.0f, "cloned[0,0] should be 1");
  ASSERT_EQ(values[1], 4.0f, "cloned[0,1] should be 4");
  ASSERT_EQ(values[2], 2.0f, "cloned[1,0] should be 2");
  ASSERT_EQ(values[3], 5.0f, "cloned[1,1] should be 5");

  freeMemory(mem);
}

static void test_clone_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor dest;

  Result r = Clone(&ctx, NULL, &dest);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Clone null should fail");

  freeMemory(mem);
}

static void test_slice_boundary_access(void) {
  u32 dims[] = {5, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 5; i++) {
    for (u32 j = 0; j < 5; j++) {
      u32 idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 5 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  // Slice rows 2-5 (exclusive), cols 1-4 (exclusive) -> 3x3 region
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 5}, (Range){.start = 1, .end = 4});

  // Test all 4 corners of the slice
  Value result;

  // Top-left: slice[0,0] = source[2,1] = 11
  u32 tl[] = {0, 0};
  GetAt(&slice, (Dim){.dims = tl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 11.0f, "top-left corner should be 11");

  // Top-right: slice[0,2] = source[2,3] = 13
  u32 tr[] = {0, 2};
  GetAt(&slice, (Dim){.dims = tr, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 13.0f, "top-right corner should be 13");

  // Bottom-left: slice[2,0] = source[4,1] = 21
  u32 bl[] = {2, 0};
  GetAt(&slice, (Dim){.dims = bl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 21.0f, "bottom-left corner should be 21");

  // Bottom-right: slice[2,2] = source[4,3] = 23
  u32 br[] = {2, 2};
  GetAt(&slice, (Dim){.dims = br, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 23.0f, "bottom-right corner should be 23");

  freeMemory(mem);
}

static Tensor createF32Tensor(Context *ctx, dim_t *dims, u8 numOfDims, float *values,
                              tensor_size_t size) {
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * numOfDims);
  tensor_size_t mult = 1;
  for (int i = numOfDims - 1; i >= 0; i--) {
    multipliers[i] = mult;
    mult *= dims[i];
  }

  float *vals = allocate(ctx->memory, sizeof(float) * size);
  memcpy(vals, values, sizeof(float) * size);

  return (Tensor){.dtype = F32,
                  .size = size,
                  .isContigous = true,
                  .isView = false,
                  .boundary = NULL,
                  .values = vals,
                  .shape = (Dim){.dims = dims, .numOfDims = numOfDims, .multipliers = multipliers}};
}

static void test_matmul_2d_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 3};
  float valsA[] = {1, 2, 3, 4, 5, 6};
  Tensor a = createF32Tensor(&ctx, dimsA, 2, valsA, 6);

  dim_t dimsB[] = {3, 2};
  float valsB[] = {1, 2, 3, 4, 5, 6};
  Tensor b = createF32Tensor(&ctx, dimsB, 2, valsB, 6);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "MatMul 2D basic should return OK");

  ASSERT_EQ(result.shape.numOfDims, 2, "result should be 2D");
  ASSERT_EQ(result.shape.dims[0], 2, "result rows should be 2");
  ASSERT_EQ(result.shape.dims[1], 2, "result cols should be 2");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 22, "[0,0] should be 22");
  ASSERT_EQ((int)vals[1], 28, "[0,1] should be 28");
  ASSERT_EQ((int)vals[2], 49, "[1,0] should be 49");
  ASSERT_EQ((int)vals[3], 64, "[1,1] should be 64");

  freeMemory(mem);
}

static void test_matmul_2d_non_square(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 4};
  float valsA[] = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor a = createF32Tensor(&ctx, dimsA, 2, valsA, 8);

  dim_t dimsB[] = {4, 3};
  float valsB[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor b = createF32Tensor(&ctx, dimsB, 2, valsB, 12);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "MatMul 2D non-square should return OK");

  ASSERT_EQ(result.shape.dims[0], 2, "result rows should be 2");
  ASSERT_EQ(result.shape.dims[1], 3, "result cols should be 3");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 70, "[0,0] should be 70");
  ASSERT_EQ((int)vals[1], 80, "[0,1] should be 80");
  ASSERT_EQ((int)vals[2], 90, "[0,2] should be 90");

  freeMemory(mem);
}

static void test_matmul_3d_batch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 2, 3};
  float valsA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor a = createF32Tensor(&ctx, dimsA, 3, valsA, 12);

  dim_t dimsB[] = {2, 3, 2};
  float valsB[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor b = createF32Tensor(&ctx, dimsB, 3, valsB, 12);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "MatMul 3D batch should return OK");

  ASSERT_EQ(result.shape.numOfDims, 3, "result should be 3D");
  ASSERT_EQ(result.shape.dims[0], 2, "batch size should be 2");
  ASSERT_EQ(result.shape.dims[1], 2, "result rows should be 2");
  ASSERT_EQ(result.shape.dims[2], 2, "result cols should be 2");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 22, "batch0[0,0] should be 22");
  ASSERT_EQ((int)vals[1], 28, "batch0[0,1] should be 28");

  freeMemory(mem);
}

static void test_matmul_broadcast_batch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 2, 3};
  float valsA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor a = createF32Tensor(&ctx, dimsA, 3, valsA, 12);

  dim_t dimsB[] = {1, 3, 2};
  float valsB[] = {1, 2, 3, 4, 5, 6};
  Tensor b = createF32Tensor(&ctx, dimsB, 3, valsB, 6);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "MatMul broadcast batch should return OK");

  ASSERT_EQ(result.shape.dims[0], 2, "output batch should be 2");
  ASSERT_EQ(result.shape.dims[1], 2, "result rows should be 2");
  ASSERT_EQ(result.shape.dims[2], 2, "result cols should be 2");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 22, "batch0[0,0] should be 22");
  ASSERT_EQ((int)vals[4], 76, "batch1[0,0] should be 76");

  freeMemory(mem);
}

static void test_matmul_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 2};
  float valsA[] = {1, 2, 3, 4};
  Tensor a = createF32Tensor(&ctx, dimsA, 2, valsA, 4);

  Tensor *b = T_Int(&ctx, (Dim){.dims = dimsA, .numOfDims = 2}, 0);

  Tensor result;
  Result r = MatMul(&ctx, &a, b, &result);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "MatMul with mismatched dtypes should fail");

  freeMemory(mem);
}

static void test_matmul_inner_dim_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 3};
  float valsA[] = {1, 2, 3, 4, 5, 6};
  Tensor a = createF32Tensor(&ctx, dimsA, 2, valsA, 6);

  dim_t dimsB[] = {2, 2};
  float valsB[] = {1, 2, 3, 4};
  Tensor b = createF32Tensor(&ctx, dimsB, 2, valsB, 4);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, ERR_MATMUL_INNER_DIM_MISMATCH, "MatMul with inner dim mismatch should fail");

  freeMemory(mem);
}

static void test_matmul_1d_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {3};
  float valsA[] = {1, 2, 3};
  Tensor a = createF32Tensor(&ctx, dimsA, 1, valsA, 3);

  dim_t dimsB[] = {3, 2};
  float valsB[] = {1, 2, 3, 4, 5, 6};
  Tensor b = createF32Tensor(&ctx, dimsB, 2, valsB, 6);

  Tensor result;
  Result r = MatMul(&ctx, &a, &b, &result);
  ASSERT_EQ(r, ERR_MATMUL_MIN_2D, "MatMul with 1D tensor should fail");

  freeMemory(mem);
}

static void test_matmul_integer_dtype_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  u32 dims[] = {2, 2};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  Tensor result;
  Result r = MatMul(&ctx, a, b, &result);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "MatMul with integer dtype should fail");

  freeMemory(mem);
}

static void test_dot_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {3};
  float valsA[] = {1, 2, 3};
  Tensor a = createF32Tensor(&ctx, dimsA, 1, valsA, 3);

  dim_t dimsB[] = {3};
  float valsB[] = {4, 5, 6};
  Tensor b = createF32Tensor(&ctx, dimsB, 1, valsB, 3);

  Tensor result;
  Result r = Dot(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "Dot basic should return OK");

  ASSERT_EQ(result.shape.numOfDims, 1, "result should be 1D");
  ASSERT_EQ(result.shape.dims[0], 1, "result size should be 1");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 32, "1*4 + 2*5 + 3*6 = 32");

  freeMemory(mem);
}

static void test_dot_larger_vectors(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {5};
  float valsA[] = {1, 2, 3, 4, 5};
  Tensor a = createF32Tensor(&ctx, dimsA, 1, valsA, 5);

  dim_t dimsB[] = {5};
  float valsB[] = {1, 1, 1, 1, 1};
  Tensor b = createF32Tensor(&ctx, dimsB, 1, valsB, 5);

  Tensor result;
  Result r = Dot(&ctx, &a, &b, &result);
  ASSERT_EQ(r, OK, "Dot larger vectors should return OK");

  float *vals = (float *)result.values;
  ASSERT_EQ((int)vals[0], 15, "1+2+3+4+5 = 15");

  freeMemory(mem);
}

static void test_dot_size_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {3};
  float valsA[] = {1, 2, 3};
  Tensor a = createF32Tensor(&ctx, dimsA, 1, valsA, 3);

  dim_t dimsB[] = {4};
  float valsB[] = {1, 2, 3, 4};
  Tensor b = createF32Tensor(&ctx, dimsB, 1, valsB, 4);

  Tensor result;
  Result r = Dot(&ctx, &a, &b, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Dot with size mismatch should fail");

  freeMemory(mem);
}

static void test_dot_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {3};
  float valsA[] = {1, 2, 3};
  Tensor a = createF32Tensor(&ctx, dimsA, 1, valsA, 3);

  Tensor *b = T_Int(&ctx, (Dim){.dims = dimsA, .numOfDims = 1}, 0);

  Tensor result;
  Result r = Dot(&ctx, &a, b, &result);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "Dot with dtype mismatch should fail");

  freeMemory(mem);
}

static void test_dot_2d_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 3};
  float valsA[] = {1, 2, 3, 4, 5, 6};
  Tensor a = createF32Tensor(&ctx, dimsA, 2, valsA, 6);

  dim_t dimsB[] = {3};
  float valsB[] = {1, 2, 3};
  Tensor b = createF32Tensor(&ctx, dimsB, 1, valsB, 3);

  Tensor result;
  Result r = Dot(&ctx, &a, &b, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Dot with 2D tensor should fail");

  freeMemory(mem);
}

static void test_dot_integer_dtype_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);

  Tensor result;
  Result r = Dot(&ctx, a, b, &result);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "Dot with integer dtype should fail");

  freeMemory(mem);
}

// Gradient initialization tests
static void test_grad_t_zeros_without_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = false};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  ASSERT_NULL(t->computation, "computation should be NULL when grad is false");

  freeMemory(mem);
}

static void test_grad_t_zeros_with_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  ASSERT_NOT_NULL(t->computation, "computation should not be NULL when grad is true");
  ASSERT_NOT_NULL(t->computation->grad, "gradient tensor should be allocated");
  ASSERT_EQ(t->computation->output, t, "output should point to the tensor");
  ASSERT_NULL(t->computation->inputs, "inputs should be NULL for leaf tensor");
  ASSERT_EQ(t->computation->numInputs, 0, "numInputs should be 0 for leaf tensor");
  ASSERT_NOT_NULL(t->computation->backward,
                  "backward should be emptyBackward (not NULL) for leaf tensor");

  freeMemory(mem);
}

static void test_grad_t_int_with_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {3, 4};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 5);

  ASSERT_NOT_NULL(t->computation, "computation should not be NULL when grad is true");
  ASSERT_NOT_NULL(t->computation->grad, "gradient tensor should be allocated");
  ASSERT_EQ(t->computation->grad->dtype, I8, "gradient dtype should match tensor dtype");

  freeMemory(mem);
}

static void test_grad_t_float_with_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 3.14f);

  ASSERT_NOT_NULL(t->computation, "computation should not be NULL when grad is true");
  ASSERT_NOT_NULL(t->computation->grad, "gradient tensor should be allocated");
  ASSERT_EQ(t->computation->grad->dtype, F32, "gradient dtype should match tensor dtype");

  freeMemory(mem);
}

static void test_grad_clone_with_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  Tensor cloned;
  Result r = Clone(&ctx, t, &cloned);

  ASSERT_EQ(r, OK, "Clone should succeed");
  ASSERT_NOT_NULL(cloned.computation, "cloned tensor should have computation node");
  ASSERT_NOT_NULL(cloned.computation->grad, "cloned tensor should have gradient");
  ASSERT_EQ(cloned.computation->output, &cloned, "output should point to cloned tensor");

  freeMemory(mem);
}

static void test_grad_tensor_shape_matches(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {3, 4, 5};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  Tensor *grad = t->computation->grad;
  ASSERT_EQ(grad->shape.numOfDims, 3, "gradient should have same number of dims");
  ASSERT_EQ(grad->shape.dims[0], 3, "gradient dim 0 should match");
  ASSERT_EQ(grad->shape.dims[1], 4, "gradient dim 1 should match");
  ASSERT_EQ(grad->shape.dims[2], 5, "gradient dim 2 should match");
  ASSERT_EQ(grad->size, t->size, "gradient size should match tensor size");
  ASSERT_EQ(grad->dtype, t->dtype, "gradient dtype should match tensor dtype");

  freeMemory(mem);
}

static void test_grad_values_initialized_to_zero(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  Tensor *grad = t->computation->grad;
  f32 *gradValues = (f32 *)grad->values;

  int all_zero = 1;
  for (u32 i = 0; i < 6; i++) {
    if (gradValues[i] != 0.0f) {
      all_zero = 0;
      break;
    }
  }
  ASSERT(all_zero, "all gradient values should be initialized to zero");

  freeMemory(mem);
}

static void test_negate_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  u32 dims[] = {3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 5.0);
  Tensor dest;
  Result r = Negate(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Negate should succeed");
  f32 *vals = (f32 *)dest.values;
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(vals[i], -5.0f, "negated value should be -5.0");
  }
  freeMemory(mem);
}

static void test_negate_already_negative(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  u32 dims[] = {2};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 1}, -3.0);
  Tensor dest;
  Result r = Negate(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Negate should succeed for negative values");
  f32 *vals = (f32 *)dest.values;
  for (int i = 0; i < 2; i++) {
    ASSERT_EQ(vals[i], 3.0f, "negated -3.0 should be 3.0");
  }
  freeMemory(mem);
}

static void test_negate_unsigned_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  u32 dims[] = {2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 1);
  t->dtype = U32;
  Result r = Negate(&ctx, t, &(Tensor){});
  ASSERT_EQ(r, ERR_NEGATE_UNSUPPORTED_DTYPE, "Negate should reject unsigned dtypes");
  freeMemory(mem);
}

// Arange tests
static void test_arange_basic_positive_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *t = T_Arange(&ctx, 0.0f, 5.0f, 1.0f);
  ASSERT_NOT_NULL(t, "Arange should create tensor");
  ASSERT_EQ(t->shape.numOfDims, 1, "Arange should create 1D tensor");
  ASSERT_EQ(t->shape.dims[0], 5, "Arange should have 5 elements");
  ASSERT_EQ(t->dtype, F32, "Arange should create F32 tensor");

  f32 *values = (f32 *)t->values;
  ASSERT_EQ(values[0], 0.0f, "arange[0] should be 0");
  ASSERT_EQ(values[1], 1.0f, "arange[1] should be 1");
  ASSERT_EQ(values[2], 2.0f, "arange[2] should be 2");
  ASSERT_EQ(values[3], 3.0f, "arange[3] should be 3");
  ASSERT_EQ(values[4], 4.0f, "arange[4] should be 4");

  freeMemory(mem);
}

static void test_arange_negative_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *t = T_Arange(&ctx, 10.0f, 0.0f, -2.0f);
  ASSERT_NOT_NULL(t, "Arange with negative step should create tensor");
  ASSERT_EQ(t->shape.dims[0], 5, "Arange should have 5 elements");

  f32 *values = (f32 *)t->values;
  ASSERT_EQ(values[0], 10.0f, "arange[0] should be 10");
  ASSERT_EQ(values[1], 8.0f, "arange[1] should be 8");
  ASSERT_EQ(values[2], 6.0f, "arange[2] should be 6");
  ASSERT_EQ(values[3], 4.0f, "arange[3] should be 4");
  ASSERT_EQ(values[4], 2.0f, "arange[4] should be 2");

  freeMemory(mem);
}

static void test_arange_non_integer_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor *t = T_Arange(&ctx, 1.0f, 5.0f, 0.5f);
  ASSERT_NOT_NULL(t, "Arange with non-integer step should create tensor");
  ASSERT_EQ(t->shape.dims[0], 8, "Arange should have 8 elements");

  f32 *values = (f32 *)t->values;
  ASSERT_EQ(values[0], 1.0f, "arange[0] should be 1.0");
  ASSERT_EQ(values[1], 1.5f, "arange[1] should be 1.5");
  ASSERT_EQ(values[2], 2.0f, "arange[2] should be 2.0");
  ASSERT_EQ(values[7], 4.5f, "arange[7] should be 4.5");

  freeMemory(mem);
}

static void test_arange_default_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Step of 0 should default to 1
  Tensor *t = T_Arange(&ctx, 0.0f, 3.0f, 0.0f);
  ASSERT_NOT_NULL(t, "Arange with step=0 should default to 1");
  ASSERT_EQ(t->shape.dims[0], 3, "Arange should have 3 elements");

  f32 *values = (f32 *)t->values;
  ASSERT_EQ(values[0], 0.0f, "arange[0] should be 0");
  ASSERT_EQ(values[1], 1.0f, "arange[1] should be 1");
  ASSERT_EQ(values[2], 2.0f, "arange[2] should be 2");

  freeMemory(mem);
}

static void test_arange_empty_range_positive_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // start >= end with positive step should return NULL
  Tensor *t = T_Arange(&ctx, 5.0f, 5.0f, 1.0f);
  ASSERT_NULL(t, "Arange with start >= end and positive step should return NULL");

  t = T_Arange(&ctx, 10.0f, 5.0f, 1.0f);
  ASSERT_NULL(t, "Arange with start > end and positive step should return NULL");

  freeMemory(mem);
}

static void test_arange_empty_range_negative_step(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // start <= end with negative step should return NULL
  Tensor *t = T_Arange(&ctx, 5.0f, 5.0f, -1.0f);
  ASSERT_NULL(t, "Arange with start <= end and negative step should return NULL");

  t = T_Arange(&ctx, 0.0f, 5.0f, -1.0f);
  ASSERT_NULL(t, "Arange with start < end and negative step should return NULL");

  freeMemory(mem);
}

static void test_arange_with_grad(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true};

  Tensor *t = T_Arange(&ctx, 0.0f, 3.0f, 1.0f);
  ASSERT_NOT_NULL(t, "Arange with grad should create tensor");
  ASSERT_NOT_NULL(t->computation, "Arange should have computation node with grad enabled");
  ASSERT_NOT_NULL(t->computation->grad, "Arange should have gradient tensor with grad enabled");
  ASSERT_EQ(t->computation->grad->size, 3, "Gradient should have same size as tensor");

  freeMemory(mem);
}

void run_tensor_tests(void) {
  printf("=== Tensor Tests ===\n");
  test_zeros_creates_tensor_with_correct_shape();
  test_zeros_values_are_zero();
  test_int_creates_tensor_with_value();
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
  // Slice tests
  test_slice_basic_2d();
  test_slice_shares_data_with_source();
  test_slice_get_at_correct_values();
  test_slice_invalid_range_end_before_start();
  test_slice_range_out_of_bounds();
  test_slice_single_element_range();
  test_slice_full_range();
  test_slice_1d_tensor();
  test_slice_modify_reflects_in_source();
  test_slice_of_slice();
  test_slice_large_4d_tensor();
  test_slice_boundary_access();
  // Reshape tests
  test_reshape_basic_2d_to_1d();
  test_reshape_1d_to_2d();
  test_reshape_preserves_data();
  test_reshape_shares_data_with_source();
  test_reshape_invalid_size_mismatch();
  test_reshape_null_tensor();
  test_reshape_null_shape();
  test_reshape_3d_to_2d();
  test_reshape_view();
  test_reshape_3d_view();
  test_reshape_4d_view();
  test_reshape_4d_view_to_1d();
  test_reshape_then_access_elements();
  // Transpose tests
  test_transpose_basic_2d();
  test_transpose_swaps_dims_and_multipliers();
  test_transpose_shares_data();
  test_transpose_access_elements();
  test_transpose_null_tensor();
  test_transpose_dim_out_of_bounds();
  test_transpose_size_less_than_2();
  test_transpose_3d();
  test_reshape_after_transpose_copies();
  // Add tests
  test_add_basic_same_shape();
  test_add_dtype_mismatch();
  test_add_broadcast_row_vector();
  test_add_broadcast_col_vector();
  test_add_broadcast_scalar();
  test_add_1d_tensors();
  test_add_2d_plus_1d_broadcast();
  test_add_non_contiguous_transposed();
  // Subtract tests
  test_subtract_basic_same_shape();
  test_subtract_broadcast();
  // Multiply tests
  test_multiply_basic_same_shape();
  test_multiply_broadcast_scalar();
  // Divide tests
  test_divide_basic_same_shape();
  test_divide_broadcast();
  // Sum tests
  test_sum_dim0_2d();
  test_sum_dim1_2d();
  test_sum_3d_middle_dim();
  test_sum_dim_out_of_bounds();
  test_sum_null_tensor();
  test_sum_non_contiguous();
  test_sum_1d_tensor();
  test_sum_4d_dim0();
  test_sum_4d_dim1();
  test_sum_4d_dim3();
  test_sum_multiple_reduces_3d();
  test_sum_multiple_reduces_4d();
  test_sum_reduce_to_scalar_2d();
  // Squeeze tests
  test_squeeze_removes_single_dims();
  test_squeeze_middle_dim();
  test_squeeze_no_single_dims();
  test_squeeze_all_ones();
  test_squeeze_shares_data();
  test_squeeze_dim_specific();
  test_squeeze_after_sum();
  // UnSqueeze tests
  test_unsqueeze_dim0();
  test_unsqueeze_middle();
  test_unsqueeze_end();
  test_unsqueeze_1d();
  test_unsqueeze_shares_data();
  test_unsqueeze_dim_out_of_bounds();
  test_unsqueeze_non_contiguous();
  test_squeeze_unsqueeze_roundtrip();
  // Clone tests
  test_clone_basic();
  test_clone_independent_data();
  test_clone_slice();
  test_clone_transposed();
  test_clone_null_tensor();
  // MatMul tests
  test_matmul_2d_basic();
  test_matmul_2d_non_square();
  test_matmul_3d_batch();
  test_matmul_broadcast_batch();
  test_matmul_dtype_mismatch();
  test_matmul_inner_dim_mismatch();
  test_matmul_1d_rejected();
  test_matmul_integer_dtype_rejected();
  // Dot tests
  test_dot_basic();
  test_dot_larger_vectors();
  test_dot_size_mismatch();
  test_dot_dtype_mismatch();
  test_dot_2d_rejected();
  test_dot_integer_dtype_rejected();
  // Gradient initialization tests
  test_grad_t_zeros_without_grad();
  test_grad_t_zeros_with_grad();
  test_grad_t_int_with_grad();
  test_grad_t_float_with_grad();
  test_grad_clone_with_grad();
  test_grad_tensor_shape_matches();
  test_grad_values_initialized_to_zero();
  // Negate tests
  test_negate_f32();
  test_negate_already_negative();
  test_negate_unsigned_rejected();
  // Arange tests
  test_arange_basic_positive_step();
  test_arange_negative_step();
  test_arange_non_integer_step();
  test_arange_default_step();
  test_arange_empty_range_positive_step();
  test_arange_empty_range_negative_step();
  test_arange_with_grad();
}
