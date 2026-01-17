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
      Value val = {.dtype = U8, .as.u8 = (u8)(i * 5 + j)};
      AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 2}, (Range){.start = 1, .end = 3});
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
  Value val = {.dtype = U8, .as.u8 = 42};
  AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 2}, (Range){.start = 0, .end = 3});
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
      Value val = {.dtype = U8, .as.u8 = (u8)(i * 5 + j)};
      AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  // Slice rows 1-2, cols 2-4 -> should get [1,2], [1,3], [1,4], [2,2], [2,3], [2,4]
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 2}, (Range){.start = 2, .end = 4});

  // Access slice[0,0] should be source[1,2] = 1*5+2 = 7
  u32 slice_idx[] = {0, 0};
  Value result;
  Result r = GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.u8, 7, "slice[0,0] should be 7 (source[1,2])");

  // Access slice[1,2] should be source[2,4] = 2*5+4 = 14
  u32 slice_idx2[] = {1, 2};
  r = GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.u8, 14, "slice[1,2] should be 14 (source[2,4])");

  freeMemory(mem);
}

static void test_slice_invalid_range_end_before_start(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 1}, (Range){.start = 0, .end = 3});
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
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 5}, (Range){.start = 0, .end = 3});
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
  Value val = {.dtype = U8, .as.u8 = 99};
  AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);

  Tensor slice;
  // Single element slice at [2,3]
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 2}, (Range){.start = 3, .end = 3});
  ASSERT_EQ(r, OK, "single element slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 1, "slice dim[0] should be 1");
  ASSERT_EQ(slice.shape.dims[1], 1, "slice dim[1] should be 1");

  u32 slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 99, "single element slice value should be 99");

  freeMemory(mem);
}

static void test_slice_full_range(void) {
  u32 dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  // Full range slice
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 0, .end = 2}, (Range){.start = 0, .end = 3});
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
    Value val = {.dtype = U8, .as.u8 = (u8)i};
    AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 1}, val);
  }

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 3, .end = 7});
  ASSERT_EQ(r, OK, "1D slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 5, "1D slice should have 5 elements");

  // slice[0] should be source[3] = 3
  u32 slice_idx[] = {0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.u8, 3, "slice[0] should be 3");

  // slice[4] should be source[7] = 7
  u32 slice_idx2[] = {4};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.u8, 7, "slice[4] should be 7");

  freeMemory(mem);
}

static void test_slice_modify_reflects_in_source(void) {
  u32 dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 2}, (Range){.start = 1, .end = 3});

  // Modify slice[0,1] which maps to source[1,2]
  u32 slice_idx[] = {0, 1};
  Value val = {.dtype = U8, .as.u8 = 77};
  AssignValue(&ctx, &slice, (Dim){.dims = slice_idx, .numOfDims = 2}, val);

  // Check source[1,2]
  u32 src_idx[] = {1, 2};
  Value result;
  GetAt(&tt.tensor, (Dim){.dims = src_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 77, "modifying slice should reflect in source");

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
      Value val = {.dtype = U8, .as.u8 = (u8)(i * 6 + j)};
      AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  // First slice: rows 1-4, cols 1-4 (4x4 region)
  Tensor slice1;
  Slice(&ctx, &tt.tensor, &slice1, (Range){.start = 1, .end = 4}, (Range){.start = 1, .end = 4});

  // Second slice of first slice: rows 1-2, cols 1-2 (2x2 region)
  // This maps to source rows 2-3, cols 2-3
  Tensor slice2;
  Result r = Slice(&ctx, &slice1, &slice2, (Range){.start = 1, .end = 2}, (Range){.start = 1, .end = 2});
  ASSERT_EQ(r, OK, "slice of slice should return OK");
  ASSERT_EQ(slice2.shape.dims[0], 2, "nested slice dim[0] should be 2");
  ASSERT_EQ(slice2.shape.dims[1], 2, "nested slice dim[1] should be 2");

  // slice2[0,0] should be source[2,2] = 2*6+2 = 14
  u32 slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice2, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 14, "nested slice[0,0] should be 14 (source[2,2])");

  freeMemory(mem);
}

static void test_slice_large_4d_tensor(void) {
  u32 dims[] = {8, 10, 12, 6};  // 8x10x12x6 = 5760 elements
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
          Value val = {.dtype = U8, .as.u8 = val_num};
          AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 4}, val);
        }
      }
    }
  }

  Tensor slice;
  // Slice: [2:5, 3:7, 4:9, 1:4] -> 4x5x6x4 = 480 elements
  Result r = Slice(&ctx, &tt.tensor, &slice,
    (Range){.start = 2, .end = 5},
    (Range){.start = 3, .end = 7},
    (Range){.start = 4, .end = 9},
    (Range){.start = 1, .end = 4});
  
  ASSERT_EQ(r, OK, "4D slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 4, "4D slice dim[0] should be 4");
  ASSERT_EQ(slice.shape.dims[1], 5, "4D slice dim[1] should be 5");
  ASSERT_EQ(slice.shape.dims[2], 6, "4D slice dim[2] should be 6");
  ASSERT_EQ(slice.shape.dims[3], 4, "4D slice dim[3] should be 4");

  // Test slice[0,0,0,0] = source[2,3,4,1]
  u32 slice_idx[] = {0, 0, 0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 4}, &result);
  u8 expected = (u8)((2 * 10 * 12 * 6 + 3 * 12 * 6 + 4 * 6 + 1) % 256);
  ASSERT_EQ(result.as.u8, expected, "4D slice[0,0,0,0] should match source[2,3,4,1]");

  // Test slice[3,4,5,3] = source[5,7,9,4]
  u32 slice_idx2[] = {3, 4, 5, 3};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 4}, &result);
  expected = (u8)((5 * 10 * 12 * 6 + 7 * 12 * 6 + 9 * 6 + 4) % 256);
  ASSERT_EQ(result.as.u8, expected, "4D slice[3,4,5,3] should match source[5,7,9,4]");

  // Test middle element: slice[2,2,3,2] = source[4,5,7,3]
  u32 slice_idx3[] = {2, 2, 3, 2};
  GetAt(&slice, (Dim){.dims = slice_idx3, .numOfDims = 4}, &result);
  expected = (u8)((4 * 10 * 12 * 6 + 5 * 12 * 6 + 7 * 6 + 3) % 256);
  ASSERT_EQ(result.as.u8, expected, "4D slice middle element should be correct");

  // Verify all elements in slice match expected source values
  int all_correct = 1;
  for (u32 i = 0; i < 4 && all_correct; i++) {
    for (u32 j = 0; j < 5 && all_correct; j++) {
      for (u32 k = 0; k < 6 && all_correct; k++) {
        for (u32 l = 0; l < 4 && all_correct; l++) {
          u32 s_idx[] = {i, j, k, l};
          GetAt(&slice, (Dim){.dims = s_idx, .numOfDims = 4}, &result);
          
          u32 src_i = i + 2, src_j = j + 3, src_k = k + 4, src_l = l + 1;
          u8 exp = (u8)((src_i * 10 * 12 * 6 + src_j * 12 * 6 + src_k * 6 + src_l) % 256);
          if (result.as.u8 != exp) {
            all_correct = 0;
          }
        }
      }
    }
  }
  ASSERT(all_correct, "all 480 elements in 4D slice should be correct");

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
      Value val = {.dtype = U8, .as.u8 = (u8)(i * 5 + j)};
      AssignValue(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  // Slice rows 2-4, cols 1-3 (3x3 region)
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 4}, (Range){.start = 1, .end = 3});

  // Test all 4 corners of the slice
  Value result;
  
  // Top-left: slice[0,0] = source[2,1] = 11
  u32 tl[] = {0, 0};
  GetAt(&slice, (Dim){.dims = tl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 11, "top-left corner should be 11");

  // Top-right: slice[0,2] = source[2,3] = 13
  u32 tr[] = {0, 2};
  GetAt(&slice, (Dim){.dims = tr, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 13, "top-right corner should be 13");

  // Bottom-left: slice[2,0] = source[4,1] = 21
  u32 bl[] = {2, 0};
  GetAt(&slice, (Dim){.dims = bl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 21, "bottom-left corner should be 21");

  // Bottom-right: slice[2,2] = source[4,3] = 23
  u32 br[] = {2, 2};
  GetAt(&slice, (Dim){.dims = br, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.u8, 23, "bottom-right corner should be 23");

  freeMemory(mem);
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
}
