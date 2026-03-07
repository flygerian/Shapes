#include "test.h"
#include "../../shapes.h"
#include <string.h>

typedef struct {
  Tensor tensor;
  Memory *mem;
} TestTensor;

static TestTensor createZerosTensor(dim_t *dims, u8 numOfDims) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = numOfDims});
  return (TestTensor){.tensor = *t, .mem = mem};
}

static void test_zeros_creates_tensor_with_correct_shape(void) {
  dim_t dims[] = {2, 3};
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 2, "tensor should have 2 dimensions");
  ASSERT_EQ(tt.tensor.dtype, F32, "T_Zeros should create F32 tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "tensor values should be allocated");

  freeMemory(tt.mem);
}

static void test_zeros_values_are_zero(void) {
  dim_t dims[] = {4};
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

  dim_t dims[] = {2, 3};
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
  dim_t dims[] = {5};
  TestTensor tt = createZerosTensor(dims, 1);

  ASSERT_EQ(tt.tensor.shape.numOfDims, 1, "should be 1D tensor");
  ASSERT_NOT_NULL(tt.tensor.values, "values should be allocated");

  freeMemory(tt.mem);
}

static void test_zeros_3d_tensor(void) {
  dim_t dims[] = {2, 3, 4};
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
  dim_t dims[] = {3, 4}; // 3 rows, 4 cols
  TestTensor tt = createZerosTensor(dims, 2);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 4, "2D: multiplier[0] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 1, "2D: multiplier[1] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2], multipliers should be [d1*d2, d2, 1]
static void test_multipliers_3d_tensor(void) {
  dim_t dims[] = {2, 3, 4}; // shape: 2x3x4
  TestTensor tt = createZerosTensor(dims, 3);

  ASSERT_NOT_NULL(tt.tensor.shape.multipliers, "multipliers should be allocated");
  ASSERT_EQ(tt.tensor.shape.multipliers[0], 12, "3D: multiplier[0] should be 12");
  ASSERT_EQ(tt.tensor.shape.multipliers[1], 4, "3D: multiplier[1] should be 4");
  ASSERT_EQ(tt.tensor.shape.multipliers[2], 1, "3D: multiplier[2] should be 1");

  freeMemory(tt.mem);
}

// For shape [d0, d1, d2, d3], multipliers should be [d1*d2*d3, d2*d3, d3, 1]
static void test_multipliers_4d_tensor(void) {
  dim_t dims[] = {2, 3, 4, 5}; // shape: 2x3x4x5
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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {1, 2};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 42.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, OK, "AssignValue should return OK");

  f32 *values = (f32 *)tt.tensor.values;
  ASSERT_EQ(values[1 * 4 + 2], 42.0f, "value at [1,2] should be 42");

  freeMemory(mem);
}

static void test_assign_value_dtype_mismatch(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = U32, .as.u32 = 100};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "should return ERR_DTYPE_MISMATCH");

  freeMemory(mem);
}

static void test_assign_value_dim_mismatch(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {0, 0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 3};
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH");

  freeMemory(mem);
}

static void test_assign_value_out_of_bounds(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {3, 0}; // 3 >= 3, out of bounds
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, &tt.tensor, idx, val);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "should return ERR_OUT_OF_BOUNDS");

  freeMemory(mem);
}

static void test_assign_value_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value val = {.dtype = F32, .as.f32 = 10.0f};

  Result r = AssignValueAt(&ctx, NULL, idx, val);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED for null tensor");

  freeMemory(mem);
}

static void test_assign_value_only_modifies_target_index(void) {
  dim_t dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {1, 2};
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
  dim_t dims[] = {2, 3}; // 6 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx0[] = {0, 0};
  dim_t idx1[] = {0, 2};
  dim_t idx2[] = {1, 1};

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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t idx_dims[] = {1, 2};
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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  dim_t idx_dims[] = {0};
  Dim idx = {.dims = idx_dims, .numOfDims = 1};
  Value result;

  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH");

  freeMemory(tt.mem);
}

static void test_get_at_out_of_bounds(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  dim_t idx_dims[] = {0, 5}; // 5 >= 4, out of bounds
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value result;

  Result r = GetAt(&tt.tensor, idx, &result);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "should return ERR_OUT_OF_BOUNDS");

  freeMemory(tt.mem);
}

static void test_get_at_null_tensor(void) {
  dim_t idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};
  Value result;

  Result r = GetAt(NULL, idx, &result);
  ASSERT_EQ(r, ERR_NULL_PTR, "should return ERR_NULL_PTR for null tensor");
}

static void test_get_at_null_result(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);

  dim_t idx_dims[] = {0, 0};
  Dim idx = {.dims = idx_dims, .numOfDims = 2};

  Result r = GetAt(&tt.tensor, idx, NULL);
  ASSERT_EQ(r, ERR_NULL_PTR, "should return ERR_NULL_PTR for null result");

  freeMemory(tt.mem);
}

// Slice tests
static void test_slice_basic_2d(void) {
  dim_t dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate tensor with values for testing
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      dim_t idx_dims[] = {i, j};
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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Set a value in source
  dim_t idx_dims[] = {1, 2};
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
  dim_t dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j] = i*5 + j
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      dim_t idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 5 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  Tensor slice;
  // Slice rows 1-3 (exclusive), cols 2-5 (exclusive) -> should get [1,2], [1,3], [1,4], [2,2],
  // [2,3], [2,4]
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 2, .end = 5});

  // Access slice[0,0] should be source[1,2] = 1*5+2 = 7
  dim_t slice_idx[] = {0, 0};
  Value result;
  Result r = GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.f32, 7.0f, "slice[0,0] should be 7 (source[1,2])");

  // Access slice[1,2] should be source[2,4] = 2*5+4 = 14
  dim_t slice_idx2[] = {1, 2};
  r = GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(r, OK, "GetAt on slice should return OK");
  ASSERT_EQ(result.as.f32, 14.0f, "slice[1,2] should be 14 (source[2,4])");

  freeMemory(mem);
}

static void test_slice_invalid_range_end_before_start(void) {
  dim_t dims[] = {4, 5};
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
  dim_t dims[] = {4, 5};
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
  dim_t dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Set value at [2,3]
  dim_t idx_dims[] = {2, 3};
  Value val = {.dtype = F32, .as.f32 = 99.0f};
  AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);

  Tensor slice;
  // Single element slice at [2,3] (exclusive end: 2:3 gives 1 element, 3:4 gives 1 element)
  Result r =
      Slice(&ctx, &tt.tensor, &slice, (Range){.start = 2, .end = 3}, (Range){.start = 3, .end = 4});
  ASSERT_EQ(r, OK, "single element slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 1, "slice dim[0] should be 1");
  ASSERT_EQ(slice.shape.dims[1], 1, "slice dim[1] should be 1");

  dim_t slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 99.0f, "single element slice value should be 99");

  freeMemory(mem);
}

static void test_slice_full_range(void) {
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {10};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with values 0-9
  for (u32 i = 0; i < 10; i++) {
    dim_t idx_dims[] = {i};
    Value val = {.dtype = F32, .as.f32 = (f32)i};
    AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 1}, val);
  }

  Tensor slice;
  Result r = Slice(&ctx, &tt.tensor, &slice, (Range){.start = 3, .end = 8});
  ASSERT_EQ(r, OK, "1D slice should return OK");
  ASSERT_EQ(slice.shape.dims[0], 5, "1D slice should have 5 elements");

  // slice[0] should be source[3] = 3
  dim_t slice_idx[] = {0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 3.0f, "slice[0] should be 3");

  // slice[4] should be source[7] = 7
  dim_t slice_idx2[] = {4};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 7.0f, "slice[4] should be 7");

  freeMemory(mem);
}

static void test_slice_modify_reflects_in_source(void) {
  dim_t dims[] = {4, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 4});

  // Modify slice[0,1] which maps to source[1,2]
  dim_t slice_idx[] = {0, 1};
  Value val = {.dtype = F32, .as.f32 = 77.0f};
  AssignValueAt(&ctx, &slice, (Dim){.dims = slice_idx, .numOfDims = 2}, val);

  // Check source[1,2]
  dim_t src_idx[] = {1, 2};
  Value result;
  GetAt(&tt.tensor, (Dim){.dims = src_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 77.0f, "modifying slice should reflect in source");

  freeMemory(mem);
}

static void test_slice_of_slice(void) {
  dim_t dims[] = {6, 6};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j] = i*6 + j
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 6; j++) {
      dim_t idx_dims[] = {i, j};
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
  dim_t slice_idx[] = {0, 0};
  Value result;
  GetAt(&slice2, (Dim){.dims = slice_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 14.0f, "nested slice[0,0] should be 14 (source[2,2])");

  freeMemory(mem);
}

static void test_slice_large_4d_tensor(void) {
  dim_t dims[] = {8, 10, 12, 6}; // 8x10x12x6 = 5760 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: value at [i,j,k,l] = (i*10*12*6 + j*12*6 + k*6 + l) % 256
  for (u32 i = 0; i < 8; i++) {
    for (u32 j = 0; j < 10; j++) {
      for (u32 k = 0; k < 12; k++) {
        for (u32 l = 0; l < 6; l++) {
          dim_t idx_dims[] = {i, j, k, l};
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
  dim_t slice_idx[] = {0, 0, 0, 0};
  Value result;
  GetAt(&slice, (Dim){.dims = slice_idx, .numOfDims = 4}, &result);
  f32 expected = (f32)((2 * 10 * 12 * 6 + 3 * 12 * 6 + 4 * 6 + 1) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice[0,0,0,0] should match source[2,3,4,1]");

  // Test slice[3,4,5,3] = source[5,7,9,4]
  dim_t slice_idx2[] = {3, 4, 5, 3};
  GetAt(&slice, (Dim){.dims = slice_idx2, .numOfDims = 4}, &result);
  expected = (f32)((5 * 10 * 12 * 6 + 7 * 12 * 6 + 9 * 6 + 4) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice[3,4,5,3] should match source[5,7,9,4]");

  // Test middle element: slice[2,2,3,2] = source[4,5,7,3]
  dim_t slice_idx3[] = {2, 2, 3, 2};
  GetAt(&slice, (Dim){.dims = slice_idx3, .numOfDims = 4}, &result);
  expected = (f32)((4 * 10 * 12 * 6 + 5 * 12 * 6 + 7 * 6 + 3) % 256);
  ASSERT_EQ(result.as.f32, expected, "4D slice middle element should be correct");

  // Verify all elements in slice match expected source values
  int all_correct = 1;
  for (u32 i = 0; i < 4 && all_correct; i++) {
    for (u32 j = 0; j < 5 && all_correct; j++) {
      for (u32 k = 0; k < 6 && all_correct; k++) {
        for (u32 l = 0; l < 4 && all_correct; l++) {
          dim_t s_idx[] = {i, j, k, l};
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
  dim_t dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t new_dims[] = {12};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 2D to 1D should return OK");
  ASSERT_EQ(reshaped.shape.numOfDims, 1, "reshaped should have 1 dimension");
  ASSERT_EQ(reshaped.shape.dims[0], 12, "reshaped dim[0] should be 12");

  freeMemory(mem);
}

static void test_reshape_1d_to_2d(void) {
  dim_t dims[] = {24};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t new_dims[] = {4, 6};
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
  dim_t dims[] = {2, 3}; // 6 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with sequential values
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 3 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  dim_t new_dims[] = {6};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Reshape(&ctx, &tt.tensor, &reshaped, newShape);

  // Verify all values preserved in row-major order
  for (u32 i = 0; i < 6; i++) {
    dim_t idx[] = {i};
    Value result;
    GetAt(&reshaped, (Dim){.dims = idx, .numOfDims = 1}, &result);
    ASSERT_EQ(result.as.f32, i, "reshaped data should be preserved");
  }

  freeMemory(mem);
}

static void test_reshape_shares_data_with_source(void) {
  dim_t dims[] = {4, 3};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t new_dims[] = {2, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Reshape(&ctx, &tt.tensor, &reshaped, newShape);

  ASSERT_EQ(reshaped.values, tt.tensor.values, "reshaped should share values pointer");

  // Modify via reshaped, check source
  dim_t r_idx[] = {0, 0};
  Value val = {.dtype = F32, .as.f32 = 55.0f};
  AssignValueAt(&ctx, &reshaped, (Dim){.dims = r_idx, .numOfDims = 2}, val);

  dim_t s_idx[] = {0, 0};
  Value result;
  GetAt(&tt.tensor, (Dim){.dims = s_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 55.0f, "modification via reshaped should reflect in source");

  freeMemory(mem);
}

static void test_reshape_invalid_size_mismatch(void) {
  dim_t dims[] = {3, 4}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  dim_t new_dims[] = {10}; // 10 != 12
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

  dim_t new_dims[] = {6};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, NULL, &reshaped, newShape);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "should return ERR_NULL_TENSOR_PROVIDED for null tensor");

  freeMemory(mem);
}

static void test_reshape_null_shape(void) {
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {2, 3, 4}; // 24 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        dim_t idx_dims[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)(i * 12 + j * 4 + k)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 3}, val);
      }
    }
  }

  dim_t new_dims[] = {6, 4};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Result r = Reshape(&ctx, &tt.tensor, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 3D to 2D should return OK");
  ASSERT_EQ(reshaped.shape.dims[0], 6, "reshaped dim[0] should be 6");
  ASSERT_EQ(reshaped.shape.dims[1], 4, "reshaped dim[1] should be 4");

  // Check reshaped[0,0] = 0, reshaped[5,3] = 23
  dim_t idx1[] = {0, 0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0,0] should be 0");

  dim_t idx2[] = {5, 3};
  GetAt(&reshaped, (Dim){.dims = idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 23.0f, "reshaped[5,3] should be 23");

  freeMemory(mem);
}

static void test_reshape_view(void) {
  dim_t dims[] = {6, 6}; // 36 elements
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 6; j++) {
      dim_t idx_dims[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 6 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 2}, val);
    }
  }

  // Create a slice: rows 1-4 (exclusive), cols 0-6 (exclusive) -> 3x6 = 18 elements
  // Note: This slice is contiguous in memory
  Tensor slice;
  Slice(&ctx, &tt.tensor, &slice, (Range){.start = 1, .end = 4}, (Range){.start = 0, .end = 6});

  // Reshape the slice to 1D (18 elements)
  dim_t new_dims[] = {18};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape of view should return OK");
  ASSERT(!reshaped.isView, "reshaped view should be copied to contiguous array");
  ASSERT_EQ(reshaped.shape.dims[0], 18, "reshaped should have 18 elements");

  freeMemory(mem);
}

static void test_reshape_3d_view(void) {
  dim_t dims[] = {4, 5, 6}; // 120 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      for (u32 k = 0; k < 6; k++) {
        dim_t idx_dims[] = {i, j, k};
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
  dim_t new_dims[] = {10, 6};
  Dim newShape = {.dims = new_dims, .numOfDims = 2};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 3D view to 2D should return OK");
  ASSERT(!reshaped.isView, "reshaped 3D view should be copied to contiguous array");
  ASSERT_EQ(reshaped.shape.dims[0], 10, "reshaped dim[0] should be 10");
  ASSERT_EQ(reshaped.shape.dims[1], 6, "reshaped dim[1] should be 6");

  // Verify reshaped[0,0] = slice[0,0,0] = source[1,0,0] = 1*30 = 30
  dim_t r_idx[] = {0, 0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = r_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 30.0f, "reshaped[0,0] should be 30");

  freeMemory(mem);
}

static void test_reshape_4d_view(void) {
  dim_t dims[] = {3, 4, 5, 6}; // 360 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with pattern
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      for (u32 k = 0; k < 5; k++) {
        for (u32 l = 0; l < 6; l++) {
          dim_t idx_dims[] = {i, j, k, l};
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
  dim_t new_dims[] = {6, 5, 6};
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
  dim_t dims[] = {2, 3, 4, 5}; // 120 elements
  TestTensor tt = createZerosTensor(dims, 4);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate sequentially
  u8 counter = 0;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        for (u32 l = 0; l < 5; l++) {
          dim_t idx_dims[] = {i, j, k, l};
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
  dim_t new_dims[] = {60};
  Dim newShape = {.dims = new_dims, .numOfDims = 1};

  Tensor reshaped;
  Result r = Reshape(&ctx, &slice, &reshaped, newShape);
  ASSERT_EQ(r, OK, "Reshape 4D view to 1D should return OK");
  ASSERT_EQ(reshaped.shape.numOfDims, 1, "reshaped should have 1 dimension");
  ASSERT_EQ(reshaped.shape.dims[0], 60, "reshaped should have 60 elements");

  // Check first element: reshaped[0] = slice[0,0,0,0] = source[0,0,0,0] = 0
  dim_t idx1[] = {0};
  Value result;
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0] should be 0");

  freeMemory(mem);
}

static void test_reshape_then_access_elements(void) {
  dim_t dims[] = {2, 2, 3}; // 12 elements
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate with values 0-11
  u8 counter = 0;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 3; k++) {
        dim_t idx_dims[] = {i, j, k};
        Value val = {.dtype = F32, .as.f32 = (f32)(counter++)};
        AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx_dims, .numOfDims = 3}, val);
      }
    }
  }

  // Reshape to 4x3
  dim_t new_dims[] = {4, 3};
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
      dim_t idx[] = {i, j};
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
  dim_t dims[] = {3, 4}; // 3 rows, 4 cols
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
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  ASSERT_EQ(transposed.values, tt.tensor.values, "transposed should share values pointer");

  freeMemory(mem);
}

static void test_transpose_access_elements(void) {
  dim_t dims[] = {2, 3};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: source[i,j] = i*3 + j
  // source[0,0]=0, source[0,1]=1, source[0,2]=2
  // source[1,0]=3, source[1,1]=4, source[1,2]=5
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t idx1[] = {0, 0};
  GetAt(&transposed, (Dim){.dims = idx1, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "transposed[0,0] should be 0");

  dim_t idx2[] = {0, 1};
  GetAt(&transposed, (Dim){.dims = idx2, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 3.0f, "transposed[0,1] should be 3");

  dim_t idx3[] = {1, 0};
  GetAt(&transposed, (Dim){.dims = idx3, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "transposed[1,0] should be 1");

  dim_t idx4[] = {2, 1};
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
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "should return ERR_DIM_MISMATCH for out of bounds dim");

  freeMemory(mem);
}

static void test_transpose_size_less_than_2(void) {
  dim_t dims[] = {1};
  TestTensor tt = createZerosTensor(dims, 1);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  Tensor transposed;
  Result r = Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)0);
  ASSERT_EQ(r, ERR_NO_OP, "should return ERR_NO_OP for tensor with size < 2");

  freeMemory(mem);
}

static void test_transpose_3d(void) {
  dim_t dims[] = {2, 3, 4}; // 2x3x4
  TestTensor tt = createZerosTensor(dims, 3);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate: source[i,j,k] = i*12 + j*4 + k
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        dim_t idx[] = {i, j, k};
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

  dim_t idx1[] = {0, 0, 0};
  GetAt(&transposed, (Dim){.dims = idx1, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "transposed[0,0,0] should be 0");

  dim_t idx2[] = {3, 2, 1};
  GetAt(&transposed, (Dim){.dims = idx2, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 23.0f, "transposed[3,2,1] should be 23");

  freeMemory(mem);
}

static void test_reshape_after_transpose_copies(void) {
  dim_t dims[] = {3, 4};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      dim_t idx[] = {i, j};
      Value val = {.dtype = F32, .as.f32 = (f32)(i * 4 + j)};
      AssignValueAt(&ctx, &tt.tensor, (Dim){.dims = idx, .numOfDims = 2}, val);
    }
  }

  // Transpose: 3x4 -> 4x3
  Tensor transposed;
  Transpose(&ctx, &tt.tensor, &transposed, (dim_t)0, (dim_t)1);

  // Reshape transposed to 1D: 12 elements
  dim_t new_dims[] = {12};
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
  dim_t idx0[] = {0};
  GetAt(&reshaped, (Dim){.dims = idx0, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 0.0f, "reshaped[0] should be 0");

  dim_t idx1[] = {1};
  GetAt(&reshaped, (Dim){.dims = idx1, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "reshaped[1] should be 4 (transposed[0,1])");

  dim_t idx3[] = {3};
  GetAt(&reshaped, (Dim){.dims = idx3, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "reshaped[3] should be 1 (transposed[1,0])");

  freeMemory(mem);
}

// Add tests
static void test_add_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  // b = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims[] = {2, 2};
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
  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10, 20, 30]]
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {0, j};
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
  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {2, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10], [20]]
  for (u32 i = 0; i < 2; i++) {
    dim_t idx[] = {i, 0};
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
  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {1, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[100]]
  dim_t idx_b[] = {0, 0};
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

  dim_t dims[] = {4};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});
  Tensor dest;

  // a = [1, 2, 3, 4], b = [10, 20, 30, 40]
  for (u32 i = 0; i < 4; i++) {
    dim_t idx[] = {i};
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

  dim_t dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  // b = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[1, 2, 3]]
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {0, j};
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

  dim_t dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  // b = [[2,2,2], [3,3,3]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {1, 1};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[5]]
  dim_t idx_b[] = {0, 0};
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

  dim_t dims[] = {2, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  // b = [[2,4,5], [8,10,12]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {1, 3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)((i * 3 + j + 1) * 10)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[2, 5, 10]]
  u8 divisors[] = {2, 5, 10};
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {0, j};
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
  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {3, 2};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 2});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [[10,20], [30,40], [50,60]] (3x2)
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 2; j++) {
      dim_t idx[] = {i, j};
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
  dim_t dims_a[] = {2, 3};
  dim_t dims_b[] = {3};
  Tensor *a = T_Zeros(&ctx, (Dim){.dims = dims_a, .numOfDims = 2});
  Tensor *b = T_Zeros(&ctx, (Dim){.dims = dims_b, .numOfDims = 1});
  Tensor dest;

  // a = [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      Value v = {.dtype = F32, .as.f32 = (f32)(i * 3 + j + 1)};
      AssignValueAt(&ctx, a, (Dim){.dims = idx, .numOfDims = 2}, v);
    }
  }

  // b = [10, 20, 30]
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {j};
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

// Comparison binary op tests
static void test_greater_than_basic_same_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);
  Tensor dest;

  i8 aVals[] = {1, 4, 3, 2, 8, 0};
  i8 bVals[] = {2, 4, 1, 3, 7, 0};
  memcpy(a->values, aVals, sizeof(aVals));
  memcpy(b->values, bVals, sizeof(bVals));

  Result r = GreaterThan(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "GreaterThan should return OK");
  ASSERT_EQ(dest.dtype, BOOL, "GreaterThan should return BOOL dtype");

  bool *vals = (bool *)dest.values;
  ASSERT_EQ(vals[0], false, "1 > 2 should be false");
  ASSERT_EQ(vals[1], false, "4 > 4 should be false");
  ASSERT_EQ(vals[2], true, "3 > 1 should be true");
  ASSERT_EQ(vals[3], false, "2 > 3 should be false");
  ASSERT_EQ(vals[4], true, "8 > 7 should be true");
  ASSERT_EQ(vals[5], false, "0 > 0 should be false");

  freeMemory(mem);
}

static void test_greater_or_equal_and_less_or_equal(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);
  Tensor ge, le;

  i8 aVals[] = {1, 4, 5};
  i8 bVals[] = {2, 4, 3};
  memcpy(a->values, aVals, sizeof(aVals));
  memcpy(b->values, bVals, sizeof(bVals));

  Result r = GreaterThanOrEqual(&ctx, a, b, &ge);
  ASSERT_EQ(r, OK, "GreaterThanOrEqual should return OK");
  r = LessThanOrEqual(&ctx, a, b, &le);
  ASSERT_EQ(r, OK, "LessThanOrEqual should return OK");

  ASSERT_EQ(ge.dtype, BOOL, "GreaterThanOrEqual should return BOOL dtype");
  ASSERT_EQ(le.dtype, BOOL, "LessThanOrEqual should return BOOL dtype");

  bool *geVals = (bool *)ge.values;
  ASSERT_EQ(geVals[0], false, "1 >= 2 should be false");
  ASSERT_EQ(geVals[1], true, "4 >= 4 should be true");
  ASSERT_EQ(geVals[2], true, "5 >= 3 should be true");

  bool *leVals = (bool *)le.values;
  ASSERT_EQ(leVals[0], true, "1 <= 2 should be true");
  ASSERT_EQ(leVals[1], true, "4 <= 4 should be true");
  ASSERT_EQ(leVals[2], false, "5 <= 3 should be false");

  freeMemory(mem);
}

static void test_less_than_broadcast_row_vector(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dimsA[] = {2, 3};
  dim_t dimsB[] = {1, 3};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dimsA, .numOfDims = 2}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dimsB, .numOfDims = 2}, 0);
  Tensor dest;

  // a = [[1,4,3], [5,2,7]], b = [[2,2,7]]
  i8 aVals[] = {1, 4, 3, 5, 2, 7};
  i8 bVals[] = {2, 2, 7};
  memcpy(a->values, aVals, sizeof(aVals));
  memcpy(b->values, bVals, sizeof(bVals));

  Result r = LessThan(&ctx, a, b, &dest);
  ASSERT_EQ(r, OK, "LessThan with broadcast should return OK");
  ASSERT_EQ(dest.dtype, BOOL, "LessThan should return BOOL dtype");
  ASSERT_EQ(dest.shape.numOfDims, 2, "LessThan result should keep rank");
  ASSERT_EQ(dest.shape.dims[0], 2, "LessThan result dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 3, "LessThan result dim 1 should be 3");

  bool *vals = (bool *)dest.values;
  // [[1<2,4<2,3<7],[5<2,2<2,7<7]] => [[1,0,1],[0,0,0]]
  ASSERT_EQ(vals[0], true, "result[0,0] should be true");
  ASSERT_EQ(vals[1], false, "result[0,1] should be false");
  ASSERT_EQ(vals[2], true, "result[0,2] should be true");
  ASSERT_EQ(vals[3], false, "result[1,0] should be false");
  ASSERT_EQ(vals[4], false, "result[1,1] should be false");
  ASSERT_EQ(vals[5], false, "result[1,2] should be false");

  freeMemory(mem);
}

static void test_comparison_dtype_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *a = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);
  Tensor *b = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);
  b->dtype = U8; // force mismatch
  Tensor dest;

  Result r = GreaterThan(&ctx, a, b, &dest);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "GreaterThan should fail on dtype mismatch");

  r = LessThanOrEqual(&ctx, a, b, &dest);
  ASSERT_EQ(r, ERR_DTYPE_MISMATCH, "LessThanOrEqual should fail on dtype mismatch");

  freeMemory(mem);
}

// Sum tests
static void test_sum_dim0_2d(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 2x3 tensor: [[1,2,3], [4,5,6]]
  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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
  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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
  dim_t dims[] = {2, 3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  // Fill with sequential values 1-12
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 2; k++) {
        dim_t idx[] = {i, j, k};
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

  dim_t dims[] = {2, 3};
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
  dim_t dims[] = {3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // t = [[1,2], [3,4], [5,6]]
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 2; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims[] = {5};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  // t = [1, 2, 3, 4, 5]
  for (u32 i = 0; i < 5; i++) {
    dim_t idx[] = {i};
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
  dim_t dims[] = {2, 2, 2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 3; l++) {
          dim_t idx[] = {i, j, k, l};
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
  dim_t dims[] = {2, 3, 2, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 2; l++) {
          dim_t idx[] = {i, j, k, l};
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
  dim_t dims[] = {2, 2, 2, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with sequential values 1-32
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 2; k++) {
        for (u32 l = 0; l < 4; l++) {
          dim_t idx[] = {i, j, k, l};
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
  dim_t dims[] = {2, 3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  // Fill with sequential values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      for (u32 k = 0; k < 4; k++) {
        dim_t idx[] = {i, j, k};
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
  dim_t dims[] = {2, 2, 3, 2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 4});

  // Fill with values 1-24
  u8 val = 1;
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 2; j++) {
      for (u32 k = 0; k < 3; k++) {
        for (u32 l = 0; l < 2; l++) {
          dim_t idx[] = {i, j, k, l};
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
  dim_t dims[] = {3, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // Fill with 1-12
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      dim_t idx[] = {i, j};
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
  dim_t dims[] = {1, 3, 1, 4};
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
  dim_t dims[] = {2, 1, 3};
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
  dim_t dims[] = {2, 3, 4};
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
  dim_t dims[] = {1, 1, 1};
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

  dim_t dims[] = {1, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // Set a value
  dim_t idx[] = {0, 1};
  Value v = {.dtype = F32, .as.f32 = 42.0f};
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 2}, v);

  Tensor squeezed;
  Squeeze(&ctx, t, &squeezed);

  // Check value is accessible in squeezed tensor
  dim_t sq_idx[] = {1};
  Value result;
  GetAt(&squeezed, (Dim){.dims = sq_idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 42.0f, "squeezed should share data with source");

  freeMemory(mem);
}

static void test_squeeze_dim_specific(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [1, 3, 1, 4] squeeze dim 0 -> [3, 1, 4]
  dim_t dims[] = {1, 3, 1, 4};
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
  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {5};
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

  dim_t dims[] = {3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  dim_t idx[] = {1};
  Value v = {.dtype = F32, .as.f32 = 42.0f};
  AssignValueAt(&ctx, t, (Dim){.dims = idx, .numOfDims = 1}, v);

  Tensor unsqueezed;
  UnSqueeze(&ctx, t, &unsqueezed, 0);

  // Access via [0, 1]
  dim_t new_idx[] = {0, 1};
  Value result;
  GetAt(&unsqueezed, (Dim){.dims = new_idx, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 42.0f, "unsqueezed should share data");

  freeMemory(mem);
}

static void test_unsqueeze_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
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
  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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
  dim_t idx1[] = {0, 0, 0};
  GetAt(&unsqueezed, (Dim){.dims = idx1, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 1.0f, "unsqueezed[0,0,0] should be 1");

  dim_t idx2[] = {0, 0, 1};
  GetAt(&unsqueezed, (Dim){.dims = idx2, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "unsqueezed[0,0,1] should be 4");

  dim_t idx3[] = {0, 1, 0};
  GetAt(&unsqueezed, (Dim){.dims = idx3, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 2.0f, "unsqueezed[0,1,0] should be 2");

  dim_t idx4[] = {0, 2, 1};
  GetAt(&unsqueezed, (Dim){.dims = idx4, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 6.0f, "unsqueezed[0,2,1] should be 6");

  freeMemory(mem);
}

static void test_squeeze_unsqueeze_roundtrip(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // [2, 3] -> squeeze (no change) -> unsqueeze dim 1 -> [2, 1, 3] -> squeeze -> [2, 3]
  dim_t dims[] = {2, 1, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 3});

  for (u32 i = 0; i < 2; i++) {
    for (u32 k = 0; k < 3; k++) {
      dim_t idx[] = {i, 0, k};
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
  dim_t idx[] = {1, 0, 2};
  Value result;
  GetAt(&unsqueezed, (Dim){.dims = idx, .numOfDims = 3}, &result);
  ASSERT_EQ(result.as.f32, 6.0f, "data should be preserved");

  freeMemory(mem);
}

// Clone tests
static void test_clone_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims[] = {3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});

  dim_t idx[] = {1};
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

  dim_t dims[] = {4, 4};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 4; j++) {
      dim_t idx[] = {i, j};
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

  dim_t dims[] = {2, 3};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});

  // [[1,2,3], [4,5,6]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
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

// Phase 1 tests: view offset/boundary correctness

// Test 1: Slice boundary propagation – GetAt on a 2D slice with non-zero starts
static void test_view_slice_boundary_propagation(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 4x5 tensor, values[i][j] = i*5 + j
  dim_t dims[] = {4, 5};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 5; j++) {
      ((f32 *)x->values)[i * 5 + j] = (f32)(i * 5 + j);
    }
  }

  // Slice rows 1-3 (exclusive), cols 2-5 (exclusive) -> 2x3 view
  Tensor s;
  Result r = Slice(&ctx, x, &s, (Range){.start = 1, .end = 3}, (Range){.start = 2, .end = 5});
  ASSERT_EQ(r, OK, "slice should succeed");
  ASSERT_EQ(s.shape.dims[0], 2, "slice dim[0] should be 2");
  ASSERT_EQ(s.shape.dims[1], 3, "slice dim[1] should be 3");

  // s[0,0] should be x[1,2] = 1*5+2 = 7
  dim_t idx00[] = {0, 0};
  Value result;
  GetAt(&s, (Dim){.dims = idx00, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 7.0f, "s[0,0] should be x[1,2]=7");

  // s[0,2] should be x[1,4] = 1*5+4 = 9
  dim_t idx02[] = {0, 2};
  GetAt(&s, (Dim){.dims = idx02, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 9.0f, "s[0,2] should be x[1,4]=9");

  // s[1,0] should be x[2,2] = 2*5+2 = 12
  dim_t idx10[] = {1, 0};
  GetAt(&s, (Dim){.dims = idx10, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 12.0f, "s[1,0] should be x[2,2]=12");

  // s[1,2] should be x[2,4] = 2*5+4 = 14
  dim_t idx12[] = {1, 2};
  GetAt(&s, (Dim){.dims = idx12, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 14.0f, "s[1,2] should be x[2,4]=14");

  freeMemory(mem);
}

// Test 2: Nested slice correctness
static void test_view_nested_slice_correctness(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 6x6 tensor, values[i][j] = i*6 + j
  dim_t dims[] = {6, 6};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 6; j++) {
      ((f32 *)x->values)[i * 6 + j] = (f32)(i * 6 + j);
    }
  }

  // s1 = x[1:5, 1:5] -> 4x4
  Tensor s1;
  Slice(&ctx, x, &s1, (Range){.start = 1, .end = 5}, (Range){.start = 1, .end = 5});

  // s2 = s1[1:3, 1:3] -> 2x2, which maps to x[2:4, 2:4]
  Tensor s2;
  Result r = Slice(&ctx, &s1, &s2, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 3});
  ASSERT_EQ(r, OK, "nested slice should succeed");
  ASSERT_EQ(s2.shape.dims[0], 2, "nested slice dim[0] should be 2");
  ASSERT_EQ(s2.shape.dims[1], 2, "nested slice dim[1] should be 2");

  // s2[0,0] = x[2,2] = 2*6+2 = 14
  dim_t idx00[] = {0, 0};
  Value result;
  GetAt(&s2, (Dim){.dims = idx00, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 14.0f, "s2[0,0] should be x[2,2]=14");

  // s2[0,1] = x[2,3] = 2*6+3 = 15
  dim_t idx01[] = {0, 1};
  GetAt(&s2, (Dim){.dims = idx01, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 15.0f, "s2[0,1] should be x[2,3]=15");

  // s2[1,0] = x[3,2] = 3*6+2 = 20
  dim_t idx10[] = {1, 0};
  GetAt(&s2, (Dim){.dims = idx10, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 20.0f, "s2[1,0] should be x[3,2]=20");

  // s2[1,1] = x[3,3] = 3*6+3 = 21
  dim_t idx11[] = {1, 1};
  GetAt(&s2, (Dim){.dims = idx11, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 21.0f, "s2[1,1] should be x[3,3]=21");

  freeMemory(mem);
}

// Test 3: GetTensorAt on a sliced tensor
static void test_view_get_tensor_at_on_slice(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 5x4 tensor, values[i][j] = i*4 + j
  dim_t dims[] = {5, 4};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 5; i++) {
    for (u32 j = 0; j < 4; j++) {
      ((f32 *)x->values)[i * 4 + j] = (f32)(i * 4 + j);
    }
  }

  // s = x[2:5, 0:4] -> 3x4 view (row offset = 2)
  Tensor s;
  Slice(&ctx, x, &s, (Range){.start = 2, .end = 5}, (Range){.start = 0, .end = 4});

  // row = GetTensorAt(s, 1) -> should be x[3, :] = [12, 13, 14, 15]
  Tensor row;
  Result r = GetTensorAt(&ctx, &s, 1, &row);
  ASSERT_EQ(r, OK, "GetTensorAt on slice should succeed");
  ASSERT_EQ(row.shape.numOfDims, 1, "row should be 1D");
  ASSERT_EQ(row.shape.dims[0], 4, "row should have 4 elements");

  dim_t idx[] = {0};
  Value result;
  GetAt(&row, (Dim){.dims = idx, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 12.0f, "row[0] should be x[3,0]=12");

  dim_t idx2[] = {3};
  GetAt(&row, (Dim){.dims = idx2, .numOfDims = 1}, &result);
  ASSERT_EQ(result.as.f32, 15.0f, "row[3] should be x[3,3]=15");

  freeMemory(mem);
}

// Test 4: Advanced indexing on view input (read correctness)
static void test_view_advanced_indexing_on_slice(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 6x4 tensor, values[i][j] = i*4 + j
  dim_t dims[] = {6, 4};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 6; i++) {
    for (u32 j = 0; j < 4; j++) {
      ((f32 *)x->values)[i * 4 + j] = (f32)(i * 4 + j);
    }
  }

  // s = x[2:6, 0:4] -> 4x4 (row offset = 2; rows 2,3,4,5 of x)
  Tensor s;
  Slice(&ctx, x, &s, (Range){.start = 2, .end = 6}, (Range){.start = 0, .end = 4});

  // IndexWithTensor(s, [0, 2]) should gather s[0,:] and s[2,:] = x[2,:] and x[4,:]
  dim_t idxDims[] = {2};
  Tensor *indices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)indices->values)[0] = 0;
  ((i8 *)indices->values)[1] = 2;

  Tensor result;
  Result r = IndexWithTensor(&ctx, &s, indices, &result);
  ASSERT_EQ(r, OK, "IndexWithTensor on slice should succeed");
  ASSERT_EQ(result.shape.numOfDims, 2, "result should be 2D");
  ASSERT_EQ(result.shape.dims[0], 2, "result dim[0] should be 2");
  ASSERT_EQ(result.shape.dims[1], 4, "result dim[1] should be 4");

  f32 *vals = (f32 *)result.values;
  // s[0,:] = x[2,:] = [8,9,10,11]
  ASSERT_EQ(vals[0], 8.0f, "result[0,0] should be x[2,0]=8");
  ASSERT_EQ(vals[1], 9.0f, "result[0,1] should be x[2,1]=9");
  ASSERT_EQ(vals[2], 10.0f, "result[0,2] should be x[2,2]=10");
  ASSERT_EQ(vals[3], 11.0f, "result[0,3] should be x[2,3]=11");
  // s[2,:] = x[4,:] = [16,17,18,19]
  ASSERT_EQ(vals[4], 16.0f, "result[1,0] should be x[4,0]=16");
  ASSERT_EQ(vals[5], 17.0f, "result[1,1] should be x[4,1]=17");
  ASSERT_EQ(vals[6], 18.0f, "result[1,2] should be x[4,2]=18");
  ASSERT_EQ(vals[7], 19.0f, "result[1,3] should be x[4,3]=19");

  freeMemory(mem);
}

// Test: AddInPlace on a slice view mutates the correct region of the base tensor
static void test_add_in_place_on_slice_view(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 4x4 base tensor with values [0..15]
  dim_t dims[] = {4, 4};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 16; i++) {
    ((f32 *)x->values)[i] = (f32)i;
  }

  // s = x[1:3, 1:3] -> 2x2 view covering x[1,1], x[1,2], x[2,1], x[2,2] = 5,6,9,10
  Tensor s;
  Slice(&ctx, x, &s, (Range){.start = 1, .end = 3}, (Range){.start = 1, .end = 3});

  // ones = 2x2 tensor of all 1.0
  Tensor *ones = T_Float(&ctx, (Dim){.dims = (dim_t[]){2, 2}, .numOfDims = 2}, 1.0f);

  Result r = AddInPlace(&ctx, &s, ones);
  ASSERT_EQ(r, OK, "AddInPlace on slice view should succeed");

  // Verify base tensor: unchanged regions outside slice
  ASSERT_EQ(((f32 *)x->values)[0], 0.0f, "x[0,0] should be unchanged (0)");
  ASSERT_EQ(((f32 *)x->values)[3], 3.0f, "x[0,3] should be unchanged (3)");
  ASSERT_EQ(((f32 *)x->values)[15], 15.0f, "x[3,3] should be unchanged (15)");

  // Slice region should be incremented: 5->6, 6->7, 9->10, 10->11
  ASSERT_EQ(((f32 *)x->values)[5], 6.0f, "x[1,1] should be 6 after +1");
  ASSERT_EQ(((f32 *)x->values)[6], 7.0f, "x[1,2] should be 7 after +1");
  ASSERT_EQ(((f32 *)x->values)[9], 10.0f, "x[2,1] should be 10 after +1");
  ASSERT_EQ(((f32 *)x->values)[10], 11.0f, "x[2,2] should be 11 after +1");

  freeMemory(mem);
}

// Test: Boundary deep-copy safety - each view gets its own boundary array
// so modifying or zeroing one boundary does not corrupt another.
static void test_view_boundary_deep_copy_transpose(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 3x4 base tensor, values[i][j] = i*4 + j (0..11)
  dim_t dims[] = {3, 4};
  Tensor *x = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u32 i = 0; i < 12; i++) {
    ((f32 *)x->values)[i] = (f32)i;
  }

  // s = x[1:3, 0:4] -> 2x4 view (rows 1 and 2 of x)
  Tensor s;
  Slice(&ctx, x, &s, (Range){.start = 1, .end = 3}, (Range){.start = 0, .end = 4});

  // t = Transpose(s, 0, 1) -> 4x2 view
  Tensor t;
  Result r = Transpose(&ctx, &s, &t, (dim_t)0, (dim_t)1);
  ASSERT_EQ(r, OK, "Transpose of slice should succeed");

  // Verify s and t have DIFFERENT boundary pointers (deep-copy)
  ASSERT_NEQ((uintptr_t)s.boundary, (uintptr_t)t.boundary,
             "s and t should have independent boundary arrays");

  // Verify correct values through both views
  // s[0,0] = x[1,0] = 4
  dim_t s00[] = {0, 0};
  Value result;
  GetAt(&s, (Dim){.dims = s00, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "s[0,0] should be x[1,0]=4");

  // t[0,0] = s[0,0] = x[1,0] = 4
  dim_t t00[] = {0, 0};
  GetAt(&t, (Dim){.dims = t00, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "t[0,0] should be x[1,0]=4");

  // t[1,0] = s[0,1] = x[1,1] = 5
  dim_t t10[] = {1, 0};
  GetAt(&t, (Dim){.dims = t10, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 5.0f, "t[1,0] should be x[1,1]=5");

  // Corrupt s's boundary to prove t is unaffected
  s.boundary[0].start = 999;

  // t should still read correctly
  GetAt(&t, (Dim){.dims = t00, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 4.0f, "t[0,0] should still be 4 after corrupting s boundary");

  freeMemory(mem);
}

static void test_slice_boundary_access(void) {
  dim_t dims[] = {5, 5};
  TestTensor tt = createZerosTensor(dims, 2);
  Memory *mem = tt.mem;
  Context ctx = {.memory = mem};

  // Populate
  for (u32 i = 0; i < 5; i++) {
    for (u32 j = 0; j < 5; j++) {
      dim_t idx_dims[] = {i, j};
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
  dim_t tl[] = {0, 0};
  GetAt(&slice, (Dim){.dims = tl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 11.0f, "top-left corner should be 11");

  // Top-right: slice[0,2] = source[2,3] = 13
  dim_t tr[] = {0, 2};
  GetAt(&slice, (Dim){.dims = tr, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 13.0f, "top-right corner should be 13");

  // Bottom-left: slice[2,0] = source[4,1] = 21
  dim_t bl[] = {2, 0};
  GetAt(&slice, (Dim){.dims = bl, .numOfDims = 2}, &result);
  ASSERT_EQ(result.as.f32, 21.0f, "bottom-left corner should be 21");

  // Bottom-right: slice[2,2] = source[4,3] = 23
  dim_t br[] = {2, 2};
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

  dim_t dims[] = {2, 2};
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

static void test_negate_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  dim_t dims[] = {3};
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
  dim_t dims[] = {2};
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
  dim_t dims[] = {2};
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

static void test_index_with_tensor_2d_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create 3x4 source tensor: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
  dim_t dims[] = {3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  for (u8 i = 0; i < 12; i++) {
    ((f32 *)source->values)[i] = (f32)i;
  }

  // Create row and column indices: extract [0,1], [1,2], [2,3] -> [1, 6, 11]
  dim_t idxDims[] = {3};
  Tensor *rowIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)rowIndices->values)[0] = 0;
  ((i8 *)rowIndices->values)[1] = 1;
  ((i8 *)rowIndices->values)[2] = 2;

  Tensor *colIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)colIndices->values)[0] = 1;
  ((i8 *)colIndices->values)[1] = 2;
  ((i8 *)colIndices->values)[2] = 3;

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, rowIndices, colIndices, &result);
  ASSERT_EQ(r, OK, "IndexWithTensor2d should succeed");
  ASSERT_EQ(result.shape.numOfDims, 1, "Result should be 1D");
  ASSERT_EQ(result.shape.dims[0], 3, "Result should have 3 elements");

  f32 *vals = (f32 *)result.values;
  ASSERT_EQ(vals[0], 1.0f, "First element should be 1 (source[0,1])");
  ASSERT_EQ(vals[1], 6.0f, "Second element should be 6 (source[1,2])");
  ASSERT_EQ(vals[2], 11.0f, "Third element should be 11 (source[2,3])");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_3d_source(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create 2x3x4 source tensor
  dim_t dims[] = {2, 3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 3}, 0.0f);
  for (u8 i = 0; i < 24; i++) {
    ((f32 *)source->values)[i] = (f32)i;
  }

  // Extract [0,0,:], [1,2,:] -> first 4 elements and last 4 elements
  dim_t idxDims[] = {2};
  Tensor *rowIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)rowIndices->values)[0] = 0;
  ((i8 *)rowIndices->values)[1] = 1;

  Tensor *colIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)colIndices->values)[0] = 0;
  ((i8 *)colIndices->values)[1] = 2;

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, rowIndices, colIndices, &result);
  ASSERT_EQ(r, OK, "IndexWithTensor2d with 3D source should succeed");
  ASSERT_EQ(result.shape.numOfDims, 2, "Result should be 2D");
  ASSERT_EQ(result.shape.dims[0], 2, "Result dim 0 should be 2");
  ASSERT_EQ(result.shape.dims[1], 4, "Result dim 1 should be 4");

  f32 *vals = (f32 *)result.values;
  // First row: source[0,0,:] = [0,1,2,3]
  ASSERT_EQ(vals[0], 0.0f, "vals[0] should be 0");
  ASSERT_EQ(vals[1], 1.0f, "vals[1] should be 1");
  ASSERT_EQ(vals[2], 2.0f, "vals[2] should be 2");
  ASSERT_EQ(vals[3], 3.0f, "vals[3] should be 3");
  // Second row: source[1,2,:] = [20,21,22,23]
  ASSERT_EQ(vals[4], 20.0f, "vals[4] should be 20");
  ASSERT_EQ(vals[5], 21.0f, "vals[5] should be 21");
  ASSERT_EQ(vals[6], 22.0f, "vals[6] should be 22");
  ASSERT_EQ(vals[7], 23.0f, "vals[7] should be 23");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  dim_t idxDims[] = {2};
  Tensor *indices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, NULL, indices, indices, &result);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Should fail with null source");

  r = IndexWithTensor2d(&ctx, source, NULL, indices, &result);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Should fail with null row indices");

  r = IndexWithTensor2d(&ctx, source, indices, NULL, &result);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Should fail with null col indices");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_insufficient_dims(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // 1D source should fail
  dim_t dims[] = {4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0.0f);
  dim_t idxDims[] = {2};
  Tensor *indices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, indices, indices, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Should fail with 1D source");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_mismatched_indices(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  dim_t rowDims[] = {2};
  dim_t colDims[] = {3};
  Tensor *rowIndices = T_Int(&ctx, (Dim){.dims = rowDims, .numOfDims = 1}, 0);
  Tensor *colIndices = T_Int(&ctx, (Dim){.dims = colDims, .numOfDims = 1}, 0);

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, rowIndices, colIndices, &result);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Should fail with mismatched index sizes");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  dim_t idxDims[] = {2};
  Tensor *rowIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)rowIndices->values)[0] = 0;
  ((i8 *)rowIndices->values)[1] = 5; // Out of bounds

  Tensor *colIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);
  ((i8 *)colIndices->values)[0] = 0;
  ((i8 *)colIndices->values)[1] = 0;

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, rowIndices, colIndices, &result);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "Should fail with out of bounds row index");

  // Reset and test column out of bounds
  ((i8 *)rowIndices->values)[1] = 0;
  ((i8 *)colIndices->values)[1] = 5; // Out of bounds

  r = IndexWithTensor2d(&ctx, source, rowIndices, colIndices, &result);
  ASSERT_EQ(r, ERR_OUT_OF_BOUNDS, "Should fail with out of bounds col index");

  freeMemory(mem);
}

static void test_index_with_tensor_2d_non_int_indices(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
  Tensor *source = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);
  dim_t idxDims[] = {2};
  Tensor *floatIndices = T_Float(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0.0f);
  Tensor *intIndices = T_Int(&ctx, (Dim){.dims = idxDims, .numOfDims = 1}, 0);

  Tensor result;
  Result r = IndexWithTensor2d(&ctx, source, floatIndices, intIndices, &result);
  ASSERT_EQ(r, ERR_ONLY_INT_TYPE_ALLOWED, "Should fail with float row indices");

  r = IndexWithTensor2d(&ctx, source, intIndices, floatIndices, &result);
  ASSERT_EQ(r, ERR_ONLY_INT_TYPE_ALLOWED, "Should fail with float col indices");

  freeMemory(mem);
}

// Mean tests
static void test_mean_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // Set values: [[1, 2, 3], [4, 5, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 2.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 5.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = Mean(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Mean should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 0, "Mean result should be scalar");
  ASSERT_EQ(dest.size, 1, "Mean scalar result should have size 1");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 3.5f, "Mean should be 3.5");

  freeMemory(mem);
}

static void test_mean_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Mean(&ctx, NULL, &dest);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Mean with null tensor should fail");

  freeMemory(mem);
}

static void test_mean_non_float_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  Tensor dest;
  Result r = Mean(&ctx, t, &dest);
  ASSERT_EQ(r, ERR_MEAN_VALUE_NOT_FLOAT, "Mean with integer dtype should fail");

  freeMemory(mem);
}

// Std tests
static void test_std_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {4};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0.0f);
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 2.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;

  Tensor dest;
  Result r = Std(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Std should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 0, "Std result should be scalar");
  ASSERT_EQ(dest.size, 1, "Std scalar result should have size 1");

  // Sample std([1,2,3,4]) = sqrt(5/3) ~= 1.2909944
  f32 got = ((f32 *)dest.values)[0];
  ASSERT(fabsf(got - 1.2909944f) < 1e-5f, "Std should match sample standard deviation");

  freeMemory(mem);
}

static void test_std_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Std(&ctx, NULL, &dest);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Std with null tensor should fail");

  freeMemory(mem);
}

static void test_std_non_float_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  Tensor dest;
  Result r = Std(&ctx, t, &dest);
  ASSERT_EQ(r, ERR_STD_NOT_FLOAT_TYPE, "Std with integer dtype should fail");

  freeMemory(mem);
}

static void test_std_requires_two_or_more_values(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {1};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 2.0f);

  Tensor dest;
  Result r = Std(&ctx, t, &dest);
  ASSERT_EQ(r, ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES,
            "Std should fail when tensor has fewer than 2 values");

  freeMemory(mem);
}

// Log tests
static void test_log_basic(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // Set values: [[1.0, 2.718], [10.0, 100.0]]
  // ln(1) = 0, ln(e) ≈ 1, ln(10) ≈ 2.302, ln(100) ≈ 4.605
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 2.71828f;
  ((f32 *)t->values)[2] = 10.0f;
  ((f32 *)t->values)[3] = 100.0f;

  Tensor dest;
  Result r = Log(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Log should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Log result should preserve shape");
  ASSERT_EQ(dest.shape.dims[0], 2, "Log result dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 2, "Log result dim 1 should be 2");

  f32 *vals = (f32 *)dest.values;
  f32 tolerance = 0.01f;
  ASSERT(fabs(vals[0] - 0.0f) < tolerance, "ln(1) should be ~0");
  ASSERT(fabs(vals[1] - 1.0f) < tolerance, "ln(e) should be ~1");
  ASSERT(fabs(vals[2] - 2.303f) < tolerance, "ln(10) should be ~2.303");
  ASSERT(fabs(vals[3] - 4.605f) < tolerance, "ln(100) should be ~4.605");

  freeMemory(mem);
}

static void test_log_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Log(&ctx, NULL, &dest);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Log with null tensor should fail");

  freeMemory(mem);
}

static void test_log_non_float_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  Tensor dest;
  Result r = Log(&ctx, t, &dest);
  ASSERT_EQ(r, ERR_LOG_VALUE_NOT_FLOAT, "Log with integer dtype should fail");

  freeMemory(mem);
}

// Abs tests
static void test_abs_signed_int(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {4};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 1}, 0);

  ((i8 *)t->values)[0] = -1;
  ((i8 *)t->values)[1] = 2;
  ((i8 *)t->values)[2] = -3;
  ((i8 *)t->values)[3] = 0;

  Tensor dest;
  Result r = Abs(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Abs on signed int tensor should succeed");
  ASSERT_EQ(dest.dtype, I8, "Abs should preserve dtype");
  ASSERT_EQ(dest.shape.numOfDims, 1, "Abs should preserve rank");
  ASSERT_EQ(dest.shape.dims[0], 4, "Abs should preserve shape");

  i8 *vals = (i8 *)dest.values;
  ASSERT_EQ(vals[0], 1, "abs(-1) should be 1");
  ASSERT_EQ(vals[1], 2, "abs(2) should be 2");
  ASSERT_EQ(vals[2], 3, "abs(-3) should be 3");
  ASSERT_EQ(vals[3], 0, "abs(0) should be 0");

  freeMemory(mem);
}

static void test_abs_float(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  ((f32 *)t->values)[0] = -1.5f;
  ((f32 *)t->values)[1] = 2.25f;
  ((f32 *)t->values)[2] = -3.0f;
  ((f32 *)t->values)[3] = 0.0f;

  Tensor dest;
  Result r = Abs(&ctx, t, &dest);
  ASSERT_EQ(r, OK, "Abs on float tensor should succeed");
  ASSERT_EQ(dest.dtype, F32, "Abs should preserve float dtype");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Abs should preserve rank");
  ASSERT_EQ(dest.shape.dims[0], 2, "Abs shape dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 2, "Abs shape dim 1 should be 2");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 1.5f, "abs(-1.5) should be 1.5");
  ASSERT_EQ(vals[1], 2.25f, "abs(2.25) should be 2.25");
  ASSERT_EQ(vals[2], 3.0f, "abs(-3.0) should be 3.0");
  ASSERT_EQ(vals[3], 0.0f, "abs(0.0) should be 0.0");

  freeMemory(mem);
}

static void test_abs_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Abs(&ctx, NULL, &dest);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Abs with null tensor should fail");

  freeMemory(mem);
}

static void test_abs_unsigned_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Tensor *t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 1});
  t->dtype = U8;

  Tensor dest;
  Result r = Abs(&ctx, t, &dest);
  ASSERT_EQ(r, ERR_ABS_VALUE_NOT_SIGNED, "Abs with unsigned dtype should fail");

  freeMemory(mem);
}

// Max tests
static void test_max_dim0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 5, 3], [4, 2, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 5.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 2.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = Max(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "Max dim0 should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Max result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 1, "Max result dim 0 should be 1");
  ASSERT_EQ(dest.shape.dims[1], 3, "Max result dim 1 should be 3");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 4.0f, "Max[0,0] should be 4 (max of 1,4)");
  ASSERT_EQ(vals[1], 5.0f, "Max[0,1] should be 5 (max of 5,2)");
  ASSERT_EQ(vals[2], 6.0f, "Max[0,2] should be 6 (max of 3,6)");

  freeMemory(mem);
}

static void test_max_dim1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 5, 3], [4, 2, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 5.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 2.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = Max(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "Max dim1 should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Max result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 2, "Max result dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "Max result dim 1 should be 1");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 5.0f, "Max[0,0] should be 5 (max of 1,5,3)");
  ASSERT_EQ(vals[1], 6.0f, "Max[1,0] should be 6 (max of 4,2,6)");

  freeMemory(mem);
}

static void test_max_int_type(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  // [[1, 5], [3, 2]]
  ((i8 *)t->values)[0] = 1;
  ((i8 *)t->values)[1] = 5;
  ((i8 *)t->values)[2] = 3;
  ((i8 *)t->values)[3] = 2;

  Tensor dest;
  Result r = Max(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "Max with int type should succeed");

  i8 *vals = (i8 *)dest.values;
  ASSERT_EQ(vals[0], 3, "Max[0,0] should be 3 (max of 1,3)");
  ASSERT_EQ(vals[1], 5, "Max[0,1] should be 5 (max of 5,2)");

  freeMemory(mem);
}

static void test_max_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Max(&ctx, NULL, &dest, 0);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Max with null tensor should fail");

  freeMemory(mem);
}

static void test_max_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  Tensor dest;
  Result r = Max(&ctx, t, &dest, 5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "Max with out of bounds dim should fail");

  freeMemory(mem);
}

static void test_max_non_contiguous(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 5, 3], [4, 2, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 5.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 2.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor transposed;
  Transpose(&ctx, t, &transposed, 0, 1);

  Tensor dest;
  Result r = Max(&ctx, &transposed, &dest, 0);
  ASSERT_EQ(r, OK, "Max on non-contiguous tensor should succeed");

  freeMemory(mem);
}

// ArgMax tests
static void test_argmax_dim0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 5, 3], [4, 2, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 5.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 2.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = ArgMax(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "ArgMax dim0 should succeed");
  ASSERT_EQ(dest.dtype, I64, "ArgMax output dtype should be I64");
  ASSERT_EQ(dest.shape.numOfDims, 2, "ArgMax result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 1, "ArgMax result dim 0 should be 1");
  ASSERT_EQ(dest.shape.dims[1], 3, "ArgMax result dim 1 should be 3");

  i64 *vals = (i64 *)dest.values;
  ASSERT_EQ(vals[0], 1, "ArgMax[0,0] should be 1");
  ASSERT_EQ(vals[1], 0, "ArgMax[0,1] should be 0");
  ASSERT_EQ(vals[2], 1, "ArgMax[0,2] should be 1");

  freeMemory(mem);
}

static void test_argmax_dim1_with_ties(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 4};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  // [[1, 5, 5, 2], [3, 3, 1, 3]]
  ((i8 *)t->values)[0] = 1;
  ((i8 *)t->values)[1] = 5;
  ((i8 *)t->values)[2] = 5;
  ((i8 *)t->values)[3] = 2;
  ((i8 *)t->values)[4] = 3;
  ((i8 *)t->values)[5] = 3;
  ((i8 *)t->values)[6] = 1;
  ((i8 *)t->values)[7] = 3;

  Tensor dest;
  Result r = ArgMax(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "ArgMax dim1 should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "ArgMax result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 2, "ArgMax result dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "ArgMax result dim 1 should be 1");

  i64 *vals = (i64 *)dest.values;
  ASSERT_EQ(vals[0], 1, "ArgMax row 0 should pick first max index");
  ASSERT_EQ(vals[1], 0, "ArgMax row 1 should pick first max index");

  freeMemory(mem);
}

static void test_argmax_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = ArgMax(&ctx, NULL, &dest, 0);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "ArgMax with null tensor should fail");

  freeMemory(mem);
}

static void test_argmax_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  Tensor dest;
  Result r = ArgMax(&ctx, t, &dest, 5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "ArgMax with out of bounds dim should fail");

  freeMemory(mem);
}

static void test_argmax_non_contiguous(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 5, 3], [4, 2, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 5.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 2.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor transposed;
  Transpose(&ctx, t, &transposed, 0, 1);

  Tensor dest;
  Result r = ArgMax(&ctx, &transposed, &dest, 0);
  ASSERT_EQ(r, OK, "ArgMax on non-contiguous tensor should succeed");
  ASSERT_EQ(dest.dtype, I64, "ArgMax output dtype should be I64");

  freeMemory(mem);
}

// MeanDim tests
static void test_meandim_dim0(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 2, 3], [4, 5, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 2.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 5.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = MeanDim(&ctx, t, &dest, 0);
  ASSERT_EQ(r, OK, "MeanDim dim0 should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "MeanDim result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 1, "MeanDim result dim 0 should be 1");
  ASSERT_EQ(dest.shape.dims[1], 3, "MeanDim result dim 1 should be 3");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 2.5f, "MeanDim[0,0] should be 2.5 (mean of 1,4)");
  ASSERT_EQ(vals[1], 3.5f, "MeanDim[0,1] should be 3.5 (mean of 2,5)");
  ASSERT_EQ(vals[2], 4.5f, "MeanDim[0,2] should be 4.5 (mean of 3,6)");

  freeMemory(mem);
}

static void test_meandim_dim1(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  // [[1, 2, 3], [4, 5, 6]]
  ((f32 *)t->values)[0] = 1.0f;
  ((f32 *)t->values)[1] = 2.0f;
  ((f32 *)t->values)[2] = 3.0f;
  ((f32 *)t->values)[3] = 4.0f;
  ((f32 *)t->values)[4] = 5.0f;
  ((f32 *)t->values)[5] = 6.0f;

  Tensor dest;
  Result r = MeanDim(&ctx, t, &dest, 1);
  ASSERT_EQ(r, OK, "MeanDim dim1 should succeed");
  ASSERT_EQ(dest.shape.numOfDims, 2, "MeanDim result should have 2 dimensions");
  ASSERT_EQ(dest.shape.dims[0], 2, "MeanDim result dim 0 should be 2");
  ASSERT_EQ(dest.shape.dims[1], 1, "MeanDim result dim 1 should be 1");

  f32 *vals = (f32 *)dest.values;
  ASSERT_EQ(vals[0], 2.0f, "MeanDim[0,0] should be 2 (mean of 1,2,3)");
  ASSERT_EQ(vals[1], 5.0f, "MeanDim[1,0] should be 5 (mean of 4,5,6)");

  freeMemory(mem);
}

static void test_meandim_non_float_rejected(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Tensor *t = T_Int(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0);

  Tensor dest;
  Result r = MeanDim(&ctx, t, &dest, 0);
  ASSERT_EQ(r, ERR_MEAN_VALUE_NOT_FLOAT, "MeanDim with integer dtype should fail");

  freeMemory(mem);
}

static void test_meandim_dim_out_of_bounds(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 3};
  Tensor *t = T_Float(&ctx, (Dim){.dims = dims, .numOfDims = 2}, 0.0f);

  Tensor dest;
  Result r = MeanDim(&ctx, t, &dest, 5);
  ASSERT_EQ(r, ERR_DIM_MISMATCH, "MeanDim with out of bounds dim should fail");

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
  // View correctness tests (Phase 1)
  test_view_slice_boundary_propagation();
  test_view_nested_slice_correctness();
  test_view_get_tensor_at_on_slice();
  test_view_advanced_indexing_on_slice();
  test_add_in_place_on_slice_view();
  test_view_boundary_deep_copy_transpose();
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
  test_greater_than_basic_same_shape();
  test_greater_or_equal_and_less_or_equal();
  test_less_than_broadcast_row_vector();
  test_comparison_dtype_mismatch();
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
  // IndexWithTensor2d tests
  test_index_with_tensor_2d_basic();
  test_index_with_tensor_2d_3d_source();
  test_index_with_tensor_2d_null_tensor();
  test_index_with_tensor_2d_insufficient_dims();
  test_index_with_tensor_2d_mismatched_indices();
  test_index_with_tensor_2d_out_of_bounds();
  test_index_with_tensor_2d_non_int_indices();
  // Mean tests
  test_mean_basic();
  test_mean_null_tensor();
  test_mean_non_float_rejected();
  // Std tests
  test_std_basic();
  test_std_null_tensor();
  test_std_non_float_rejected();
  test_std_requires_two_or_more_values();
  // Log tests
  test_log_basic();
  test_log_null_tensor();
  test_log_non_float_rejected();
  // Abs tests
  test_abs_signed_int();
  test_abs_float();
  test_abs_null_tensor();
  test_abs_unsigned_rejected();
  // Max tests
  test_max_dim0();
  test_max_dim1();
  test_max_int_type();
  test_max_null_tensor();
  test_max_dim_out_of_bounds();
  test_max_non_contiguous();
  // ArgMax tests
  test_argmax_dim0();
  test_argmax_dim1_with_ties();
  test_argmax_null_tensor();
  test_argmax_dim_out_of_bounds();
  test_argmax_non_contiguous();
  // MeanDim tests
  test_meandim_dim0();
  test_meandim_dim1();
  test_meandim_non_float_rejected();
  test_meandim_dim_out_of_bounds();
}
