#include "test.h"
#include "common.h"
#include "tensor/tensor.h"
#include "tensor/value.h"
#include "memory.h"
#include <string.h>

// --- Valid cast tests ---

static void test_cast_i8_to_i16(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 5);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I16);
  ASSERT_EQ(r, OK, "Cast I8 -> I16 should return OK");
  ASSERT_EQ(dest.dtype, I16, "Cast I8 -> I16 dest dtype should be I16");
  ASSERT_EQ(dest.size, 3, "Cast I8 -> I16 should preserve size");
  ASSERT_EQ(dest.shape.numOfDims, 1, "Cast I8 -> I16 should preserve ndims");
  ASSERT_EQ(dest.shape.dims[0], 3, "Cast I8 -> I16 should preserve dim 0");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    i16 val = ((i16 *)dest.values)[i];
    ASSERT_EQ(val, 5, "Cast I8 -> I16 element should be 5");
  }

  freeMemory(mem);
}

static void test_cast_i8_to_i32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, -3);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I32);
  ASSERT_EQ(r, OK, "Cast I8 -> I32 should return OK");
  ASSERT_EQ(dest.dtype, I32, "Cast I8 -> I32 dest dtype should be I32");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    i32 val = ((i32 *)dest.values)[i];
    ASSERT_EQ(val, -3, "Cast I8 -> I32 element should be -3");
  }

  freeMemory(mem);
}

static void test_cast_i8_to_i64(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 7);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I64);
  ASSERT_EQ(r, OK, "Cast I8 -> I64 should return OK");
  ASSERT_EQ(dest.dtype, I64, "Cast I8 -> I64 dest dtype should be I64");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    i64 val = ((i64 *)dest.values)[i];
    ASSERT_EQ(val, 7, "Cast I8 -> I64 element should be 7");
  }

  freeMemory(mem);
}

static void test_cast_i16_to_i32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  // Manually set I16 values
  src->dtype = I16;
  size_t bytes = getBytesForDtype(I16) * src->size;
  src->values = allocate(mem, bytes);
  ((i16 *)src->values)[0] = 1000;
  ((i16 *)src->values)[1] = -500;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I32);
  ASSERT_EQ(r, OK, "Cast I16 -> I32 should return OK");
  ASSERT_EQ(((i32 *)dest.values)[0], 1000, "Cast I16 -> I32 element 0 should be 1000");
  ASSERT_EQ(((i32 *)dest.values)[1], -500, "Cast I16 -> I32 element 1 should be -500");

  freeMemory(mem);
}

static void test_cast_f32_to_f64(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2, 2};
  Dim shape = {.dims = dims, .numOfDims = 2};
  Tensor *src = T_Float(&ctx, shape, 3.14f);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F64);
  ASSERT_EQ(r, OK, "Cast F32 -> F64 should return OK");
  ASSERT_EQ(dest.dtype, F64, "Cast F32 -> F64 dest dtype should be F64");
  ASSERT_EQ(dest.size, 4, "Cast F32 -> F64 should preserve size");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Cast F32 -> F64 should preserve ndims");
  ASSERT_EQ(dest.shape.dims[0], 2, "Cast F32 -> F64 should preserve dim 0");
  ASSERT_EQ(dest.shape.dims[1], 2, "Cast F32 -> F64 should preserve dim 1");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    f64 val = ((f64 *)dest.values)[i];
    ASSERT(val > 3.13 && val < 3.15, "Cast F32 -> F64 element should be ~3.14");
  }

  freeMemory(mem);
}

static void test_cast_u8_to_u16(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  // Manually set U8 values
  src->dtype = U8;
  size_t bytes = getBytesForDtype(U8) * src->size;
  src->values = allocate(mem, bytes);
  ((u8 *)src->values)[0] = 0;
  ((u8 *)src->values)[1] = 128;
  ((u8 *)src->values)[2] = 255;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, U16);
  ASSERT_EQ(r, OK, "Cast U8 -> U16 should return OK");
  ASSERT_EQ(dest.dtype, U16, "Cast U8 -> U16 dest dtype should be U16");
  ASSERT_EQ(((u16 *)dest.values)[0], 0, "Cast U8 -> U16 element 0 should be 0");
  ASSERT_EQ(((u16 *)dest.values)[1], 128, "Cast U8 -> U16 element 1 should be 128");
  ASSERT_EQ(((u16 *)dest.values)[2], 255, "Cast U8 -> U16 element 2 should be 255");

  freeMemory(mem);
}

static void test_cast_u8_to_u32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  src->dtype = U8;
  size_t bytes = getBytesForDtype(U8) * src->size;
  src->values = allocate(mem, bytes);
  ((u8 *)src->values)[0] = 42;
  ((u8 *)src->values)[1] = 200;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, U32);
  ASSERT_EQ(r, OK, "Cast U8 -> U32 should return OK");
  ASSERT_EQ(((u32 *)dest.values)[0], 42, "Cast U8 -> U32 element 0 should be 42");
  ASSERT_EQ(((u32 *)dest.values)[1], 200, "Cast U8 -> U32 element 1 should be 200");

  freeMemory(mem);
}

static void test_cast_i8_to_f32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, -2);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F32);
  ASSERT_EQ(r, OK, "Cast I8 -> F32 should return OK (signed int to float is safe)");
  ASSERT_EQ(dest.dtype, F32, "Cast I8 -> F32 dest dtype should be F32");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    f32 val = ((f32 *)dest.values)[i];
    ASSERT(val > -2.01f && val < -1.99f, "Cast I8 -> F32 element should be -2.0");
  }

  freeMemory(mem);
}

static void test_cast_i32_to_f64(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  src->dtype = I32;
  size_t bytes = getBytesForDtype(I32) * src->size;
  src->values = allocate(mem, bytes);
  ((i32 *)src->values)[0] = 100000;
  ((i32 *)src->values)[1] = -99999;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F64);
  ASSERT_EQ(r, OK, "Cast I32 -> F64 should return OK");
  ASSERT(((f64 *)dest.values)[0] > 99999.0 && ((f64 *)dest.values)[0] < 100001.0,
         "Cast I32 -> F64 element 0 should be 100000.0");
  ASSERT(((f64 *)dest.values)[1] > -100000.0 && ((f64 *)dest.values)[1] < -99998.0,
         "Cast I32 -> F64 element 1 should be -99999.0");

  freeMemory(mem);
}

static void test_cast_same_dtype_clones(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Float(&ctx, shape, 2.5f);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F32);
  ASSERT_EQ(r, OK, "Cast same dtype should return OK (clone)");
  ASSERT_EQ(dest.dtype, F32, "Cast same dtype should preserve dtype");
  ASSERT(dest.values != src->values, "Cast same dtype should produce a new allocation");

  for (tensor_size_t i = 0; i < dest.size; i++) {
    f32 val = ((f32 *)dest.values)[i];
    ASSERT(val > 2.49f && val < 2.51f, "Cast same dtype element should be 2.5");
  }

  freeMemory(mem);
}

// --- Invalid cast tests ---

static void test_cast_f32_to_i32(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Float(&ctx, shape, 0.0f);
  ((f32 *)src->values)[0] = 1.5f;
  ((f32 *)src->values)[1] = -3.9f;
  ((f32 *)src->values)[2] = 42.0f;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I32);
  ASSERT_EQ(r, OK, "Cast F32 -> I32 should return OK");
  ASSERT_EQ(dest.dtype, I32, "Cast F32 -> I32 dest dtype should be I32");
  ASSERT_EQ(((i32 *)dest.values)[0], 1, "Cast F32 1.5 -> I32 should truncate to 1");
  ASSERT_EQ(((i32 *)dest.values)[1], -3, "Cast F32 -3.9 -> I32 should truncate to -3");
  ASSERT_EQ(((i32 *)dest.values)[2], 42, "Cast F32 42.0 -> I32 should be 42");

  freeMemory(mem);
}

static void test_cast_f64_to_i64(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Float(&ctx, shape, 0.0f);
  src->dtype = F64;
  size_t bytes = getBytesForDtype(F64) * src->size;
  src->values = allocate(mem, bytes);
  ((f64 *)src->values)[0] = 99.9;
  ((f64 *)src->values)[1] = -50.1;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I64);
  ASSERT_EQ(r, OK, "Cast F64 -> I64 should return OK");
  ASSERT_EQ(((i64 *)dest.values)[0], 99, "Cast F64 99.9 -> I64 should truncate to 99");
  ASSERT_EQ(((i64 *)dest.values)[1], -50, "Cast F64 -50.1 -> I64 should truncate to -50");

  freeMemory(mem);
}

static void test_cast_f32_to_u32_sign_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Float(&ctx, shape, 1.0f);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, U32);
  ASSERT_EQ(r, ERR_SIGN_MISMATCH_CAST,
            "Cast F32 -> U32 should return ERR_SIGN_MISMATCH_CAST (float to unsigned)");

  freeMemory(mem);
}

static void test_cast_f64_to_f32_truncating(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Float(&ctx, shape, 0.0f);
  src->dtype = F64;
  size_t bytes = getBytesForDtype(F64) * src->size;
  src->values = allocate(mem, bytes);
  ((f64 *)src->values)[0] = 1.0;
  ((f64 *)src->values)[1] = 2.0;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F32);
  ASSERT_EQ(r, ERR_TRUNCATING_CAST,
            "Cast F64 -> F32 should return ERR_TRUNCATING_CAST (narrowing)");

  freeMemory(mem);
}

static void test_cast_i32_to_i16_truncating(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  src->dtype = I32;
  size_t bytes = getBytesForDtype(I32) * src->size;
  src->values = allocate(mem, bytes);
  ((i32 *)src->values)[0] = 100;
  ((i32 *)src->values)[1] = 200;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I16);
  ASSERT_EQ(r, ERR_TRUNCATING_CAST,
            "Cast I32 -> I16 should return ERR_TRUNCATING_CAST (narrowing)");

  freeMemory(mem);
}

static void test_cast_i8_to_u8_sign_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 5);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, U8);
  ASSERT_EQ(r, ERR_SIGN_MISMATCH_CAST,
            "Cast I8 -> U8 should return ERR_SIGN_MISMATCH_CAST (signed to unsigned)");

  freeMemory(mem);
}

static void test_cast_u16_to_i16_sign_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  src->dtype = U16;
  size_t bytes = getBytesForDtype(U16) * src->size;
  src->values = allocate(mem, bytes);
  ((u16 *)src->values)[0] = 10;
  ((u16 *)src->values)[1] = 20;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I16);
  ASSERT_EQ(r, ERR_SIGN_MISMATCH_CAST,
            "Cast U16 -> I16 should return ERR_SIGN_MISMATCH_CAST (unsigned to signed)");

  freeMemory(mem);
}

static void test_cast_u8_to_f32_sign_mismatch(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {2};
  Dim shape = {.dims = dims, .numOfDims = 1};
  Tensor *src = T_Int(&ctx, shape, 0);
  src->dtype = U8;
  size_t bytes = getBytesForDtype(U8) * src->size;
  src->values = allocate(mem, bytes);
  ((u8 *)src->values)[0] = 1;
  ((u8 *)src->values)[1] = 2;

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, F32);
  ASSERT_EQ(r, ERR_SIGN_MISMATCH_CAST,
            "Cast U8 -> F32 should return ERR_SIGN_MISMATCH_CAST (unsigned to float)");

  freeMemory(mem);
}

static void test_cast_null_tensor(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  Tensor dest;
  Result r = Cast(&ctx, NULL, &dest, F32);
  ASSERT_EQ(r, ERR_NULL_TENSOR_PROVIDED, "Cast NULL tensor should return ERR_NULL_TENSOR_PROVIDED");

  freeMemory(mem);
}

static void test_cast_preserves_2d_shape(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  dim_t dims[] = {3, 4};
  Dim shape = {.dims = dims, .numOfDims = 2};
  Tensor *src = T_Int(&ctx, shape, 1);

  Tensor dest;
  Result r = Cast(&ctx, src, &dest, I16);
  ASSERT_EQ(r, OK, "Cast 2D I8 -> I16 should return OK");
  ASSERT_EQ(dest.shape.numOfDims, 2, "Cast 2D should preserve ndims");
  ASSERT_EQ(dest.shape.dims[0], 3, "Cast 2D should preserve dim 0");
  ASSERT_EQ(dest.shape.dims[1], 4, "Cast 2D should preserve dim 1");
  ASSERT_EQ(dest.size, 12, "Cast 2D should preserve total size");

  freeMemory(mem);
}

void run_cast_tests(void) {
  // Valid casts
  test_cast_i8_to_i16();
  test_cast_i8_to_i32();
  test_cast_i8_to_i64();
  test_cast_i16_to_i32();
  test_cast_f32_to_f64();
  test_cast_u8_to_u16();
  test_cast_u8_to_u32();
  test_cast_i8_to_f32();
  test_cast_i32_to_f64();
  test_cast_same_dtype_clones();
  test_cast_preserves_2d_shape();

  // Float <-> signed int casts
  test_cast_f32_to_i32();
  test_cast_f64_to_i64();

  // Invalid casts
  test_cast_f32_to_u32_sign_mismatch();
  test_cast_f64_to_f32_truncating();
  test_cast_i32_to_i16_truncating();
  test_cast_i8_to_u8_sign_mismatch();
  test_cast_u16_to_i16_sign_mismatch();
  test_cast_u8_to_f32_sign_mismatch();
  test_cast_null_tensor();
}
