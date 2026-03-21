#include "../../shapes.h"
#include "cblas.h"
#include "common.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <termios.h>
#include "memory.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"

void showcase(Context *ctx) {

  // Create a 3x4 tensor
  dim_t dims[] = {3, 4};
  Tensor *t = T_Zeros(ctx, (Dim){.dims = dims, .numOfDims = 2});
  printf("=== Original 3x4 Tensor ===\n");

  // Populate with values: t[i,j] = i*4 + j
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      dim_t idx[] = {i, j};
      AssignValueAt(ctx, t, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)(i * 4 + j)});
    }
  }

  // Print original tensor
  Value v;
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      dim_t idx[] = {i, j};
      GetAt(t, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Transpose: swap dims 0 and 1 -> 4x3
  printf("\n=== Transposed (4x3) ===\n");
  Tensor transposed;
  Transpose(ctx, t, &transposed, (dim_t)0, (dim_t)1);
  printf("Shape: %dx%d\n", transposed.shape.dims[0], transposed.shape.dims[1]);

  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&transposed, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Reshape original to 1D (12 elements)
  printf("\n=== Reshaped to 1D (12 elements) ===\n");
  dim_t flat_dims[] = {12};
  Tensor flat;
  Reshape(ctx, t, &flat, (Dim){.dims = flat_dims, .numOfDims = 1});

  for (u32 i = 0; i < 12; i++) {
    dim_t idx[] = {i};
    GetAt(&flat, (Dim){.dims = idx, .numOfDims = 1}, &v);
    printf("%d ", v.as.u8);
  }
  printf("\n");

  // Reshape original to 2x6
  printf("\n=== Reshaped to 2x6 ===\n");
  dim_t new_dims[] = {2, 6};
  Tensor reshaped;
  Reshape(ctx, t, &reshaped, (Dim){.dims = new_dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 6; j++) {
      dim_t idx[] = {i, j};
      GetAt(&reshaped, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Reshape after transpose (copies data)
  printf("\n=== Reshape Transposed to 1D (copies data) ===\n");
  Tensor trans_flat;
  Reshape(ctx, &transposed, &trans_flat, (Dim){.dims = flat_dims, .numOfDims = 1});

  for (u32 i = 0; i < 12; i++) {
    dim_t idx[] = {i};
    GetAt(&trans_flat, (Dim){.dims = idx, .numOfDims = 1}, &v);
    printf("%d ", v.as.u8);
  }
  printf("\n");

  // Binary operations
  printf("\n=== Binary Operations ===\n");

  // Create two 2x3 tensors
  dim_t op_dims[] = {2, 3};
  Tensor *a = T_Zeros(ctx, (Dim){.dims = op_dims, .numOfDims = 2});
  Tensor *b = T_Zeros(ctx, (Dim){.dims = op_dims, .numOfDims = 2});

  // a = [[1,2,3], [4,5,6]], b = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      AssignValueAt(ctx, a, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)(i * 3 + j + 1)});
      AssignValueAt(ctx, b, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)((i * 3 + j + 1) * 10)});
    }
  }

  printf("Tensor A:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(a, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  printf("\nTensor B:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(b, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Add
  Tensor sum;
  Add(ctx, a, b, &sum);
  printf("\nA + B:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&sum, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Subtract
  Tensor diff;
  Subtract(ctx, b, a, &diff);
  printf("\nB - A:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&diff, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Multiply with scalar broadcast
  printf("\n=== Broadcasting: A * scalar ===\n");
  dim_t scalar_dims[] = {1, 1};
  Tensor *scalar = T_Zeros(ctx, (Dim){.dims = scalar_dims, .numOfDims = 2});
  dim_t scalar_idx[] = {0, 0};
  AssignValueAt(ctx, scalar, (Dim){.dims = scalar_idx, .numOfDims = 2},
                (Value){.dtype = U8, .as.u8 = 5});

  Tensor product;
  Multiply(ctx, a, scalar, &product);
  printf("A * 5:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&product, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Divide with row broadcast
  printf("\n=== Broadcasting: B / row_vector ===\n");
  dim_t row_dims[] = {1, 3};
  Tensor *row = T_Zeros(ctx, (Dim){.dims = row_dims, .numOfDims = 2});
  u8 divisors[] = {10, 10, 10};
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {0, j};
    AssignValueAt(ctx, row, (Dim){.dims = idx, .numOfDims = 2},
                  (Value){.dtype = U8, .as.u8 = divisors[j]});
  }

  Tensor quotient;
  Divide(ctx, b, row, &quotient);
  printf("B / [10,10,10]:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&quotient, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // 2D + 1D broadcast
  printf("\n=== Broadcasting: 2D + 1D ===\n");
  dim_t vec_dims[] = {3};
  Tensor *vec = T_Zeros(ctx, (Dim){.dims = vec_dims, .numOfDims = 1});
  for (u32 j = 0; j < 3; j++) {
    dim_t idx[] = {j};
    AssignValueAt(ctx, vec, (Dim){.dims = idx, .numOfDims = 1},
                  (Value){.dtype = U8, .as.u8 = (u8)(100)});
  }

  Tensor broadcast_sum;
  Add(ctx, a, vec, &broadcast_sum);
  printf("A + [100,100,100]:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      dim_t idx[] = {i, j};
      GetAt(&broadcast_sum, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }

    printf("\n");
  }
}

void tensorExp(Context ctx) {

  dim_t shape[1] = {1};
  Dim tDim = {.dims = shape, .numOfDims = 1};

  Tensor *a = T_Float(&ctx, tDim, 2);
  Tensor *b = T_Float(&ctx, tDim, -3);
  Tensor *c = T_Float(&ctx, tDim, 10);

  Tensor e;
  Multiply(&ctx, a, b, &e);

  Tensor d;
  Add(&ctx, &e, c, &d);

  Tensor *f = T_Float(&ctx, tDim, -2);

  Tensor L;
  Multiply(&ctx, &d, f, &L);


  Value dL_dd;
  GetAt(f, DIM_ZERO, &dL_dd);

  Value dL_df;
  GetAt(&d, DIM_ZERO, &dL_df);
}

double now_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  Memory *mem = initializeArena((size_t)1024 * 1024 * 4096, 1);
  Context ctx = {.memory = mem};

  int M = 4096, N = 4096, K = 4096;

  f32 *h_A = (f32 *)allocate(ctx.memory, (size_t)M * K * sizeof(f32));
  f32 *h_B = (f32 *)allocate(ctx.memory, (size_t)K * N * sizeof(f32));
  f32 *h_C = (f32 *)allocate(ctx.memory, (size_t)M * N * sizeof(f32));

  for (int i = 0; i < M * K; i++) {
    h_A[i] = (float)rand() / RAND_MAX;
  }

  for (int i = 0; i < K * N; i++) {
    h_B[i] = (float)rand() / RAND_MAX;
  }

  float alpha = 1.0f, beta = 0.0f;

  double start = now_ms();
  for (int i = 0; i < 10; i++) {
    runGemm(&ctx, F32, CblasNoTrans, CblasNoTrans, M, N, K, h_A, M, h_B, K, false, h_C, M);
  }
  double elapsed = now_ms() - start;

  printf("OpenBLAS: %.2f ms/iter\n", elapsed / 10);

  // CPU comparison with a naive loop would be unfair, but you can compare
  // against your existing cblas_sgemm timing on the same size

  // freeAlloc(ctx.memory, h_A);
  // freeAlloc(ctx.memory, h_B);
  // freeAlloc(ctx.memory, h_C);
  //
  // freeMemory(ctx.memory);
  return 0;
}
