#include "../../tensor/tensor.h"
#include "common.h"
#include <stdbool.h>
#include <stdio.h>
#include <termios.h>
#include "memory.h"
#include "visual/visual.h"

void showcase(Context *ctx) {

  // Create a 3x4 tensor
  u32 dims[] = {3, 4};
  Tensor *t = T_Zeros(ctx, (Dim){.dims = dims, .numOfDims = 2});
  printf("=== Original 3x4 Tensor ===\n");

  // Populate with values: t[i,j] = i*4 + j
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      AssignValueAt(ctx, t, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)(i * 4 + j)});
    }
  }

  // Print original tensor
  Value v;
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
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
      u32 idx[] = {i, j};
      GetAt(&transposed, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Reshape original to 1D (12 elements)
  printf("\n=== Reshaped to 1D (12 elements) ===\n");
  u32 flat_dims[] = {12};
  Tensor flat;
  Reshape(ctx, t, &flat, (Dim){.dims = flat_dims, .numOfDims = 1});

  for (u32 i = 0; i < 12; i++) {
    u32 idx[] = {i};
    GetAt(&flat, (Dim){.dims = idx, .numOfDims = 1}, &v);
    printf("%d ", v.as.u8);
  }
  printf("\n");

  // Reshape original to 2x6
  printf("\n=== Reshaped to 2x6 ===\n");
  u32 new_dims[] = {2, 6};
  Tensor reshaped;
  Reshape(ctx, t, &reshaped, (Dim){.dims = new_dims, .numOfDims = 2});

  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 6; j++) {
      u32 idx[] = {i, j};
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
    u32 idx[] = {i};
    GetAt(&trans_flat, (Dim){.dims = idx, .numOfDims = 1}, &v);
    printf("%d ", v.as.u8);
  }
  printf("\n");

  // Binary operations
  printf("\n=== Binary Operations ===\n");

  // Create two 2x3 tensors
  u32 op_dims[] = {2, 3};
  Tensor *a = T_Zeros(ctx, (Dim){.dims = op_dims, .numOfDims = 2});
  Tensor *b = T_Zeros(ctx, (Dim){.dims = op_dims, .numOfDims = 2});

  // a = [[1,2,3], [4,5,6]], b = [[10,20,30], [40,50,60]]
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      AssignValueAt(ctx, a, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)(i * 3 + j + 1)});
      AssignValueAt(ctx, b, (Dim){.dims = idx, .numOfDims = 2},
                    (Value){.dtype = U8, .as.u8 = (u8)((i * 3 + j + 1) * 10)});
    }
  }

  printf("Tensor A:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      GetAt(a, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  printf("\nTensor B:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
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
      u32 idx[] = {i, j};
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
      u32 idx[] = {i, j};
      GetAt(&diff, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Multiply with scalar broadcast
  printf("\n=== Broadcasting: A * scalar ===\n");
  u32 scalar_dims[] = {1, 1};
  Tensor *scalar = T_Zeros(ctx, (Dim){.dims = scalar_dims, .numOfDims = 2});
  u32 scalar_idx[] = {0, 0};
  AssignValueAt(ctx, scalar, (Dim){.dims = scalar_idx, .numOfDims = 2},
                (Value){.dtype = U8, .as.u8 = 5});

  Tensor product;
  Multiply(ctx, a, scalar, &product);
  printf("A * 5:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      GetAt(&product, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Divide with row broadcast
  printf("\n=== Broadcasting: B / row_vector ===\n");
  u32 row_dims[] = {1, 3};
  Tensor *row = T_Zeros(ctx, (Dim){.dims = row_dims, .numOfDims = 2});
  u8 divisors[] = {10, 10, 10};
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {0, j};
    AssignValueAt(ctx, row, (Dim){.dims = idx, .numOfDims = 2},
                  (Value){.dtype = U8, .as.u8 = divisors[j]});
  }

  Tensor quotient;
  Divide(ctx, b, row, &quotient);
  printf("B / [10,10,10]:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      GetAt(&quotient, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // 2D + 1D broadcast
  printf("\n=== Broadcasting: 2D + 1D ===\n");
  u32 vec_dims[] = {3};
  Tensor *vec = T_Zeros(ctx, (Dim){.dims = vec_dims, .numOfDims = 1});
  for (u32 j = 0; j < 3; j++) {
    u32 idx[] = {j};
    AssignValueAt(ctx, vec, (Dim){.dims = idx, .numOfDims = 1},
                  (Value){.dtype = U8, .as.u8 = (u8)(100)});
  }

  Tensor broadcast_sum;
  Add(ctx, a, vec, &broadcast_sum);
  printf("A + [100,100,100]:\n");
  for (u32 i = 0; i < 2; i++) {
    for (u32 j = 0; j < 3; j++) {
      u32 idx[] = {i, j};
      GetAt(&broadcast_sum, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }

    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem, .grad = true, .screenConfig = allocate(mem, sizeof(ScreenConfig))};
  ctx.screenConfig->orig_termios = allocate(mem, sizeof(struct termios));

  EnableRawMode(&ctx);

  dim_t shape[1] = {1};

  Dim tDim = {.dims = shape, .numOfDims = 1};

  Tensor *a = T_Int(&ctx, tDim, 2);
  Tensor *b = T_Int(&ctx, tDim, -3);
  Tensor *c = T_Int(&ctx, tDim, 10);

  Tensor e;
  Multiply(&ctx, a, b, &e);

  Tensor d;

  Add(&ctx, &e, c, &d);

  VisualizeOps(&ctx, &d);

  DisableRawMode(&ctx);
  freeMemory(ctx.memory);
  return 0;
}
