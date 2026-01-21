#include "../../tensor/tensor.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create a 3x4 tensor
  u32 dims[] = {3, 4};
  Tensor t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  printf("=== Original 3x4 Tensor ===\n");

  // Populate with values: t[i,j] = i*4 + j
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      AssignValueAt(&ctx, &t, (Dim){.dims = idx, .numOfDims = 2}, 
                    (Value){.dtype = U8, .as.u8 = (u8)(i * 4 + j)});
    }
  }

  // Print original tensor
  Value v;
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      GetAt(&t, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  // Transpose: swap dims 0 and 1 -> 4x3
  printf("\n=== Transposed (4x3) ===\n");
  Tensor transposed;
  Transpose(&ctx, &t, &transposed, (dim_t)0, (dim_t)1);
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
  Reshape(&ctx, &t, &flat, (Dim){.dims = flat_dims, .numOfDims = 1});

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
  Reshape(&ctx, &t, &reshaped, (Dim){.dims = new_dims, .numOfDims = 2});

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
  Reshape(&ctx, &transposed, &trans_flat, (Dim){.dims = flat_dims, .numOfDims = 1});

  for (u32 i = 0; i < 12; i++) {
    u32 idx[] = {i};
    GetAt(&trans_flat, (Dim){.dims = idx, .numOfDims = 1}, &v);
    printf("%d ", v.as.u8);
  }
  printf("\n");

  freeMemory(mem);
  return 0;
}
