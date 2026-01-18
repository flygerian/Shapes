#include "../../tensor/tensor.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};

  // Create a 3x4 tensor
  u32 dims[] = {3, 4};
  Tensor t = T_Zeros(&ctx, (Dim){.dims = dims, .numOfDims = 2});
  printf("Created 3x4 tensor (dtype: U8)\n");

  // Assign some values
  u32 idx0[] = {0, 0};
  u32 idx1[] = {1, 2};
  u32 idx2[] = {2, 3};

  AssignValueAt(&ctx, &t, (Dim){.dims = idx0, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 10});
  AssignValueAt(&ctx, &t, (Dim){.dims = idx1, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 20});
  AssignValueAt(&ctx, &t, (Dim){.dims = idx2, .numOfDims = 2}, (Value){.dtype = U8, .as.u8 = 30});
  printf("Assigned values: [0,0]=10, [1,2]=20, [2,3]=30\n");

  // Read values back
  Value v;
  GetAt(&t, (Dim){.dims = idx0, .numOfDims = 2}, &v);
  printf("GetAt [0,0] = %d\n", v.as.u8);

  GetAt(&t, (Dim){.dims = idx1, .numOfDims = 2}, &v);
  printf("GetAt [1,2] = %d\n", v.as.u8);

  GetAt(&t, (Dim){.dims = idx2, .numOfDims = 2}, &v);
  printf("GetAt [2,3] = %d\n", v.as.u8);

  // Print full tensor
  printf("\nFull tensor:\n");
  for (u32 i = 0; i < 3; i++) {
    for (u32 j = 0; j < 4; j++) {
      u32 idx[] = {i, j};
      GetAt(&t, (Dim){.dims = idx, .numOfDims = 2}, &v);
      printf("%3d ", v.as.u8);
    }
    printf("\n");
  }

  freeMemory(mem);
  return 0;
}
