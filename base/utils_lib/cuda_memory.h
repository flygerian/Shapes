#ifndef utils_lib_cuda_memory_h
#define utils_lib_cuda_memory_h

#include "utils_lib/array.h"
#include "utils_lib/memory.h"

typedef struct CudaMemory {
  Array *blocks;
} CudaMemory;

CudaBlock* AllocateOnCuda(CudaMemory cudaMemory, Memory *hostMemory, size_t size);
void ReleaseCudaBlocks(CudaMemory cudaMemory);
CudaMemory Make_CudaMemory(Memory *hostMemory);

#endif
