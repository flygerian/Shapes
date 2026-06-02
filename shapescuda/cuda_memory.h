#ifndef utils_lib_cuda_memory_h
#define utils_lib_cuda_memory_h

#include "array.h"
#include "memory.h"
#include "olib.h"
#include <stddef.h>

typedef struct CudaMemory {
  Array *blocks;
  size_t allocationPointer;
  i64 allocationCheckpoint;
} CudaMemory;

CudaBlock AllocateOnCuda(CudaMemory *cudaMemory, Memory *hostMemory, size_t size);
void ReleaseCudaBlocks(CudaMemory *cudaMemory);
CudaMemory Make_CudaMemory(Memory *hostMemory);
CudaMemory GetCudaMemoryScratchCheckPoint(CudaMemory *cudaMemory);
void Rewind(CudaMemory *cudaMemory);
void FreeCudaScratchMemory(CudaMemory *cudaMemory);

#endif
