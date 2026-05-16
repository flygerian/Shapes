#include "result/result.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"
#include <stddef.h>
#include "cuda_memory.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

static size_t roundCudaAllocationSize(size_t size) {
  const size_t alignment = 256;
  if (size == 0) {
    return alignment;
  }

  size_t remainder = size % alignment;
  if (remainder == 0) {
    return size;
  }

  return size + (alignment - remainder);
}

static inline void Array_AppendCudaBlock(Array *array, CudaBlock *block) {
  PANIC_IF(array->elemSize != sizeof(CudaBlock*), ARRAY_ELEM_SIZE_MISMATCH);
  Array_Append(array, (void *)&block);
}

static inline CudaBlock* Array_CudaBlockIdx(Array *array, size_t idx) {
  return *(CudaBlock **) Array_Idx(array, idx);
}

CudaMemory Make_CudaMemory(Memory *hostMemory) {
  return (CudaMemory){.blocks = MakeDynamicArray(hostMemory, sizeof(CudaBlock*))}; 
}

void ReleaseCudaBlocks(CudaMemory cudaMemory) {
  for (RANGE(i, cudaMemory.blocks->size)) {
    CudaBlock *block = Array_CudaBlockIdx(cudaMemory.blocks, i);
    cudaFree(block->ptr);
  }
}

CudaBlock* AllocateOnCuda(CudaMemory cudaMemory, Memory *hostMemory, size_t size) {
  PANIC_IF(cudaMemory.blocks == NULL, ERR_NULL_PTR);
  void *locationOnDestCtx = NULL;
  size_t roundedSize = roundCudaAllocationSize(size);

  cudaError_t cudaResult = cudaMalloc(&locationOnDestCtx, roundedSize);
  PANIC_IF(cudaResult != cudaSuccess, ALLOCATION_FAILED); 

  CudaBlock *newBlock = allocate(hostMemory, sizeof(CudaBlock)); 
  *newBlock = (CudaBlock){.ptr = locationOnDestCtx, .size = size};

  Array_AppendCudaBlock((Array*) cudaMemory.blocks, newBlock);
  return newBlock;
}
