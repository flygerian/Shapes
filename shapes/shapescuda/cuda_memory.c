#include "result.h"
#include "array.h"
#include "memory.h"
#include <stddef.h>
#include "shapescuda.h"
#include <stdio.h>
#include "cuda_runtime.h"

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

static inline void Array_AppendCudaBlock(olib_Array *array, shapescuda_Block *block) {
  PANIC_IF(array->elemSize != sizeof(shapescuda_Block*), ARRAY_ELEM_SIZE_MISMATCH);
  olib_ArrayAppend(array, (void *)&block);
}

static inline shapescuda_Block* Array_CudaBlockIdx(olib_Array *array, size_t idx) {
  return *(shapescuda_Block **) olib_ArrayIdx(array, idx);
}

shapescuda_Memory shapescuda_Make_Memory(olib_Memory *restrict hostMemory) {
  return (shapescuda_Memory){.blocks = olib_MakeDynamicArray(hostMemory, sizeof(shapescuda_Block*)), .allocationPointer = 0, .allocationCheckpoint = -1}; 
}

void  shapescuda_ReleaseBlocks(shapescuda_Memory *restrict cudaMemory) {
  for (RANGE(i, cudaMemory->blocks->size)) {
    shapescuda_Block *block = Array_CudaBlockIdx(cudaMemory->blocks, i);
    cudaFree(block->ptr);
  }
}

void shapescuda_RewindMemory(shapescuda_Memory *restrict cudaMemory) {
  PANIC_IF(cudaMemory->allocationCheckpoint < 0, ERR_CUDA_BLOCK_NO_ALLOCATION_CHECKPOINT); 
  cudaMemory->allocationPointer = cudaMemory->allocationCheckpoint;
}

shapescuda_Memory  shapescuda_GetMemoryScratchCheckPoint(shapescuda_Memory *restrict cudaMemory) {
  size_t currentAllocPoint = cudaMemory->blocks->size - 1;
  return (shapescuda_Memory){.allocationPointer = currentAllocPoint, .allocationCheckpoint = currentAllocPoint, .blocks = cudaMemory->blocks};
}

void shapescuda_FreeScratchMemory(shapescuda_Memory *restrict cudaMemory) {
  for(RANGE_FROM(cudaMemory->allocationPointer, cudaMemory->blocks->size, i)) {
    shapescuda_Block *block = Array_CudaBlockIdx(cudaMemory->blocks, i);
    cudaFree(block->ptr);
  }
}

shapescuda_Block shapescuda_Allocate(shapescuda_Memory *restrict cudaMemory, olib_Memory *restrict hostMemory, size_t size) {
  PANIC_IF(cudaMemory->blocks == NULL, ERR_NULL_PTR);
  void *locationOnDestCtx = NULL;
  size_t roundedSize = roundCudaAllocationSize(size);

  if (cudaMemory->allocationPointer > 0 && cudaMemory->allocationPointer < cudaMemory->blocks->size - 1) {
    // start allocation from the scratch pointer
    shapescuda_Block *nextBlock = Array_CudaBlockIdx(cudaMemory->blocks, cudaMemory->allocationPointer + 1); 
    PANIC_IF(nextBlock->size != roundedSize, ERR_CUDA_BLOCK_MISMATCH);
    // if (cudaMemory->allocationPointer == 6272) {
    //   printf("block at 6272: %zu, roundedSize: %zu \n", nextBlock->size, roundedSize);
    // }
    cudaMemory->allocationPointer += 1;

    return *nextBlock;
  } 

  cudaError_t cudaResult = cudaMalloc(&locationOnDestCtx, roundedSize);
  PANIC_IF(cudaResult != cudaSuccess, cudaGetErrorString(cudaResult)); 

  shapescuda_Block *newBlock = olib_Allocate(hostMemory, sizeof(shapescuda_Block)); 
  *newBlock = (shapescuda_Block){.ptr = locationOnDestCtx, .size = roundedSize};

  Array_AppendCudaBlock((olib_Array*) cudaMemory->blocks, newBlock);
  cudaMemory->allocationPointer += 1;

  return *newBlock;
}
