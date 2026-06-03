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

static inline void Array_AppendCudaBlock(Array *array, CudaBlock *block) {
  PANIC_IF(array->elemSize != sizeof(CudaBlock*), ARRAY_ELEM_SIZE_MISMATCH);
  Array_Append(array, (void *)&block);
}

static inline CudaBlock* Array_CudaBlockIdx(Array *array, size_t idx) {
  return *(CudaBlock **) Array_Idx(array, idx);
}

CudaMemory Make_CudaMemory(Memory *restrict hostMemory) {
  return (CudaMemory){.blocks = MakeDynamicArray(hostMemory, sizeof(CudaBlock*)), .allocationPointer = 0, .allocationCheckpoint = -1}; 
}

void ReleaseCudaBlocks(CudaMemory *restrict cudaMemory) {
  for (RANGE(i, cudaMemory->blocks->size)) {
    CudaBlock *block = Array_CudaBlockIdx(cudaMemory->blocks, i);
    cudaFree(block->ptr);
  }
}

void Rewind(CudaMemory *restrict cudaMemory) {
  PANIC_IF(cudaMemory->allocationCheckpoint < 0, ERR_CUDA_BLOCK_NO_ALLOCATION_CHECKPOINT); 
  cudaMemory->allocationPointer = cudaMemory->allocationCheckpoint;
}

CudaMemory GetCudaMemoryScratchCheckPoint(CudaMemory *restrict cudaMemory) {
  size_t currentAllocPoint = cudaMemory->blocks->size - 1;
  return (CudaMemory){.allocationPointer = currentAllocPoint, .allocationCheckpoint = currentAllocPoint, .blocks = cudaMemory->blocks};
}

void FreeCudaScratchMemory(CudaMemory *restrict cudaMemory) {
  for(RANGE_FROM(cudaMemory->allocationPointer, cudaMemory->blocks->size, i)) {
    CudaBlock *block = Array_CudaBlockIdx(cudaMemory->blocks, i);
    cudaFree(block->ptr);
  }
}

CudaBlock AllocateOnCuda(CudaMemory *restrict cudaMemory, Memory *restrict hostMemory, size_t size) {
  PANIC_IF(cudaMemory->blocks == NULL, ERR_NULL_PTR);
  void *locationOnDestCtx = NULL;
  size_t roundedSize = roundCudaAllocationSize(size);

  if (cudaMemory->allocationPointer > 0 && cudaMemory->allocationPointer < cudaMemory->blocks->size - 1) {
    // start allocation from the scratch pointer
    CudaBlock *nextBlock = Array_CudaBlockIdx(cudaMemory->blocks, cudaMemory->allocationPointer + 1); 
    PANIC_IF(nextBlock->size != roundedSize, ERR_CUDA_BLOCK_MISMATCH);
    // if (cudaMemory->allocationPointer == 6272) {
    //   printf("block at 6272: %zu, roundedSize: %zu \n", nextBlock->size, roundedSize);
    // }
    cudaMemory->allocationPointer += 1;

    return *nextBlock;
  } 

  cudaError_t cudaResult = cudaMalloc(&locationOnDestCtx, roundedSize);
  PANIC_IF(cudaResult != cudaSuccess, cudaGetErrorString(cudaResult)); 

  CudaBlock *newBlock = allocate(hostMemory, sizeof(CudaBlock)); 
  *newBlock = (CudaBlock){.ptr = locationOnDestCtx, .size = roundedSize};

  Array_AppendCudaBlock((Array*) cudaMemory->blocks, newBlock);
  cudaMemory->allocationPointer += 1;

  return *newBlock;
}
