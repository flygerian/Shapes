#include "memory.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

Memory *initializeMemory() {
  Memory *head;
  head = malloc(sizeof(Memory) + ALLOCATION); // return the top of the heap;
                                              // top of the heap
  assert(head != NULL);

  head->capacity = ALLOCATION;
  head->allocated = 0;

  return head;
}

void *findAvailableSpace(Memory *memory, size_t size) {
  size_t idxToCheck = 0;

  uint8_t *arena = ARENA(memory);

  while (1) {
    blockheader *headerAtIdx = (blockheader *)(arena + idxToCheck);
    assert(headerAtIdx != NULL);

    if (headerAtIdx->free && headerAtIdx->blockSize >= size) {
      headerAtIdx->free = false;
      return (uint8_t *)(headerAtIdx + 1);
    }

    size_t totalBlockSize = sizeof(blockheader) + headerAtIdx->blockSize + sizeof(blockfooter);

    if (memory->allocated <= (idxToCheck + totalBlockSize)) {
      return NULL;
    }

    idxToCheck += totalBlockSize;
  }
}

void *allocate(Memory *memory, size_t size) {
  if (size == 0) {
    return NULL;
  }

  size_t totalBlockSize = sizeof(blockheader) + size + sizeof(blockfooter);

  if (memory->allocated + totalBlockSize > memory->capacity) {
    return findAvailableSpace(memory, size);
  }

  uint8_t *arena = ARENA(memory);
  size_t headerOffset = memory->allocated;

  blockheader *header = (blockheader *)(arena + headerOffset);
  header->free = false;
  header->blockSize = size;

  void *blockToReturn = (uint8_t *)(header + 1);

  blockfooter *footer = (blockfooter *)((uint8_t *)blockToReturn + size);
  footer->headerOffset = headerOffset;

  memory->allocated += totalBlockSize;

  return blockToReturn;
}

void createFreeBlockFromLeftover(uint8_t *arena, blockheader *blockHeader, size_t leftover) {
  uint8_t *blockData = (uint8_t *)(blockHeader + 1);

  blockfooter *footer = (blockfooter *)(blockData + blockHeader->blockSize);
  footer->headerOffset = (uint8_t *)blockHeader - arena;

  blockheader *splitHeader = (blockheader *)((uint8_t *)footer + sizeof(blockfooter));
  splitHeader->blockSize = leftover - sizeof(blockheader) - sizeof(blockfooter);
  splitHeader->free = true;

  blockfooter *splitFooter = (blockfooter *)((uint8_t *)(splitHeader + 1) + splitHeader->blockSize);
  splitFooter->headerOffset = (uint8_t *)splitHeader - arena;
}

void *findSpaceAtEndOfBlock(Memory *memory, blockheader *currentBlockHeader,
                            size_t totalSpaceNeeded) {
  size_t oldSize = currentBlockHeader->blockSize;
  uint8_t *currentBlock = (uint8_t *)(currentBlockHeader + 1);
  uint8_t *arena = ARENA(memory);

  uint8_t *nextBlockPos = currentBlock + oldSize + sizeof(blockfooter);
  if (nextBlockPos >= arena + memory->allocated) {
    return NULL;
  }

  blockheader *nextBlock = (blockheader *)(nextBlockPos);
  if (!nextBlock->free) {
    return NULL;
  }

  size_t mergedCapacity =
      oldSize + sizeof(blockfooter) + sizeof(blockheader) + nextBlock->blockSize;

  if (mergedCapacity < totalSpaceNeeded) {
    return NULL;
  }

  size_t leftover = mergedCapacity - totalSpaceNeeded;
  size_t minBlockSize = sizeof(blockheader) + sizeof(blockfooter) + 1;

  if (leftover > minBlockSize) {
    currentBlockHeader->blockSize = totalSpaceNeeded;
    createFreeBlockFromLeftover(arena, currentBlockHeader, leftover);
  } else {
    currentBlockHeader->blockSize = mergedCapacity;

    blockfooter *footer = (blockfooter *)(currentBlock + currentBlockHeader->blockSize);
    footer->headerOffset = (uint8_t *)currentBlockHeader - arena;
  }

  return currentBlock;
}

void *reallocate(Memory *memory, void *ptr, size_t size) {
  if (ptr != NULL) {
    blockheader *memBlockHeader = BLOCK_HEADER(ptr);

    if (memBlockHeader->free) {
      return NULL;
    }

    if (size == 0) {
      freeAlloc(memory, ptr);
      return NULL;
    }

    if (size <= memBlockHeader->blockSize) {
      return ptr;
    }

    void *endOfBlockSpace = findSpaceAtEndOfBlock(memory, memBlockHeader, size);
    if (endOfBlockSpace != NULL) {
      return endOfBlockSpace;
    }

    void *newSpace = allocate(memory, size);
    memcpy(newSpace, ptr, memBlockHeader->blockSize);
    freeAlloc(memory, ptr);
    return newSpace;
  } else {
    return allocate(memory, size);
  }
}

void coalesceBackwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = (uint8_t *)memBlockHeader - arena;

  if (headerOffset > 0) {
    blockfooter *prevFooter = (blockfooter *)((uint8_t *)memBlockHeader - sizeof(blockfooter));
    blockheader *prevHeader = (blockheader *)(arena + prevFooter->headerOffset);

    if (prevHeader->free) {
      prevHeader->blockSize +=
          sizeof(blockfooter) + sizeof(blockheader) + memBlockHeader->blockSize;

      blockfooter *footer = (blockfooter *)((uint8_t *)(prevHeader + 1) + prevHeader->blockSize);
      footer->headerOffset = (uint8_t *)prevHeader - arena;
    }
  }
}

void freeAlloc(Memory *memory, void *ptr) {
  if (ptr == NULL)
    return;

  blockheader *memBlockHeader = BLOCK_HEADER(ptr);

  memBlockHeader->free = true;

  coalesceBackwards(memory, memBlockHeader);
}

void freeMemory(Memory *memory) {
  free(memory);
}
