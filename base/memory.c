#include "memory.h"
#include "result/result.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


Memory *initializeArena(size_t arenaSize, size_t minBlockSize) {
  Memory *head;
  head = malloc(sizeof(Memory) + arenaSize); // return the top of the heap;
                                             // top of the heap
  assert(head != NULL);

  head->capacity = arenaSize;
  head->allocated = 0;
  head->numBlocks = 0;
  head->numFreeBlocks = 0;
  head->minBlockSize = minBlockSize;
  head->freeHeadOffset = INVALID_OFFSET;

  return head;
}

Memory *initializeMemory() {
  return initializeArena((size_t)DEFAULT_ALLOCATION, 1);
}


// Writes a block header and footer for a payload of `size` bytes at `header`.
// Returns header on success, NULL if the block would exceed arena capacity.
void *adjustBlock(Memory *memory, blockheader *header, size_t size) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena, header);

  if (headerOffset + TOTAL_BLOCK_SIZE(size) > memory->capacity) {
    return NULL;
  }

  header->blockSize = size;
  header->nextFreeOffset = INVALID_OFFSET;
  header->prevFreeOffset = INVALID_OFFSET;

  blockfooter *footer = BLOCK_FOOTER(header);
  footer->headerOffset = headerOffset;

  return header;
}

static void removeFreeBlock(Memory *memory, blockheader *header) {
  size_t next = header->nextFreeOffset;
  size_t prev = header->prevFreeOffset;

  if (prev != INVALID_OFFSET) {
    HEADER_AT(memory, prev)->nextFreeOffset = next;
  } else {
    memory->freeHeadOffset = next;
  }

  if (next != INVALID_OFFSET) {
    HEADER_AT(memory, next)->prevFreeOffset = prev;
  }

  header->nextFreeOffset = INVALID_OFFSET;
  header->prevFreeOffset = INVALID_OFFSET;
}

static void insertFreeBlock(Memory *memory, blockheader *header) {
  size_t offset = HEADER_OFFSET(memory, header);
  header->nextFreeOffset = INVALID_OFFSET;
  header->prevFreeOffset = INVALID_OFFSET;

  if (memory->freeHeadOffset == INVALID_OFFSET) {
    memory->freeHeadOffset = offset;
    return;
  }

  size_t currentHeadOffset = memory->freeHeadOffset;
  blockheader *currentHead = HEADER_AT(memory, currentHeadOffset);
  header->nextFreeOffset = currentHeadOffset;
  currentHead->prevFreeOffset = offset;
  memory->freeHeadOffset = offset;
}


// Splits a block into two: resizes header to aSize, writes a new block of bSize
// immediately after, and increments numBlocks. Does NOT touch memory->allocated
// since the memory is already accounted for (either freshly grown or reused).
blockheader *splitBlock(Memory *memory, blockheader *header, size_t aSize, size_t bSize) {
  uint8_t *arena = ARENA(memory);
  adjustBlock(memory, header, aSize);
  size_t nextOffset = BLOCK_HEADER_OFFSET(arena, header) + NEXT_BLOCK_OFFSET(header);
  blockheader *remainder = (blockheader *)(arena + nextOffset);
  adjustBlock(memory, remainder, bSize);
  memory->numBlocks += 1;
  return remainder;
}

void *findAvailableSpace(Memory *memory, size_t size) {
  if (memory->numFreeBlocks == 0 || memory->freeHeadOffset == INVALID_OFFSET) {
    return NULL;
  }

  size_t currentOffset = memory->freeHeadOffset;
  while (currentOffset != INVALID_OFFSET) {
    blockheader *header = HEADER_AT(memory, currentOffset);
    currentOffset = header->nextFreeOffset;

    if (header->blockSize < size) {
      continue;
    }

    removeFreeBlock(memory, header);
    header->free = false;
    memory->numFreeBlocks -= 1;

    size_t currentBlockSize = header->blockSize;
    if (currentBlockSize - size >= TOTAL_BLOCK_SIZE(memory->minBlockSize)) {
      size_t leftover = currentBlockSize - size - sizeof(blockheader) - sizeof(blockfooter);
      blockheader *remainder = splitBlock(memory, header, size, leftover);
      remainder->free = true;
      insertFreeBlock(memory, remainder);
      memory->numFreeBlocks += 1;
    }

    return (uint8_t *)(header + 1);
  }

  return NULL;
}

void *allocate(Memory *memory, size_t size) {
  PANIC_IF(size == 0, ALLOCATING_ZERO); 

  void *reused = findAvailableSpace(memory, size);
  if (reused != NULL) {
    return reused;
  }

  size_t totalBlockSize = TOTAL_BLOCK_SIZE(size);
  // Grow at the end of the arena if there's room; this is the fast path.
  if (memory->allocated + totalBlockSize <= memory->capacity) {
    blockheader *header = (blockheader *)(ARENA(memory) + memory->allocated);
    adjustBlock(memory, header, size);
    header->free = false;

    memory->allocated += totalBlockSize;
    memory->numBlocks += 1;

    return header + 1;
  }

  PANIC_IF(true, ALLOCATION_FAILED);
}


void *findSpaceAtEndOfBlock(Memory *memory, blockheader *currentBlockHeader,
                            size_t totalSpaceNeeded) {
  size_t oldSize = currentBlockHeader->blockSize;
  uint8_t *arena = ARENA(memory);

  size_t nextBlockOffset =
      BLOCK_HEADER_OFFSET(arena, currentBlockHeader) + NEXT_BLOCK_OFFSET(currentBlockHeader);
  if (nextBlockOffset >= memory->allocated) {
    return NULL;
  }

  blockheader *nextBlock = (blockheader *)(arena + nextBlockOffset);
  if (!nextBlock->free) {
    return NULL;
  }

  size_t mergedCapacity = oldSize + TOTAL_BLOCK_SIZE(nextBlock->blockSize);

  if (mergedCapacity < totalSpaceNeeded) {
    return NULL;
  }

  size_t leftover = mergedCapacity - totalSpaceNeeded;

  // The free next-block is being consumed; account for that.
  removeFreeBlock(memory, nextBlock);
  memory->numFreeBlocks -= 1;

  if (leftover > memory->minBlockSize) {
    blockheader *newBlock =
        (blockheader *)splitBlock(memory, currentBlockHeader, totalSpaceNeeded, leftover);
    newBlock->free = true;
    insertFreeBlock(memory, newBlock);
    memory->numFreeBlocks += 1;
  } else {
    adjustBlock(memory, currentBlockHeader, mergedCapacity);
    memory->numBlocks -= 1;
  }

  return currentBlockHeader + 1;
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
    if (newSpace == NULL) {
      return NULL;
    }
    memcpy(newSpace, ptr, memBlockHeader->blockSize);
    freeAlloc(memory, ptr);
    return newSpace;
  } else {
    return allocate(memory, size);
  }
}

static blockheader *coalesceBackwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena, memBlockHeader);

  if (headerOffset >= sizeof(blockheader) + sizeof(blockfooter)) {
    blockfooter *prevFooter = (blockfooter *)((uint8_t *)memBlockHeader - sizeof(blockfooter));
    blockheader *prevHeader = (blockheader *)(arena + prevFooter->headerOffset);

    if (prevHeader->free) {
      removeFreeBlock(memory, prevHeader);
      size_t mergedPayload = prevHeader->blockSize + sizeof(blockfooter) + sizeof(blockheader) +
                             memBlockHeader->blockSize;
      adjustBlock(memory, prevHeader, mergedPayload);
      memory->numBlocks -= 1;
      memory->numFreeBlocks -= 1;
      return prevHeader;
    }
  }

  return memBlockHeader;
}

static blockheader *coalesceForwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena, memBlockHeader);

  // Calculate position of next block
  size_t nextBlockPos = headerOffset + NEXT_BLOCK_OFFSET(memBlockHeader);

  // Check if there's a next block within allocated space
  if (nextBlockPos >= memory->allocated) {
    return memBlockHeader;
  }

  blockheader *nextHeader = (blockheader *)(ARENA(memory) + nextBlockPos);

  // If next block is free, merge it into current block
  if (nextHeader->free) {
    removeFreeBlock(memory, nextHeader);
    size_t mergedPayload = memBlockHeader->blockSize + sizeof(blockfooter) + sizeof(blockheader) +
                           nextHeader->blockSize;
    adjustBlock(memory, memBlockHeader, mergedPayload);
    memory->numBlocks -= 1;
    memory->numFreeBlocks -= 1;
  }

  return memBlockHeader;
}

void freeAlloc(Memory *memory, void *ptr) {
  if (ptr == NULL)
    return;

  blockheader *memBlockHeader = BLOCK_HEADER(ptr);
  if (memBlockHeader->free) {
    return;
  }

  memBlockHeader->free = true;
  memBlockHeader->nextFreeOffset = INVALID_OFFSET;
  memBlockHeader->prevFreeOffset = INVALID_OFFSET;
  memory->numFreeBlocks += 1;

  // TODO: revisit this allocated calulcation

  // Coalesce with neighboring free blocks
  memBlockHeader = coalesceBackwards(memory, memBlockHeader);
  memBlockHeader = coalesceForwards(memory, memBlockHeader);
  insertFreeBlock(memory, memBlockHeader);
}

void freeMemory(Memory *memory) {
  free(memory);
}


void printMemoryFragmentationChart(Memory *memory) {
  printf("== Memory Fragmentation Chart ==\n");
  printf("Allocated: %zu / %zu bytes\n", memory->allocated, memory->capacity);
  printf("Blocks: %zu total, %zu free\n\n", memory->numBlocks, memory->numFreeBlocks);

  uint8_t *arena = ARENA(memory);
  size_t byteIdx = 0;
  int col = 0;

  while (byteIdx < memory->allocated) {
    blockheader *header = (blockheader *)(arena + byteIdx);
    size_t blockSize = sizeof(blockheader) + header->blockSize + sizeof(blockfooter);

    if (header->free) {
      fprintf(stdout, "(%d) ", header->blockSize);
    } else {
      fprintf(stdout, "%d ", header->blockSize);
    }

    col++;
    if (col == 16) {
      fprintf(stdout, "\n");
      col = 0;
    }

    byteIdx += blockSize;
  }

  if (col != 0) {
    fprintf(stdout, "\n");
  }
}
