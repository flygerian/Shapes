#include "memory.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



Memory *initializeArena(size_t arenaSize, size_t minBlockSize) {
  Memory *head;
  head = malloc(sizeof(Memory) + arenaSize); // return the top of the heap;
                                              // top of the heap
  assert(head != NULL);

  head->capacity = arenaSize;
  head->allocated = 0;
  head->numBlocks = 0;
  head->minBlockSize = minBlockSize;

  return head;
}

Memory *initializeMemory() {
  return initializeArena((size_t) DEFAULT_ALLOCATION, 1);
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

  blockfooter *footer = BLOCK_FOOTER(header);
  footer->headerOffset = headerOffset;

  return header;
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
  if (memory->allocated == 0) {
    return NULL;
  }

  uint8_t *arena = ARENA(memory);
  size_t arenaOffset = 0;
  // Snapshot before the loop: splitBlock increments memory->allocated, so using
  // the live value as the bound causes the scan to run into uninitialized memory.
  size_t scanLimit = memory->allocated;

  // Scan through all allocated blocks looking for a free one that fits
  while (arenaOffset < scanLimit) {
    blockheader *headerAtOffset = (blockheader *)(arena + arenaOffset);

    if (headerAtOffset->free && headerAtOffset->blockSize >= size) {
      // If the block is much larger than needed, split it. The remainder payload
      // must account for the header+footer that the split consumes.
      size_t currentBlockSize = headerAtOffset->blockSize;

      if (currentBlockSize - size >= TOTAL_BLOCK_SIZE(memory->minBlockSize)) {
        size_t leftover = currentBlockSize - size - sizeof(blockheader) - sizeof(blockfooter);
        blockheader *remainder = splitBlock(memory, headerAtOffset, size, leftover);
        remainder->free = true;
      }

      // Found a suitable free block
      headerAtOffset->free = false;

      return (uint8_t *)(headerAtOffset + 1);
    }

    arenaOffset += TOTAL_BLOCK_SIZE(headerAtOffset->blockSize);
  }
  
  return NULL;
}

void *allocate(Memory *memory, size_t size) {
  if (size == 0) {
    return NULL;
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

  // Arena is full; scan for a freed block to reuse.
  return findAvailableSpace(memory, size);
}


void *findSpaceAtEndOfBlock(Memory *memory, blockheader *currentBlockHeader,
                            size_t totalSpaceNeeded) {
  size_t oldSize = currentBlockHeader->blockSize;
  uint8_t *arena = ARENA(memory);

  size_t nextBlockOffset = BLOCK_HEADER_OFFSET(arena, currentBlockHeader) + NEXT_BLOCK_OFFSET(currentBlockHeader);
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

  if (leftover > memory->minBlockSize) {
    blockheader* newBlock = (blockheader*) splitBlock(memory, currentBlockHeader, totalSpaceNeeded, leftover);
    newBlock->free = true;
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
    memcpy(newSpace, ptr, memBlockHeader->blockSize);
    freeAlloc(memory, ptr);
    return newSpace;
  } else {
    return allocate(memory, size);
  }
}

void coalesceBackwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena , memBlockHeader);

  if (headerOffset >= sizeof(blockheader) + sizeof(blockfooter)) {
    blockfooter *prevFooter = (blockfooter *)((uint8_t *)memBlockHeader - sizeof(blockfooter));
    blockheader *prevHeader = (blockheader *)(arena + prevFooter->headerOffset);

    if (prevHeader->free) {
      size_t mergedPayload = prevHeader->blockSize + sizeof(blockfooter) + sizeof(blockheader) + memBlockHeader->blockSize;
      adjustBlock(memory, prevHeader, mergedPayload);
      memory->numBlocks -= 1;
      return;
    }
  }
}

void coalesceForwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena, memBlockHeader);
  
  // Calculate position of next block
  size_t nextBlockPos = headerOffset + NEXT_BLOCK_OFFSET(memBlockHeader);
  
  // Check if there's a next block within allocated space
  if (nextBlockPos >= memory->allocated) {
    return;
  }
  
  blockheader *nextHeader = (blockheader *) (ARENA(memory) + nextBlockPos);
  
  // If next block is free, merge it into current block
  if (nextHeader->free) {
    size_t mergedPayload = memBlockHeader->blockSize + sizeof(blockfooter) + sizeof(blockheader) + nextHeader->blockSize;
    adjustBlock(memory, memBlockHeader, mergedPayload);
    memory->numBlocks -= 1;
  }
}

void freeAlloc(Memory *memory, void *ptr) {
  if (ptr == NULL)
    return;

  blockheader *memBlockHeader = BLOCK_HEADER(ptr);

  memBlockHeader->free = true;

  // TODO: revisit this allocated calulcation

  // Coalesce with neighboring free blocks
  coalesceBackwards(memory, memBlockHeader);
  coalesceForwards(memory, memBlockHeader);
}

void freeMemory(Memory *memory) {
  free(memory);
}


void printMemoryFragmentationChart(Memory *memory) {
  printf("== Memory Fragmentation Chart ==\n");
  printf("Allocated: %zu / %zu bytes\n\n", memory->allocated, memory->capacity);

  uint8_t *arena = ARENA(memory);
  size_t byteIdx = 0;
  int col = 0;

  while (byteIdx < memory->allocated) {
    blockheader *header = (blockheader *)(arena + byteIdx);
    size_t blockSize =
        sizeof(blockheader) + header->blockSize + sizeof(blockfooter);

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
