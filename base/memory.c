#include "memory.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

Memory *initializeArena(size_t arenaSize) {
  Memory *head;
  head = malloc(sizeof(Memory) + arenaSize); // return the top of the heap;
                                              // top of the heap
  assert(head != NULL);

  head->capacity = arenaSize;
  head->allocated = 0;

  return head;
}

Memory *initializeMemory() {
  return initializeArena((size_t) DEFAULT_ALLOCATION);
}


void *findAvailableSpace(Memory *memory, size_t size) {
  if (memory->allocated == 0) {
    return NULL;
  }

  uint8_t *arena = ARENA(memory);
  size_t idxToCheck = 0;

  // Scan through all allocated blocks looking for a free one that fits
  while (idxToCheck < memory->allocated) {
    blockheader *headerAtIdx = (blockheader *)(arena + idxToCheck);

    if (headerAtIdx->free && headerAtIdx->blockSize >= size) {
      // Found a suitable free block
      headerAtIdx->free = false;
      
      // If the block is much larger than needed, split it
      size_t leftover = headerAtIdx->blockSize - size;
      size_t minBlockSize = sizeof(blockheader) + sizeof(blockfooter) + 1;
      
      if (leftover >= minBlockSize) {
        // Split the block
        size_t oldSize = headerAtIdx->blockSize;
        headerAtIdx->blockSize = size;
        
        // Update footer for the allocated block
        uint8_t *blockData = (uint8_t *)(headerAtIdx + 1);
        blockfooter *footer = (blockfooter *)(blockData + size);
        footer->headerOffset = idxToCheck;
        
        // Create a new free block with the leftover space
        blockheader *newHeader = (blockheader *)(footer + 1);
        newHeader->free = true;
        newHeader->blockSize = leftover - sizeof(blockheader) - sizeof(blockfooter);
        
        blockfooter *newFooter = (blockfooter *)((uint8_t *)(newHeader + 1) + newHeader->blockSize);
        newFooter->headerOffset = (uint8_t *)newHeader - arena;
      }
      
      return (uint8_t *)(headerAtIdx + 1);
    }

    size_t totalBlockSize = sizeof(blockheader) + headerAtIdx->blockSize + sizeof(blockfooter);
    idxToCheck += totalBlockSize;
  }
  
  return NULL;
}

void *allocate(Memory *memory, size_t size) {
  if (size == 0) {
    return NULL;
  }

  size_t totalBlockSize = sizeof(blockheader) + size + sizeof(blockfooter);

  // First try to find a free block that can accommodate this allocation
  // This prevents the arena from growing unnecessarily
  void *found = findAvailableSpace(memory, size);
  if (found != NULL) {
    return found;
  }

  // No suitable free block found, allocate at the end if there's space
  if (memory->allocated + totalBlockSize > memory->capacity) {
    fprintf(stderr, "ALLOCATOR: Out of memory! Tried to allocate %zu bytes, allocated=%zu, capacity=%zu\n",
            size, memory->allocated, memory->capacity);
    printMemoryFragmentationChart(memory);

    return NULL;
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

  if (headerOffset >= sizeof(blockheader) + sizeof(blockfooter)) {
    blockfooter *prevFooter = (blockfooter *)((uint8_t *)memBlockHeader - sizeof(blockfooter));
    blockheader *prevHeader = (blockheader *)(arena + prevFooter->headerOffset);

    if (prevHeader->free) {
      prevHeader->blockSize +=
          sizeof(blockfooter) + sizeof(blockheader) + memBlockHeader->blockSize;

      blockfooter *footer = (blockfooter *)((uint8_t *)(prevHeader + 1) + prevHeader->blockSize);
      footer->headerOffset = (uint8_t *)prevHeader - arena;
      
      // Return the merged header so caller can continue coalescing forward
      return;
    }
  }
}

void coalesceForwards(Memory *memory, blockheader *memBlockHeader) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = (uint8_t *)memBlockHeader - arena;
  
  // Calculate position of next block
  uint8_t *blockData = (uint8_t *)(memBlockHeader + 1);
  size_t currentBlockTotal = sizeof(blockheader) + memBlockHeader->blockSize + sizeof(blockfooter);
  uint8_t *nextBlockPos = (uint8_t *)memBlockHeader + currentBlockTotal;
  
  // Check if there's a next block within allocated space
  if ((size_t)(nextBlockPos - arena) >= memory->allocated) {
    return;
  }
  
  blockheader *nextHeader = (blockheader *)nextBlockPos;
  
  // If next block is free, merge it into current block
  if (nextHeader->free) {
    memBlockHeader->blockSize += 
        sizeof(blockfooter) + sizeof(blockheader) + nextHeader->blockSize;
    
    blockfooter *footer = (blockfooter *)((uint8_t *)(memBlockHeader + 1) + memBlockHeader->blockSize);
    footer->headerOffset = headerOffset;
  }
}

void freeAlloc(Memory *memory, void *ptr) {
  if (ptr == NULL)
    return;

  blockheader *memBlockHeader = BLOCK_HEADER(ptr);

  memBlockHeader->free = true;

  // Coalesce with neighboring free blocks
  coalesceBackwards(memory, memBlockHeader);
  coalesceForwards(memory, memBlockHeader);
  
  // If the freed block (possibly after coalescing) is at the end of allocated space,
  // rewind memory->allocated to reclaim that space
  uint8_t *arena = ARENA(memory);
  size_t blockOffset = (uint8_t *)memBlockHeader - arena;
  size_t blockTotalSize = sizeof(blockheader) + memBlockHeader->blockSize + sizeof(blockfooter);
  size_t blockEnd = blockOffset + blockTotalSize;
  
  // Check if this is the last allocated block
  if (blockEnd == memory->allocated && memBlockHeader->free) {
    // This block is at the end and is free, we can reclaim it
    memory->allocated = blockOffset;
    
    // Keep rewinding if the previous block is also free
    while (blockOffset > 0) {
      // Check if there's a previous block
      if (blockOffset < sizeof(blockheader) + sizeof(blockfooter)) {
        break;
      }
      
      blockfooter *prevFooter = (blockfooter *)(arena + blockOffset - sizeof(blockfooter));
      blockheader *prevHeader = (blockheader *)(arena + prevFooter->headerOffset);
      
      if (!prevHeader->free) {
        break;
      }
      
      // Previous block is free, reclaim it too
      blockOffset = prevFooter->headerOffset;
      memory->allocated = blockOffset;
    }
  }
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
      printf("(%d) ", header->blockSize);
    } else {
      printf("%d ", header->blockSize);
    }

    col++;
    if (col == 16) {
      printf("\n");
      col = 0;
    }

    byteIdx += blockSize;
  }

  if (col != 0) {
    printf("\n");
  }
}
