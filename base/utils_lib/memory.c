#include "memory.h"
#include "result/result.h"
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define HUGE_PAGE_SIZE (2UL * 1024 * 1024)
#define ALIGN_TO_HUGE(size) (((size) + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1))
#define BLOCK_ID 0x10010110

Memory *initializeArena(size_t arenaSize, size_t minBlockSize) {
  size_t totalSize = sizeof(Memory) + arenaSize;
  size_t alignedSize = ALIGN_TO_HUGE(totalSize);  // round up to 2MB boundary

  Memory *head = mmap(NULL, alignedSize,
                      PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB,
                      -1, 0);

  if (head == MAP_FAILED) {
    printf("mmap failed: %d", errno);
    // fall back to regular pages
    head = mmap(NULL, totalSize,
                PROT_READ|PROT_WRITE,
                MAP_PRIVATE|MAP_ANONYMOUS,
                -1, 0);
  }

  assert(head != MAP_FAILED);

  head->capacity = arenaSize;
  head->allocated = 0;
  head->numBlocks = 0;
  head->numFreeBlocks = 0;
  head->minBlockSize = minBlockSize;
  head->freeHeadOffset = INVALID_OFFSET;

  return head;
}

Memory *initializeArenaWithBuffer(void *buffer, size_t bufferSize, size_t minBlockSize) {
  assert(buffer != NULL);
  assert(bufferSize > sizeof(Memory));

  Memory *head = (Memory *)buffer;
  head->capacity = bufferSize - sizeof(Memory);
  head->allocated = 0;
  head->numBlocks = 0;
  head->numFreeBlocks = 0;
  head->minBlockSize = minBlockSize;
  head->freeHeadOffset = INVALID_OFFSET;

  return head;
}

void resetArena(Memory *memory) {
  assert(memory != NULL);

  memory->allocated = 0;
  memory->numBlocks = 0;
  memory->numFreeBlocks = 0;
  memory->freeHeadOffset = INVALID_OFFSET;
}

void popScratch(void *ptr) {
  blockheader *header = BLOCK_HEADER(ptr);
  Memory *memory = ((void*)header->arena) - sizeof(Memory);
  memory->allocated -= header->blockSize;
}

Memory *initializeMemory() {
  return initializeArena((size_t)DEFAULT_ALLOCATION, 1);
}

Memory* GetScratchArena(Memory *memory, size_t scratchBufferSize) {
  void *scratchBuffer = allocate(memory, scratchBufferSize);
  return initializeArenaWithBuffer(scratchBuffer, scratchBufferSize, 1);
}

// Writes a block header and footer for a payload of `size` bytes at `header`.
// Returns header on success, NULL if the block would exceed arena capacity.
static inline void *adjustBlock(Memory *memory, blockheader *header, size_t size) {
  uint8_t *arena = ARENA(memory);
  size_t headerOffset = BLOCK_HEADER_OFFSET(arena, header);

  if (headerOffset + TOTAL_BLOCK_SIZE(size) > memory->capacity) {
    return NULL;
  }

  header->blockID = (uint8_t) BLOCK_ID;
  header->arena = arena;
  header->blockSize = size;
  header->nextFreeOffset = INVALID_OFFSET;
  header->prevFreeOffset = INVALID_OFFSET;

  blockfooter *footer = BLOCK_FOOTER(header);
  footer->headerOffset = headerOffset;

  return header;
}

void *allocate(Memory *memory, size_t size) {
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
