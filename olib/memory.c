#include "memory.h"
#include "result.h"
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
#define ALLOC_ALIGN ((size_t)16)
#define ALIGN_UP(value, align) (((value) + ((align) - 1)) & ~((align) - 1))
#define BLOCK_ID 0x10010110
#ifdef __APPLE__
  #include <mach/vm_statistics.h>
  #define HUGE_PAGE_FD  VM_FLAGS_SUPERPAGE_SIZE_2MB
  #define HUGE_PAGE_FLAGS (MAP_PRIVATE | MAP_ANON)
#else
  // Linux
  #define HUGE_PAGE_FD  -1
  #define HUGE_PAGE_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#endif

olib_Memory *olib_InitializeArena(size_t arenaSize, size_t minBlockSize) {
  size_t totalSize = sizeof(olib_Memory) + arenaSize;
  size_t alignedSize = ALIGN_TO_HUGE(totalSize);  // round up to 2MB boundary

  olib_Memory *head = mmap(NULL, alignedSize, PROT_READ | PROT_WRITE,
                 HUGE_PAGE_FLAGS, HUGE_PAGE_FD, 0);

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

olib_Memory *olib_InitializeArenaWithBuffer(void *buffer, size_t bufferSize, size_t minBlockSize) {
  assert(buffer != NULL);
  assert(bufferSize > sizeof(olib_Memory));

  olib_Memory *head = (olib_Memory *)buffer;
  head->capacity = bufferSize - sizeof(olib_Memory);
  head->allocated = 0;
  head->numBlocks = 0;
  head->numFreeBlocks = 0;
  head->minBlockSize = minBlockSize;
  head->freeHeadOffset = INVALID_OFFSET;

  return head;
}

void olib_ResetArena(olib_Memory *memory) {
  assert(memory != NULL);

  memory->allocated = 0;
  memory->numBlocks = 0;
  memory->numFreeBlocks = 0;
  memory->freeHeadOffset = INVALID_OFFSET;
}

void olib_PopScratch(void *ptr) {
  blockheader *header = BLOCK_HEADER(ptr);
  olib_Memory *memory = ((void*)header->arena) - sizeof(olib_Memory);
  memory->allocated -= header->blockSize;
}

olib_Memory *olib_InitializeMemory() {
  return olib_InitializeArena((size_t)DEFAULT_ALLOCATION, 1);
}

olib_Memory* olib_GetScratchArena(olib_Memory *memory, size_t scratchBufferSize) {
  void *scratchBuffer = olib_Allocate(memory, scratchBufferSize);
  return olib_InitializeArenaWithBuffer(scratchBuffer, scratchBufferSize, 1);
}
// Writes a block header and footer for a payload of `size` bytes at `header`.
// Returns header on success, NULL if the block would exceed arena capacity.
static inline void *adjustBlock(olib_Memory *memory, blockheader *header, size_t size) {
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

void *olib_Allocate(olib_Memory *memory, size_t size) {
  // Round the payload up so every block keeps the arena cursor aligned. Without this,
  // odd-byte allocations (e.g. strings, bitsets) leave subsequent float blocks unaligned,
  // which breaks Accelerate SIMD routines like vvtanhf even though scalar loops tolerate it.
  size_t alignedSize = ALIGN_UP(size, ALLOC_ALIGN);
  size_t totalBlockSize = TOTAL_BLOCK_SIZE(alignedSize);
  // Grow at the end of the arena if there's room; this is the fast path.
  if (memory->allocated + totalBlockSize <= memory->capacity) {
    blockheader *header = (blockheader *)(ARENA(memory) + memory->allocated);
    adjustBlock(memory, header, alignedSize);
    header->free = false;

    memory->allocated += totalBlockSize;
    memory->numBlocks += 1;

    return header + 1;
  }

  PANIC_IF(true, ALLOCATION_FAILED);
}

void freeAlloc(olib_Memory *memory, void *ptr) {
  if (ptr == NULL) {
    return; 
  }

  blockheader *memBlockHeader = BLOCK_HEADER(ptr);
  if (memBlockHeader->free) {
    return;
  }

  memBlockHeader->free = true;
  memBlockHeader->nextFreeOffset = INVALID_OFFSET;
  memBlockHeader->prevFreeOffset = INVALID_OFFSET;
  memory->numFreeBlocks += 1;
}

void *olib_Reallocate(olib_Memory *memory, void *ptr, size_t size) {
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

    void *newSpace = olib_Allocate(memory, size);
    if (newSpace == NULL) {
      return NULL;
    }
    memcpy(newSpace, ptr, memBlockHeader->blockSize);
    freeAlloc(memory, ptr);
    return newSpace;
  } else {
    return olib_Allocate(memory, size);
  }
}

void olib_FreeMemory(olib_Memory *memory) {
  free(memory);
}

