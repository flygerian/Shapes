#ifndef utils_lib_memory_h
#define utils_lib_memory_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define DEFAULT_ALLOCATION 1024 * 1024 * 64 // 64mb

#define NEXT_BLOCK_OFFSET(blockHeader)                                                             \
  (sizeof(blockheader) + blockHeader->blockSize + sizeof(blockfooter))

#define ARENA(memory) ((uint8_t *)(memory + 1))

#define BLOCK_HEADER(ptr) ((blockheader *)((uint8_t *)(ptr) - sizeof(blockheader)))
#define BLOCK_HEADER_OFFSET(arena, blockHeader) ((uint8_t *)blockHeader - arena)


#define BLOCK_FOOTER(blockHeader)                                                                  \
  ((blockfooter *)((uint8_t *)(blockHeader) + sizeof(blockheader) + blockHeader->blockSize))

#define TOTAL_BLOCK_SIZE(blockSize) (blockSize + (sizeof(blockheader) + sizeof(blockfooter)))
#define INVALID_OFFSET              ((size_t)-1)

#define HEADER_AT(memory, offset)     ((blockheader *)(ARENA(memory) + (offset)))
#define HEADER_OFFSET(memory, header) BLOCK_HEADER_OFFSET(ARENA(memory), (header))

typedef struct {
  uint8_t blockID;
  bool free;
  uint8_t *arena;
  size_t blockSize;
  size_t nextFreeOffset;
  size_t prevFreeOffset;
  // Pads sizeof(blockheader) from 40 to 48 so the user payload (header + 1) lands
  // on a 16-byte boundary. Needed for SIMD routines like vvtanhf.
  size_t _padding;
} blockheader;

typedef struct {
  size_t headerOffset;
} blockfooter;

typedef struct {
  size_t capacity;
  size_t allocated;
  size_t numBlocks;
  size_t numFreeBlocks;
  size_t minBlockSize;
  size_t freeHeadOffset;
} olib_Memory;

olib_Memory* olib_InitializeMemory();
olib_Memory* olib_InitializeArena(size_t arenaSize, size_t minBlockSize);
olib_Memory* olib_InitializeArenaWithBuffer(void *buffer, size_t bufferSize, size_t minBlockSize);
olib_Memory* olib_GetScratchArena(olib_Memory *memory, size_t scratchBufferSize);
void  olib_ResetArena(olib_Memory *memory);
void* olib_Allocate(olib_Memory *memory, size_t size);
void* olib_Reallocate(olib_Memory *memory, void *ptr, size_t size);
void  olib_FreeMemory(olib_Memory *memory);
void  olib_PopScratch(void *ptr);

#endif
