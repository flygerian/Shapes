#ifndef clox_memory_h
#define clox_memory_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define DEFAULT_ALLOCATION 1024 * 1024 * 64 // 64mb

#define NEXT_BLOCK_OFFSET(blockHeader) (sizeof(blockheader) + blockHeader->blockSize + sizeof(blockfooter))


#define ARENA(memory) ((uint8_t *)(memory + 1))

#define BLOCK_HEADER(ptr) ((blockheader *) ((uint8_t *)(ptr) - sizeof(blockheader)))
#define BLOCK_HEADER_OFFSET(arena, blockHeader) ((uint8_t *)blockHeader - arena)


#define BLOCK_FOOTER(blockHeader) ((blockfooter *)((uint8_t *)(blockHeader) + sizeof(blockheader) + blockHeader->blockSize))

#define TOTAL_BLOCK_SIZE(blockSize) (blockSize + (sizeof(blockheader) + sizeof(blockfooter)))

#define GROW_CAPACITY(capacity) (capacity) < 8 ? 8 : (capacity) * 2

#define GROW_ARRAY(memory, type, pointer, newcount)                                                \
  (type *)reallocate(memory, pointer, sizeof(type) * (newcount))

typedef struct {
  bool free;
  size_t blockSize;
} blockheader;

typedef struct {
  size_t headerOffset;
} blockfooter;

typedef struct {
  size_t capacity;
  size_t allocated;
  size_t numBlocks;
  size_t minBlockSize;
} Memory;

Memory *initializeMemory();
Memory *initializeArena(size_t arenaSize, size_t minBlockSize);
void *allocate(Memory *memory, size_t size);
void *reallocate(Memory *memory, void *ptr, size_t size);
void freeAlloc(Memory *memory, void *ptr);
void freeMemory(Memory *memory);
void printMemoryFragmentationChart(Memory *memory);

#endif
