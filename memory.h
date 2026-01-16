#ifndef clox_memory_h
#define clox_memory_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define ALLOCATION 1024 * 1024 * 1 // 1mb

#define ARENA(memory) ((uint8_t *)(memory + 1))

#define BLOCK_HEADER(ptr)                                                      \
  ((blockheader *)((uint8_t *)(ptr) - sizeof(blockheader)))

#define GROW_CAPACITY(capacity) (capacity) < 8 ? 8 : (capacity) * 2

#define GROW_ARRAY(memory, type, pointer, newcount)                            \
  (type *)reallocate(memory, pointer, sizeof(type) * (newcount))

typedef struct {
  bool free;
  uint16_t blockSize;
} blockheader;

typedef struct {
  size_t headerOffset;
} blockfooter;

typedef struct {
  size_t capacity;
  size_t allocated;
} Memory;

Memory *initializeMemory();
void *allocate(Memory *memory, size_t size);
void *reallocate(Memory *memory, void *ptr, size_t size);
void freeAlloc(Memory *memory, void *ptr);
void freeMemory(Memory *memory);

#endif
