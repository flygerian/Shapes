#ifndef utils_lib_set_h
#define utils_lib_set_h

#include "memory.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uintptr_t key;
  void *value;
} PtrMapEntry;

typedef struct {
  Array *entries;
  size_t capacity;
  size_t count;
  size_t valueElemsize;
  Memory *memory;
} Map;

typedef Map PtrSet;
typedef Map PtrMap;

PtrSet *Make_PtrSet(Memory *memory);
void PtrSet_Put(PtrSet *pt, void *key);
bool PtrSet_Contains(PtrSet *pt, void *key);
bool PtrSet_DoesNotContain(PtrSet *pt, void *key);

PtrMap *Make_PtrMap(Memory *memory, size_t initialCapacity);
PtrMapEntry *Make_PtrMapEntry(Memory *memory, uintptr_t key, void *value);
void PtrMap_Put(PtrMap *pm, void *key, void *value);
void *PtrMap_Get(PtrMap *pt, void *key);
bool PtrMap_Contains(PtrMap *pt, void *key);

void Array_SetPtrMapEntryAt(Array *array, size_t idx, PtrMapEntry *entry);
PtrMapEntry *Array_PtrMapEntryIdx(Array *array, size_t idx);

#endif
