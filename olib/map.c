#include "map.h"
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "result.h"
#include "array.h"
#include "memory.h"

#define INITIAL_PTR_SET_CAPACITY     16
#define PTR_SET_MAX_LOAD_NUMERATOR   3
#define PTR_SET_MAX_LOAD_DENOMINATOR 4

void Array_SetPtrMapEntryAt(Array *array, size_t idx, PtrMapEntry *entry) {
  Array_SetAt(array, idx, (void *)&entry);
}

PtrMapEntry *Array_PtrMapEntryIdx(Array *array, size_t idx) {
  PtrMapEntry **item = (PtrMapEntry **)Array_Idx(array, idx);
  if (item == NULL) {
    return NULL;
  }
  return *item;
}

static inline size_t bucketIndex(void *key, size_t capacity) {
  uintptr_t addr = (uintptr_t)key;
  addr ^= addr >> 4;
  return addr % capacity;
}

static inline bool shouldGrow(PtrSet *pt) {
  return (pt->count + 1) * PTR_SET_MAX_LOAD_DENOMINATOR > pt->capacity * PTR_SET_MAX_LOAD_NUMERATOR;
}

static inline size_t findSlot(Array *entries, size_t capacity, void *key) {
  size_t idx = bucketIndex(key, capacity);

  while (Array_PtrMapEntryIdx(entries, idx) != NULL &&
         Array_PtrMapEntryIdx(entries, idx)->key != (uintptr_t)key) {
    idx = (idx + 1) % capacity;
  }

  return idx;
}

static void growPtrMap(PtrMap *pt) {
  size_t oldCapacity = pt->capacity;
  Array *oldEntries = pt->entries;
  size_t newCapacity = oldCapacity * 2;
  Array *newEntries = MakeArray(pt->memory, sizeof(PtrMapEntry *), newCapacity);

  for (size_t i = 0; i < oldCapacity; i++) {
    PtrMapEntry *entry = Array_PtrMapEntryIdx(oldEntries, i);
    if (entry == NULL) {
      continue;
    }

    size_t idx = findSlot(newEntries, newCapacity, (void *)entry->key);
    Array_SetPtrMapEntryAt(newEntries, idx, entry);
  }

  pt->entries = newEntries;
  pt->capacity = newCapacity;
}

PtrMap *Make_PtrMap(Memory *memory, size_t initialCapacity) {
  PtrMap *ptrSetAlloc = allocate(memory, sizeof(PtrMap));
  ptrSetAlloc->capacity = initialCapacity;
  ptrSetAlloc->count = 0;
  ptrSetAlloc->memory = memory;
  ptrSetAlloc->entries = MakeArray(memory, sizeof(PtrMapEntry *), initialCapacity);

  return ptrSetAlloc;
}

PtrSet *Make_PtrSet(Memory *memory) {
  return Make_PtrMap(memory, INITIAL_PTR_SET_CAPACITY);
}

PtrMapEntry *Make_PtrMapEntry(Memory *memory, uintptr_t key, void *value) {
  PtrMapEntry *entry = allocate(memory, sizeof(PtrMapEntry));
  *entry = (PtrMapEntry){.key = key, .value = value};
  return entry;
}

void PtrMap_Put(PtrMap *pm, void *key, void *value) {
  PANIC_IF(pm == NULL, ERR_NULL_PTR);
  PANIC_IF(key == 0, ERR_NULL_PTR);

  if (shouldGrow(pm)) {
    growPtrMap(pm);
  }

  size_t idx = findSlot(pm->entries, pm->capacity, key);
  if (Array_PtrMapEntryIdx(pm->entries, idx) == NULL) {
    PtrMapEntry *entry = Make_PtrMapEntry(pm->memory, (uintptr_t)key, value);
    Array_SetPtrMapEntryAt(pm->entries, idx, entry);
    pm->entries->size += 1;
    pm->count += 1;
  }
}

void *PtrMap_Get(PtrMap *pt, void *key) {
  PANIC_IF(pt == NULL, ERR_NULL_PTR);
  PANIC_IF(key == 0, ERR_NULL_PTR);

  size_t idx = findSlot(pt->entries, pt->capacity, key);
  PtrMapEntry *entry = Array_PtrMapEntryIdx(pt->entries, idx);
  if (entry == NULL) {
    return NULL;
  }

  return entry->value;
}

bool PtrMap_Contains(PtrMap *pt, void *key) {
  PANIC_IF(pt == NULL, ERR_NULL_PTR);
  PANIC_IF(key == 0, ERR_NULL_PTR);

  size_t idx = findSlot(pt->entries, pt->capacity, key);
  return Array_PtrMapEntryIdx(pt->entries, idx) != NULL;
}

bool PtrSet_DoesNotContain(PtrSet *pt, void *key) {
  return !PtrSet_Contains(pt, key);
}

void PtrSet_Put(PtrSet *pt, void *key) {
  PtrMap_Put(pt, key, (void *)key);
}

bool PtrSet_Contains(PtrSet *pt, void *key) {
  return PtrMap_Contains(pt, key);
}
