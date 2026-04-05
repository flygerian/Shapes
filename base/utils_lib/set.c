#include "set.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "result/result.h"

#define INITIAL_PTR_SET_CAPACITY     16
#define PTR_SET_MAX_LOAD_NUMERATOR   3
#define PTR_SET_MAX_LOAD_DENOMINATOR 4

static inline size_t bucketIndex(void *key, size_t capacity) {
  uintptr_t addr = (uintptr_t)key;

  addr ^= addr >> 4;
  return addr % capacity;
}

static inline bool shouldGrow(PtrSet *pt) {
  return (pt->count + 1) * PTR_SET_MAX_LOAD_DENOMINATOR > pt->capacity * PTR_SET_MAX_LOAD_NUMERATOR;
}

static inline void initializeEntries(void **entries, size_t capacity) {
  memset(entries, 0, sizeof(void *) * capacity);
}

static inline size_t findSlot(void **entries, size_t capacity, void *key) {
  size_t idx = bucketIndex(key, capacity);

  while (entries[idx] != NULL && entries[idx] != key) {
    idx = (idx + 1) % capacity;
  }

  return idx;
}

static void growPtrSet(PtrSet *pt) {
  size_t oldCapacity = pt->capacity;
  void **oldEntries = pt->entries;
  size_t newCapacity = oldCapacity * 2;
  void **newEntries = allocate(pt->memory, sizeof(void *) * newCapacity);

  initializeEntries(newEntries, newCapacity);

  for (size_t i = 0; i < oldCapacity; i++) {
    void *entry = oldEntries[i];
    if (entry == NULL) {
      continue;
    }

    size_t idx = findSlot(newEntries, newCapacity, entry);
    newEntries[idx] = entry;
  }

  pt->entries = newEntries;
  pt->capacity = newCapacity;
}

PtrSet *MakePtrSet(Memory *memory) {
  PtrSet *ptrSetAlloc = allocate(memory, sizeof(PtrSet));
  ptrSetAlloc->entries = allocate(memory, sizeof(void *) * INITIAL_PTR_SET_CAPACITY);
  ptrSetAlloc->capacity = INITIAL_PTR_SET_CAPACITY;
  ptrSetAlloc->count = 0;
  ptrSetAlloc->memory = memory;
  initializeEntries(ptrSetAlloc->entries, ptrSetAlloc->capacity);

  return ptrSetAlloc;
}

void PtrSet_Put(PtrSet *pt, void *key) {
  PANIC_IF(pt == NULL, ERR_NULL_PTR);
  PANIC_IF(key == NULL, ERR_NULL_PTR);

  if (shouldGrow(pt)) {
    growPtrSet(pt);
  }

  size_t idx = findSlot(pt->entries, pt->capacity, key);
  if (pt->entries[idx] == NULL) {
    pt->entries[idx] = key;
    pt->count += 1;
  }
}

bool PtrSet_Contains(PtrSet *pt, void *key) {
  PANIC_IF(pt == NULL, ERR_NULL_PTR);
  PANIC_IF(key == NULL, ERR_NULL_PTR);

  size_t idx = findSlot(pt->entries, pt->capacity, key);
  return pt->entries[idx] == key;
}
