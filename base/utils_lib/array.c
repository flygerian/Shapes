#include "memory.h"
#include "result/result.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "array.h"

#define INITIAL_SLICE_CAPACITY 16
#define SLICE_GROW_FACTOR      8

#define ARRAY_PTR_AT_IDX(slice, idx) ((slice)->items + ((idx) * (slice)->elemSize))

Array *contructNewArray(Memory *memory, size_t elemSize, size_t capacity) {
  Array *alloc = allocate(memory, sizeof(Array));

  alloc->items = allocate(memory, capacity * elemSize), alloc->capacity = capacity,
  alloc->elemSize = elemSize;
  alloc->isCapacityFixed = true;
  alloc->size = 0;
  alloc->memory = memory;

  return alloc;
}

static inline void expandCapacity(Array *slice) {
  PANIC_IF(slice == NULL, ERR_NULL_PTR);

  const size_t newCapacity = slice->capacity + SLICE_GROW_FACTOR;
  void *newItemsAllocation =
      reallocate(slice->memory, slice->items, (newCapacity * slice->elemSize));

  slice->capacity = newCapacity;
  slice->items = newItemsAllocation;
}

Array *MakeDynamicArray(Memory *memory, const size_t elemSize) {
  Array *arr = contructNewArray(memory, elemSize, INITIAL_SLICE_CAPACITY);
  arr->isCapacityFixed = false;

  return arr;
}

Array *MakeArray(Memory *memory, const size_t elemSize, size_t capacity) {
  return contructNewArray(memory, elemSize, capacity);
}

void Array_SetAt(Array *array, size_t idx, void *ptr) {
  PANIC_IF(array == NULL, ERR_NULL_PTR);
  PANIC_IF(ptr == NULL, ERR_NULL_PTR);
  PANIC_IF(idx >= array->capacity, ERR_OUT_OF_BOUNDS);
  PANIC_IF(idx < 0, ERR_OUT_OF_BOUNDS);

  void *slotStart = ARRAY_PTR_AT_IDX(array, idx);
  memcpy(slotStart, ptr, array->elemSize);
}

void Array_Append(Array *array, void *ptr) {
  if (array->size + 1 >= array->capacity && !array->isCapacityFixed) {
    expandCapacity(array);
  }

  PANIC_IF(array->size == array->capacity, ERR_OUT_OF_BOUNDS);

  Array_SetAt(array, array->size, ptr);
  array->size += 1;
}

void *Array_Idx(Array *slice, size_t idx) {
  PANIC_IF(slice == NULL, ERR_NULL_PTR);
  PANIC_IF(idx >= slice->size, ERR_OUT_OF_BOUNDS);
  PANIC_IF(idx < 0, ERR_OUT_OF_BOUNDS);

  return ARRAY_PTR_AT_IDX(slice, idx);
}

