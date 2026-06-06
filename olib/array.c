#include "memory.h"
#include "result.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "array.h"

#define INITIAL_SLICE_CAPACITY 16
#define SLICE_GROW_FACTOR      8

olib_Array *contructNewArray(olib_Memory *memory, size_t elemSize, size_t capacity) {
  olib_Array *alloc = olib_Allocate(memory, sizeof(olib_Array));

  alloc->items = olib_Allocate(memory, capacity * elemSize), alloc->capacity = capacity,
  alloc->elemSize = elemSize;
  alloc->isCapacityFixed = true;
  alloc->size = 0;
  alloc->memory = memory;
  memset(alloc->items, 0, elemSize * capacity);

  return alloc;
}

static inline void expandIfNeeded(olib_Array *array, size_t numItemsToAdd) {
  PANIC_IF(array == NULL, ERR_NULL_PTR);
  bool needsExpansion = array->size + numItemsToAdd > array->capacity;
  PANIC_IF(needsExpansion && array->isCapacityFixed, ERR_EXPAND_FIXED_ARRAY);

  if (!needsExpansion) {
    return;
  }

  size_t newCapacity = array->capacity;
  while (newCapacity < array->size + numItemsToAdd) {
    newCapacity += SLICE_GROW_FACTOR;
  }

  void *newItemsAllocation =
      olib_Reallocate(array->memory, array->items, (newCapacity * array->elemSize));

  array->capacity = newCapacity;
  array->items = newItemsAllocation;
}

olib_Array *olib_MakeDynamicArray(olib_Memory *memory, const size_t elemSize) {
  olib_Array *arr = contructNewArray(memory, elemSize, INITIAL_SLICE_CAPACITY);
  arr->isCapacityFixed = false;

  return arr;
}

olib_Array *olib_MakeArray(olib_Memory *memory, const size_t elemSize, size_t capacity) {
  return contructNewArray(memory, elemSize, capacity);
}

void olib_ArraySetAt(olib_Array *array, size_t idx, void *ptr) {
  PANIC_IF(array == NULL, ERR_NULL_PTR);
  PANIC_IF(ptr == NULL, ERR_NULL_PTR);
  PANIC_IF(idx >= array->capacity, ERR_OUT_OF_BOUNDS);
  PANIC_IF(idx < 0, ERR_OUT_OF_BOUNDS);

  void *slotStart = ARRAY_PTR_AT_IDX(array, idx);
  memcpy(slotStart, ptr, array->elemSize);
}

void olib_ArrayAppend(olib_Array *array, void *ptr) {
  expandIfNeeded(array, 1);

  PANIC_IF(array->size == array->capacity, ERR_OUT_OF_BOUNDS);

  olib_ArraySetAt(array, array->size, ptr);
  array->size += 1;
}

void olib_ArrayAppendString(olib_Array *array, olib_String str) {
  olib_ArrayAppend(array, (void *)&str);
}

void olib_ArrayReset(olib_Array *array) {
    memset(array->items, 0, array->size);
    array->size = 0;
}

olib_Array *olib_ArraySlice(olib_Array *src, size_t start, size_t count) {
  PANIC_IF(src == NULL, ERR_NULL_PTR);
  PANIC_IF(start + count > src->size, ERR_OUT_OF_BOUNDS);

  olib_Array *slice = olib_MakeArray(src->memory, src->elemSize, count);
  memcpy(slice->items, (byte *)src->items + start * src->elemSize, count * src->elemSize);
  slice->size = count;
  return slice;
}

olib_String olib_ArrayStringIdx(olib_Array *array, size_t idx) {
  return *((olib_String *)olib_ArrayIdx(array, idx));
}

// Only meant for string literals
olib_String olib_MakeString(olib_Memory *memory, char *stringData) {
  size_t len = strlen(stringData);
  return olib_MakeStringN(memory, stringData, len);
}

olib_String olib_MakeStringN(olib_Memory *memory, char *stringData, size_t len) {
  olib_String str = olib_MakeDynamicArray(memory, sizeof(char));
  expandIfNeeded(str, len + 1);
  memcpy(str->items, stringData, len);
  ((char *)str->items)[len] = '\0';
  str->size = len;
  return str;
}

void olib_StringAppendCString(olib_String str, const char *cstr) {
  PANIC_IF(str == NULL, ERR_NULL_PTR);
  PANIC_IF(cstr == NULL, ERR_NULL_PTR);
  PANIC_IF(str->elemSize != sizeof(char), ARRAY_ELEM_SIZE_MISMATCH);

  size_t len = strlen(cstr);
  expandIfNeeded(str, len + 1);
  memcpy((char *)str->items + str->size, cstr, len);
  str->size += len;
  ((char *)str->items)[str->size] = '\0';
}

void olib_StringAppendFormat(olib_String str, const char *fmt, ...) {
  PANIC_IF(str == NULL, ERR_NULL_PTR);
  PANIC_IF(fmt == NULL, ERR_NULL_PTR);
  PANIC_IF(str->elemSize != sizeof(char), ARRAY_ELEM_SIZE_MISMATCH);

  va_list args;
  va_start(args, fmt);

  va_list measure;
  va_copy(measure, args);
  int needed = vsnprintf(NULL, 0, fmt, measure);
  va_end(measure);

  PANIC_IF(needed < 0, ERR_NO_OP);

  expandIfNeeded(str, (size_t)needed + 1);
  vsnprintf((char *)str->items + str->size, (size_t)needed + 1, fmt, args);
  str->size += (size_t)needed;

  va_end(args);
}

olib_ArrayF32 olib_MakeDynamicF32Array(olib_Memory *memory) {
  return olib_MakeDynamicArray(memory, sizeof(f32));
}

olib_ArrayF32 olib_MakeF32Array(olib_Memory *memory, size_t capacity) {
  return olib_MakeArray(memory, sizeof(f32), capacity);
}

void olib_ArrayAppendF32(olib_Array *array, f32 num) {
  olib_ArrayAppend(array, &num);
}

void olib_ArrayAppendF32Buffer(olib_Array *array, f32 *num, size_t numItems) {
  PANIC_IF_NULL(num);
  PANIC_IF(array->elemSize != sizeof(f32), ARRAY_ELEM_SIZE_MISMATCH);

  expandIfNeeded(array, numItems);

  f32* items = array->items;
  f32* arrEnd = items + array->size;

  memcpy(arrEnd, num, numItems * sizeof(f32));
  array->size += numItems;
}

f32 olib_ArrayF32Idx(olib_Array *array, size_t idx) {
  return *( (f32*) olib_ArrayIdx(array, idx) );
}
