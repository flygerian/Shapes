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


Array *contructNewArray(Memory *memory, size_t elemSize, size_t capacity) {
  Array *alloc = allocate(memory, sizeof(Array));

  alloc->items = allocate(memory, capacity * elemSize), alloc->capacity = capacity,
  alloc->elemSize = elemSize;
  alloc->isCapacityFixed = true;
  alloc->size = 0;
  alloc->memory = memory;
  memset(alloc->items, 0, elemSize * capacity);

  return alloc;
}

static inline void expandIfNeeded(Array *array, size_t numItemsToAdd) {
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
      reallocate(array->memory, array->items, (newCapacity * array->elemSize));

  array->capacity = newCapacity;
  array->items = newItemsAllocation;
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
  expandIfNeeded(array, 1);

  PANIC_IF(array->size == array->capacity, ERR_OUT_OF_BOUNDS);

  Array_SetAt(array, array->size, ptr);
  array->size += 1;
}

void Array_AppendString(Array *array, String str) {
  Array_Append(array, (void *)&str);
}

void Array_Reset(Array *array) {
    memset(array->items, 0, array->size);
    array->size = 0;
}

Array *Array_Slice(Array *src, size_t start, size_t count) {
  PANIC_IF(src == NULL, ERR_NULL_PTR);
  PANIC_IF(start + count > src->size, ERR_OUT_OF_BOUNDS);

  Array *slice = MakeArray(src->memory, src->elemSize, count);
  memcpy(slice->items, (byte *)src->items + start * src->elemSize, count * src->elemSize);
  slice->size = count;
  return slice;
}

String Array_StringIdx(Array *array, size_t idx) {
  return *((String *)Array_Idx(array, idx));
}

// Only meant for string literals
String MakeString(Memory *memory, char *stringData) {
  size_t len = strlen(stringData);
  return MakeStringN(memory, stringData, len);
}

String MakeStringN(Memory *memory, char *stringData, size_t len) {
  String str = MakeDynamicArray(memory, sizeof(char));
  expandIfNeeded(str, len + 1);
  memcpy(str->items, stringData, len);
  ((char *)str->items)[len] = '\0';
  str->size = len;
  return str;
}

void String_AppendCString(String str, const char *cstr) {
  PANIC_IF(str == NULL, ERR_NULL_PTR);
  PANIC_IF(cstr == NULL, ERR_NULL_PTR);
  PANIC_IF(str->elemSize != sizeof(char), ARRAY_ELEM_SIZE_MISMATCH);

  size_t len = strlen(cstr);
  expandIfNeeded(str, len + 1);
  memcpy((char *)str->items + str->size, cstr, len);
  str->size += len;
  ((char *)str->items)[str->size] = '\0';
}

void String_AppendFormat(String str, const char *fmt, ...) {
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

Array_F32 Make_DynamicF32Array(Memory *memory) {
  return MakeDynamicArray(memory, sizeof(f32));
}

Array_F32 Make_F32Array(Memory *memory, size_t capacity) {
  return MakeArray(memory, sizeof(f32), capacity);
}

void Array_AppendF32(Array *array, f32 num) {
  Array_Append(array, &num);
}

void Array_AppendF32Buffer(Array *array, f32 *num, size_t numItems) {
  PANIC_IF_NULL(num);
  PANIC_IF(array->elemSize != sizeof(f32), ARRAY_ELEM_SIZE_MISMATCH);

  expandIfNeeded(array, numItems);

  f32* items = array->items;
  f32* arrEnd = items + array->size;

  memcpy(arrEnd, num, numItems * sizeof(f32));
  array->size += numItems;
}

f32 Array_F32Idx(Array *array, size_t idx) {
  return *( (f32*) Array_Idx(array, idx) );
}
