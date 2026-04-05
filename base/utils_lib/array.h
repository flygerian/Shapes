#ifndef utils_lib_array_h
#define utils_lib_array_h

#include "memory.h"

typedef struct Array {
  void *items;
  size_t capacity;
  size_t size;
  size_t elemSize;
  bool isCapacityFixed;
  Memory *memory;
} Array;

Array *MakeDynamicArray(Memory *memory, const size_t elemSize);
Array *MakeArray(Memory *memory, const size_t elemSize, size_t capacity);
void *Array_Idx(Array *slice, size_t idx);
void Array_SetAt(Array *array, size_t idx, void *ptr);
void Array_Append(Array *array, void *ptr);

#endif
