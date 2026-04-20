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

typedef Array *String;

Array *MakeDynamicArray(Memory *memory, const size_t elemSize);
Array *MakeArray(Memory *memory, const size_t elemSize, size_t capacity);
void *Array_Idx(Array *slice, size_t idx);
void Array_SetAt(Array *array, size_t idx, void *ptr);
void Array_Append(Array *array, void *ptr);
void Array_AppendStructPtr(Array *array, void *ptr);
void *Array_StructPtrIdx(Array *array, size_t idx);

String MakeString(Memory *memory, char *stringData);
String MakeStringN(Memory *memory, char *stringData, size_t len);
void Array_AppendString(Array *array, String str);
String Array_StringIdx(Array *array, size_t idx);

#endif
