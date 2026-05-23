#ifndef utils_lib_array_h
#define utils_lib_array_h

#include "memory.h"
#include "result/result.h"
#include "utils_lib.h"

#define ARRAY_PTR_AT_IDX(slice, idx) ((byte *)(slice)->items + ((idx) * (slice)->elemSize))

typedef struct Array {
  void *items;
  size_t capacity;
  size_t size;
  size_t elemSize;
  bool isCapacityFixed;
  Memory *memory;
} Array;

typedef struct ArrayPair {
  Array *a;
  Array *b;
} ArrayPair;

#define ARRAY_PAIR(aArr, bArr)  (ArrayPair) {.a = ( aArr ), .b = ( bArr )}

typedef Array *String;
typedef Array *Array_F32;


#define STR(stringStruct) ((char*)( stringStruct )->items)

Array *MakeDynamicArray(Memory *memory, const size_t elemSize);
Array *MakeArray(Memory *memory, const size_t elemSize, size_t capacity);

// declared in the header file so it can be inlined at the call site
static inline void *Array_Idx(Array *slice, size_t idx) {
  PANIC_IF(slice == NULL, ERR_NULL_PTR);
  PANIC_IF(idx > slice->capacity - 1, ERR_OUT_OF_BOUNDS);

  if (idx >= slice->capacity) {
    return NULL;
  }

  return ARRAY_PTR_AT_IDX(slice, idx);
}

void Array_SetAt(Array *array, size_t idx, void *ptr);
void Array_Append(Array *array, void *ptr);
void Array_Reset(Array *array);
void Array_AppendStructPtr(Array *array, void *ptr);
void *Array_StructPtrIdx(Array *array, size_t idx);

String MakeString(Memory *memory, char *stringData);
String MakeStringN(Memory *memory, char *stringData, size_t len);
void Array_AppendString(Array *array, String str);
String Array_StringIdx(Array *array, size_t idx);

Array_F32 Make_DynamicF32Array(Memory *memory);
Array_F32 Make_F32Array(Memory *memory, size_t capacity);
void Array_AppendF32(Array *array, f32 num);
void Array_AppendF32Buffer(Array *array, f32 *num, size_t numItems);
f32 Array_F32Idx(Array *array, size_t idx);

#endif
