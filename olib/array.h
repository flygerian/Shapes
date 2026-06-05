#ifndef utils_lib_array_h
#define utils_lib_array_h

#include "memory.h"
#include "result.h"
#include "olib.h"

#define ARRAY_PTR_AT_IDX(slice, idx) ((byte *)(slice)->items + ((idx) * (slice)->elemSize))

typedef struct olib_Array {
  void *items;
  size_t capacity;
  size_t size;
  size_t elemSize;
  bool isCapacityFixed;
  olib_Memory *memory;
} olib_Array;

typedef struct olib_ArrayPair {
  olib_Array *a;
  olib_Array *b;
} olib_ArrayPair;

#define ARRAY_PAIR(aArr, bArr)  (olib_ArrayPair) {.a = ( aArr ), .b = ( bArr )}

typedef olib_Array *olib_String;
typedef olib_Array *olib_ArrayF32;

#define STR(stringStruct) ((char*)( stringStruct )->items)

olib_Array *olib_MakeDynamicArray(olib_Memory *memory, const size_t elemSize);
olib_Array *olib_MakeArray(olib_Memory *memory, const size_t elemSize, size_t capacity);

// declared in the header file so it can be inlined at the call site
static inline void *olib_ArrayIdx(olib_Array *slice, size_t idx) {
  PANIC_IF(slice == NULL, ERR_NULL_PTR);
  PANIC_IF(idx > slice->capacity - 1, ERR_OUT_OF_BOUNDS);

  if (idx >= slice->capacity) {
    return NULL;
  }

  return ARRAY_PTR_AT_IDX(slice, idx);
}

void olib_ArraySetAt(olib_Array *array, size_t idx, void *ptr);
void olib_ArrayAppend(olib_Array *array, void *ptr);
void olib_ArrayReset(olib_Array *array);
void olib_ArrayAppendStructPtr(olib_Array *array, void *ptr);
void *olib_ArrayStructPtrIdx(olib_Array *array, size_t idx);
olib_Array *olib_ArraySlice(olib_Array *src, size_t start, size_t count);

olib_String olib_MakeString(olib_Memory *memory, char *stringData);
olib_String olib_MakeStringN(olib_Memory *memory, char *stringData, size_t len);
void olib_ArrayAppendString(olib_Array *array, olib_String str);
olib_String olib_ArrayStringIdx(olib_Array *array, size_t idx);

void olib_StringAppendCString(olib_String str, const char *cstr);
void olib_StringAppendFormat(olib_String str, const char *fmt, ...);

olib_ArrayF32 olib_MakeDynamicF32Array(olib_Memory *memory);
olib_ArrayF32 olib_MakeF32Array(olib_Memory *memory, size_t capacity);
void olib_ArrayAppendF32(olib_Array *array, f32 num);
void olib_ArrayAppendF32Buffer(olib_Array *array, f32 *num, size_t numItems);
f32 olib_ArrayF32Idx(olib_Array *array, size_t idx);

#endif
