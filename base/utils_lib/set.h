#ifndef utils_lib_set_h
#define utils_lib_set_h

#include "memory.h"

typedef struct {
  void **entries;
  size_t capacity;
  size_t count;
  Memory *memory;
} PtrSet;

PtrSet *MakePtrSet(Memory *memory);
void PtrSet_Put(PtrSet *pt, void *key);
bool PtrSet_Contains(PtrSet *pt, void *key);

#endif
