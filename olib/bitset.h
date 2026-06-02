#ifndef utils_lib_bitset_h
#define utils_lib_bitset_h

#include "memory.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  uint32_t bits[8];
  Memory *memory;
} Bitset;

Bitset *Make_Bitset(Memory *memory);
void Bitset_Put(Bitset *bs, unsigned char c);
bool Bitset_Contains(Bitset *bs, unsigned char c);
bool Bitset_DoesNotContain(Bitset *bs, unsigned char c);

#endif
