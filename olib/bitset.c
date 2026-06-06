#include "bitset.h"
#include <stdint.h>

Bitset *Make_Bitset(olib_Memory *memory) {
  Bitset *bs = olib_Allocate(memory, sizeof(Bitset));
  bs->memory = memory;
  for (int i = 0; i < 8; i++) {
    bs->bits[i] = 0;
  }
  return bs;
}

void Bitset_Put(Bitset *bs, unsigned char c) {
  bs->bits[c >> 5] |= (1u << (c & 31));
}

bool Bitset_Contains(Bitset *bs, unsigned char c) {
  return (bs->bits[c >> 5] & (1u << (c & 31))) != 0;
}

bool Bitset_DoesNotContain(Bitset *bs, unsigned char c) {
  return !Bitset_Contains(bs, c);
}
