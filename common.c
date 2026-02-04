#include "common.h"

size_t getBytesForDtype(Dtype type) {
  switch(type) {
    case U8:  return sizeof(u8);
    case U16: return sizeof(u16);
    case U32: return sizeof(u32);
    case U64: return sizeof(u64);
    case F16:
    case F32: return sizeof(float);
    case F64: return sizeof(double);
    default:  return 0;
  }
}
