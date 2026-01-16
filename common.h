#ifndef shapes_common_h
#define shapes_common_h

#include "memory.h"
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;


typedef struct {
 Memory *memory;  
} Context;

#endif
