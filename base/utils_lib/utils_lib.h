
#include <math.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float_t f16;
typedef float_t f32;
typedef double_t f64;

typedef u8 byte;
typedef byte* bytebuffer;
typedef char* string;

#define KB ((size_t)1024)
#define MB ((size_t)1024 * KB)
#define GB ((size_t)1024 * MB)

#define RANGE(iterator, bufferSize) \
  size_t iterator = 0; iterator < (size_t) bufferSize; iterator++

#define RANGE_FROM(from, bufferSize, iterator) \
  size_t iterator = from; iterator < (size_t) bufferSize; iterator++

#define F32_(var) (f32) var
