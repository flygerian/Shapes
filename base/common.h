#ifndef shapes_common_h
#define shapes_common_h

#include "memory.h"
#include "result/result.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <termios.h>
#include <time.h>

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

typedef size_t tensor_size_t;
typedef size_t dim_t;
typedef u8 multiplier_t;

typedef struct {
  dim_t *dims;
  u8 *multipliers;
  u8 numOfDims;
} Dim;

typedef struct {
  size_t start;
  size_t end;
} Range;

typedef enum { F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64 } Dtype;

typedef struct {
  Dtype dtype;
  union {
    u8 u8;
    u16 u16;
    u32 u32;
    u64 u64;

    i8 i8;
    i16 i16;
    i32 i32;
    i64 i64;

    f16 f16;
    f32 f32;
    f64 f64;
  } as;
} Value;

typedef struct {
  void *values;
  Range *boundary;
  tensor_size_t size;
  Dim shape;

  Dtype dtype;
  bool isView;
  bool isContigous;
} Tensor;

typedef struct Context {
  Memory *memory;
} Context;

typedef enum { 
  OP_ADD, 
  OP_SUBTRACT, 
  OP_MULTIPLY, 
  OP_GREATER, 
  OP_GREATER_OR_EQUAL, 
  OP_LESS, 
  OP_LESS_OR_EQUAL 
} OpType;

size_t getBytesForDtype(Dtype type);

#endif
