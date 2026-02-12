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

typedef u64 tensor_size_t;
typedef u32 dim_t;
typedef u8 multiplier_t;

struct GraphNode;

typedef struct {
  dim_t *dims;
  u8 numOfDims;

  u8 *multipliers;
} Dim;

typedef struct {
  u64 start;
  u64 end;
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
  Dtype dtype;
  void *values;
  tensor_size_t size;
  Dim shape;
  bool isView;
  bool isContigous;
  Range *boundary;
  struct GraphNode *computation;
  char *label;
} Tensor;

typedef struct ScreenConfig {
  int cx, cy;
  int rx;
  int screenrows;
  int screencols;
  int numrows;
  int rowoff;
  int coloff;
  struct termios *orig_termios;
} ScreenConfig;

typedef struct Context {
  Memory *memory;
  bool grad;
  ScreenConfig *screenConfig;
} Context;

typedef Result (*BackwardFn)(struct Context *, struct GraphNode *);

typedef enum { OP_ADD, OP_SUBTRACT, OP_MULTIPLY, OP_TANH, OP_POW } OpType;

typedef struct GraphNode {
  Tensor *output;
  Tensor *grad;
  Tensor **inputs;
  u8 numInputs;
  BackwardFn backward;
  OpType optype;
  void *metadata;  // Operation-specific data (e.g., power value for Pow)
} GraphNode;

size_t getBytesForDtype(Dtype type);


Context NoGradContext(Context *ctx);

#endif
