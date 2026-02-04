#ifndef shapes_common_h
#define shapes_common_h

#include "memory.h"
#include "result/result.h"
#include <stdbool.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

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

typedef enum {
  F16, F32, F64, U8, U16, U32, U64
} Dtype;

typedef struct {
  Dtype dtype;
  union {
    u8 u8;
    u16 u16;
    u32 u32;
    u64 u64;
    float f16;
    float f32;
    double f64;
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
} Tensor;

typedef struct Context {
 Memory *memory;  
 bool grad;
 struct GraphNode *computationGraph;
} Context;

typedef Result (*BackwardFn)(struct Context*, struct GraphNode*);

typedef enum {
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE
} OpType;

typedef struct GraphNode {
  Tensor *output;
  Tensor *grad;
  Tensor **inputs;
  u8 numInputs;
  BackwardFn backward;
  OpType optype;
} GraphNode;


size_t getBytesForDtype(Dtype type);

#endif
