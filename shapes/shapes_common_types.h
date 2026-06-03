#ifndef shapes_common_types_h
#define shapes_common_types_h

typedef enum {
  OP_NONE,
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_GREATER,
  OP_GREATER_OR_EQUAL,
  OP_LESS,
  OP_LESS_OR_EQUAL,
  OP_EQUAL,
  OP_DENSE,
  OP_EMBEDDING,
  OP_RESHAPE,
  OP_MSE,
  OP_CROSS_ENTHROPY,
  OP_SGD,
  OP_ADAM,
  OP_BATCH_NORM,
  OP_TANH,
  OP_SQRT,
  OP_RELU,
  OP_MAXPOOL2D,
  OP_ADAPTIVE_AVG_POOL2D,
  OP_CONV2D,
  OP_SEQUENTIAL,
  OP_FLATTEN
} OpType;

typedef enum {
  REDUCTION_OP_SUM,
  REDUCTION_OP_MEAN,
  REDUCTION_OP_MAX,
  REDUCTION_OP_ARGMAX
} ReductionOpType;

typedef enum {
  UNARY_OP_POW,
  UNARY_OP_TANH,
  UNARY_OP_RELU,
  UNARY_OP_NEGATE,
  UNARY_OP_EXP,
  UNARY_OP_LOG,
  UNARY_OP_ABS,
  UNARY_OP_SQRT
} UnaryOpType;

typedef enum { F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, BOOL } Dtype;

typedef struct {
  Dtype dtype;
  union {
    bool boolean;
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
  size_t start;
  size_t end;
} Range;
#endif
