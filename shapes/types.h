#ifndef shapes_types_h
#define shapes_types_h

#include "cuda_compat.h"
#include <stdbool.h>
#include <stddef.h>
#include "array.h"
#include "memory.h"
#include "olib.h"
#include "shapes_internal.h"

typedef size_t tensor_size_t;
typedef size_t dim_t;
typedef size_t multiplier_t;

typedef struct {
  dim_t *dims;
  multiplier_t *multipliers;
  u8 numOfDims;
} Dim;

typedef struct {
  size_t start;
  size_t end;
} Range;

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
  UNARY_OP_POW,
  UNARY_OP_TANH,
  UNARY_OP_RELU,
  UNARY_OP_NEGATE,
  UNARY_OP_EXP,
  UNARY_OP_LOG,
  UNARY_OP_ABS,
  UNARY_OP_SQRT
} UnaryOpType;

typedef enum {
  REDUCTION_OP_SUM,
  REDUCTION_OP_MEAN,
  REDUCTION_OP_MAX,
  REDUCTION_OP_ARGMAX
} ReductionOpType;


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

typedef struct ValuePair {
  Value a;
  Value b;
} ValuePair;



typedef struct sizeAndMultipliers {
  tensor_size_t size;
  multiplier_t *multipliers;
} sizeAndMultipliers;

typedef struct Tensor {
  Context *context;
  string label;
  Memory *metadataMemory;
  void *values;
  Range *boundary;
  tensor_size_t size;
  Dim shape;
  Dtype dtype;
  bool isView;
  bool isContigous;
  bool isContigousCopy;
  struct Tensor *grad;
  Array *inputs;
  OpType opType;
  void *opMetadata;
  u64 nodeId;
} Tensor;

typedef struct {
  void *param;
  void *grad;
  void *m;
  void *v;
  tensor_size_t size;
  Dtype dtype;
} AdamData;

typedef struct BatchNormFowardResult {
  Tensor out;
  Tensor mean;
  Tensor variance;
} BatchNormFowardResult;

typedef struct BatchNormBackwardResult {
  Tensor dx2d;
  Tensor dGamma;
  Tensor dBeta;
} BatchNormBackwardResult;

typedef struct {
  Tensor a;
  Tensor b;
} TensorPair;

size_t getBytesForDtype(Dtype type);

#endif
