#ifndef shapes_common_h
#define shapes_common_h

#include "memory.h"
#include "result/result.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
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

typedef enum { F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, BOOL } Dtype;

typedef enum {
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_GREATER,
  OP_GREATER_OR_EQUAL,
  OP_LESS,
  OP_LESS_OR_EQUAL
} OpType;

typedef enum {
  UNARY_OP_POW,
  UNARY_OP_TANH,
  UNARY_OP_RELU,
  UNARY_OP_NEGATE,
  UNARY_OP_EXP,
  UNARY_OP_LOG,
  UNARY_OP_ABS
} UnaryOpType;

typedef enum {
  REDUCTION_OP_SUM,
  REDUCTION_OP_MEAN,
  REDUCTION_OP_MAX,
  REDUCTION_OP_ARGMAX
} ReductionOpType;

typedef enum { CPU, CUDA } DeviceType;

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

typedef struct CudaCachedBlock {
  void *ptr;
  size_t size;
  struct CudaCachedBlock *next;
} CudaCachedBlock;

typedef struct {
  DeviceType type;
  char *id;
  CudaCachedBlock *activeBlocks;
  CudaCachedBlock *cachedBlocks;
} Device;

typedef struct Context {
  Memory *memory;
  Device *device;
  cublasHandle_t handle;
} Context;

typedef struct {
  Context *context;
  Memory *metadataMemory;
  void *values;
  Range *boundary;
  tensor_size_t size;
  Dim shape;

  Dtype dtype;
  bool isView;
  bool isContigous;
  bool isContigousCopy;
} Tensor;

typedef struct {
  Tensor *param;
  Tensor *paramGrad;
  Tensor *m;
  Tensor *v;
} AdamData;


size_t getBytesForDtype(Dtype type);

#define DIM_ZERO ((Dim){.dims = NULL, .numOfDims = 0})
#define DIM1D(dimSize) ((Dim){.dims = (dim_t[]){dimSize}, .numOfDims = 1})
#define DIM2D(dim0Size, dim1Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define DIM3D(dim0Size, dim1Size, dim2Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define DIM4D(dim0Size, dim1Size, dim2Size, dim3Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})
#endif
