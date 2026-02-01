#ifndef shapes_tensor_h
#define shapes_tensor_h

#include <stddef.h>
#include <stdint.h>
#include "../common.h"
#include "../result/result.h"
#include "value.h"

#define MAX_SUM_N_DIMS 2
#define MAX_PARALLEL_SUMS 4

typedef u64 tensor_size_t;
typedef u32 dim_t;
typedef u8 multiplier_t;

typedef struct {
  dim_t *dims;
  u8 numOfDims;

  u8 *multipliers;
} Dim;

typedef struct {
  u64 start;
  u64 end;
} Range;

typedef struct {
  Dtype dtype;
  void *values;
  tensor_size_t size;
  Dim shape;
  bool isView;
  bool isContigous;
  Range *boundary;
} Tensor;

// Binary Ops
Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination);
Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination); 

// Access and shapes
Result GetAt(Tensor *t, Dim dim, Value *result);
Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value); 
Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape);\
Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Squeeze(Context *ctx, Tensor *t, Tensor *dest);
Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Clone(Context *ctx, Tensor *t, Tensor *dest);

// Unary
Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);

// Matrix ops
Result MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor *result);
Result Dot(Context *ctx, Tensor *a, Tensor *b, Tensor *result);

// Tensor creation
Tensor* T_Zeros(Context *ctx, Dim shape);

// Tensor destruction
Result FreeTensor(Context *ctx, Tensor *t);

#endif
