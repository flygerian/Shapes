#ifndef shapes_tensor_h
#define shapes_tensor_h

#include <stddef.h>
#include <stdint.h>
#include "../common.h"
#include "../result/result.h"

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

// Binary Ops
Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination);
Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);

// Access and shapes
Result GetAt(Tensor *t, Dim dim, Value *result);
Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value);
Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape);
Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Squeeze(Context *ctx, Tensor *t, Tensor *dest);
Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Clone(Context *ctx, Tensor *t, Tensor *dest);

// Unary
Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest);
Result Exp(Context *ctx, Tensor *t, Tensor *dest);

// Matrix ops
Result MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor *result);
Result Dot(Context *ctx, Tensor *a, Tensor *b, Tensor *result);

// Debug
void PrintItem(Tensor *t);
char *GetItem(Context *ctx, Tensor *t);

// Tensor creation
Tensor *T_Zeros(Context *ctx, Dim shape);
Tensor *T_Int(Context *ctx, Dim shape, i8 initialValues);
Tensor *T_Float(Context *ctx, Dim shape, f32 initialValues);
void SetValues(Tensor *t, Value value);

// Tensor destruction
Result FreeTensor(Context *ctx, Tensor *t);

#endif
