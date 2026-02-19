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
Result AddInPlace(Context *ctx, Tensor *a, Tensor *b);

// Access and shapes
Result GetAt(Tensor *t, Dim dim, Value *result);
Result GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor *dest);
Result GetScalar(Tensor *t, Value *result);
Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value);
Result IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor *dest);
Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape);
Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Squeeze(Context *ctx, Tensor *t, Tensor *dest);
Result SqueezeDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Clone(Context *ctx, Tensor *t, Tensor *dest);

// Unary
Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest);
Result Exp(Context *ctx, Tensor *t, Tensor *dest);
Result Negate(Context *ctx, Tensor *t, Tensor *dest);

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
Tensor *T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses);
void SetValues(Tensor *t, Value value);

// Tensor destruction
Result FreeTensor(Context *ctx, Tensor *t);
Result FreeViewTensor(Context *ctx, Tensor *t);

#endif
