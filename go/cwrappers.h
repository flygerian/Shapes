#ifndef shapes_go_cwrappers_h
#define shapes_go_cwrappers_h

#include "common.h"
#include "memory.h"
#include "shapes.h"
#include <stddef.h>

Context *newContext(bool grad, size_t arenaSize);
Dim *makeDim(Memory *mem, dim_t *dims, u8 numDims);
void zeroTensorValues(Tensor *t);

Result wrap_GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor **out);
Result wrap_IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor **out);
Result wrap_IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices,
                              Tensor **out);
Result wrap_GetScalar(Tensor *t, Value *result);

double value_as_double(Value v);
u64 value_as_u64(Value v);
i64 value_as_i64(Value v);
bool value_as_bool(Value v);

Result wrap_Add(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_Divide(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_GreaterThan(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_LessThan(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_AddInPlace(Context *ctx, Tensor *a, Tensor *b);
Result wrap_SubtractInPlace(Context *ctx, Tensor *a, Tensor *b);
Result wrap_MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b);

Result wrap_Cast(Context *ctx, Tensor *src, Dtype target, Tensor **out);

Result wrap_IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad);
Result wrap_IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices,
                              Tensor *srcGrad);
Result wrap_SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad);

Result wrap_Pow(Context *ctx, Tensor *t, f32 power, Tensor **out);
Result wrap_Exp(Context *ctx, Tensor *t, Tensor **out);
Result wrap_Tanh(Context *ctx, Tensor *t, Tensor **out);
Result wrap_Negate(Context *ctx, Tensor *t, Tensor **out);
Result wrap_MeanWithDim(Context *ctx, Tensor *t, dim_t dim, Tensor **out);
Result wrap_Log(Context *ctx, Tensor *t, Tensor **out);
Result wrap_Abs(Context *ctx, Tensor *t, Tensor **out);
Result wrap_CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits, Tensor **loss,
                                Tensor **probs);
Result wrap_CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut,
                                 Tensor **dLogits);

Result wrap_Sum(Context *ctx, Tensor *t, dim_t dim, Tensor **out);
Result wrap_Mean(Context *ctx, Tensor *t, dim_t dim, Tensor **out);
Result wrap_Std(Context *ctx, Tensor *t, Tensor **out);
Result wrap_Max(Context *ctx, Tensor *t, dim_t dim, Tensor **out);
Result wrap_ArgMax(Context *ctx, Tensor *t, dim_t dim, Tensor **out);

Tensor *wrap_T_Zeros(Context *ctx, Dim *shape);
Tensor *wrap_T_Int(Context *ctx, Dim *shape, i8 value);
Tensor *wrap_T_Float(Context *ctx, Dim *shape, f32 value);
Result wrap_Clone(Context *ctx, Tensor *src, Tensor **out);
Result wrap_Copy(Context *ctx, Tensor *src, Tensor *dest);
Tensor *wrap_T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses);
Tensor *wrap_T_Arange(Context *ctx, f32 start, f32 end, f32 step);

Result wrap_MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_Dot(Context *ctx, Tensor *a, Tensor *b, Tensor **out);
Result wrap_DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor **out);
Result wrap_DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor **dX,
                          Tensor **dW, Tensor **dB);
Result wrap_BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta,
                                     f32 epsilon, Tensor **out, Tensor **mean, Tensor **variance);
Result wrap_BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon,
                              Tensor **dX, Tensor **dGamma, Tensor **dBeta);

Result wrap_Slice1(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice2(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice3(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice4(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice5(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice6(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice7(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Slice8(Context *ctx, Tensor *src, Tensor **out, Range *r);
Result wrap_Reshape(Context *ctx, Tensor *src, Tensor **out, Dim *newShape);
Result wrap_Transpose(Context *ctx, Tensor *src, Tensor **out, dim_t d0, dim_t d1);
Result wrap_Permute(Context *ctx, Tensor *src, Tensor **out, Dim *order);
Result wrap_Squeeze(Context *ctx, Tensor *src, Tensor **out);
Result wrap_SqueezeDim(Context *ctx, Tensor *src, Tensor **out, dim_t d);
Result wrap_UnSqueeze(Context *ctx, Tensor *src, Tensor **out, dim_t d);

#endif
