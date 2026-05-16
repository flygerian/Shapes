#ifndef shapes_h
#define shapes_h

#include "result/result.h"
#include "tensor/types.h"
#include "utils_lib/array.h"
#include <stddef.h>
#include <stdint.h>

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

// Arrays
typedef Array *Array_Tensor;
Array_Tensor Make_DynamicTensorArray(Memory *memory);
Array *Make_TensorArray(Memory *memory, size_t capacity);
void Array_AppendTensor(Array *array, Tensor *tensor);
void Array_AppendTensorArray(Array *array, Array *tensorArray);
static inline Tensor *Array_TensorIdx(Array *array, size_t idx) {
  return *((Tensor **)Array_Idx(array, idx));
}

// Context
Context InitializeHostContext(size_t arenaSize, size_t minBlockSize);
Context InitializeCudaContext(size_t hostArenaSize);
Context GetScratchContext(Context *ctx, size_t bufferSize);
void DestroyContext(Context *ctx);
void FreeContext(Context *ctx);
Result Flush(Context *ctx);
Result CopyBetweenDevices(DeviceType srcType, DeviceType destType, void *restrict srcPtr, void *restrict destPtr, size_t size);
void MoveToCuda(Context *destCtx, Array *tensors);

// Binary Ops
Tensor *Add(Context *ctx, Tensor *a, Tensor *b);
Tensor *Subtract(Context *ctx, Tensor *a, Tensor *b);
Tensor *Divide(Context *ctx, Tensor *numerator, Tensor *denominator);
Tensor *Multiply(Context *ctx, Tensor *a, Tensor *b);
Tensor *GreaterThan(Context *ctx, Tensor *a, Tensor *b);
Tensor *GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b);
Tensor *Equal(Context *ctx, Tensor *a, Tensor *b);
Tensor *LessThan(Context *ctx, Tensor *a, Tensor *b);
Tensor *LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b);

void AddInPlace(Context *ctx, Tensor *a, Tensor *b);
void SubtractInPlace(Context *ctx, Tensor *a, Tensor *b);
void MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b);

// Access and shapes
Value *GetAt(Tensor *t, Dim dim);
Result CopyShape(Tensor *t, dim_t *destDims, u8 *numDims);
Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value);
Tensor *IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices);
Tensor *IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices);
Tensor *Slice(Context *ctx, Tensor *source, ...);
Tensor *Reshape(Context *ctx, Tensor *source, Dim newShape);
void ReshapeBackward(Context *ctx, Tensor *node);
Tensor *Transpose(Context *ctx, Tensor *source, ...);
Tensor *Permute(Context *ctx, Tensor *source, Dim order);
Tensor *Squeeze(Context *ctx, Tensor *t);
Tensor *SqueezeDim(Context *ctx, Tensor *t, dim_t dim);
Tensor *UnSqueeze(Context *ctx, Tensor *t, dim_t dim);
Tensor *Clone(Context *ctx, Tensor *t);
void Copy(Context *ctx, Tensor *src, Tensor *dest);
Tensor *Concat(Context *ctx, Tensor *target, dim_t targetDim, Tensor **tensors, u32 numTensorsToAdd);
Tensor *Stack(Context *ctx, Array_Tensor tensors);

// Cast
Tensor *Cast(Context *ctx, Tensor *source, Dtype targetDtype);

// Unary
Tensor *Pow(Context *ctx, Tensor *t, f32 power);
Tensor *Exp(Context *ctx, Tensor *t);
Tensor *Tanh(Context *ctx, Tensor *t);
Tensor *Relu(Context *ctx, Tensor *t);
Tensor *ReluBackward(Context *ctx, Tensor *output, Tensor *gradOut);
void ReluBackwardAccumulate(Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest);
Tensor *Negate(Context *ctx, Tensor *t);
Tensor *Log(Context *ctx, Tensor *t);
Tensor *Abs(Context *ctx, Tensor *t);
Tensor *Sqrt(Context *ctx, Tensor *t);
void SqrtBackward(Context *ctx, Tensor *node);

// Reduction
Tensor *Sum(Context *ctx, Tensor *t, dim_t dim);
Tensor *ReduceBroadcast(Context *ctx, Tensor *input, Tensor *grad);
Tensor *Mean(Context *ctx, Tensor *t);
Tensor *MeanDim(Context *ctx, Tensor *t, dim_t dim);
Tensor *Std(Context *ctx, Tensor *t);
Tensor *Max(Context *ctx, Tensor *t, dim_t dim);
Tensor *ArgMax(Context *ctx, Tensor *t, dim_t dim);

// Accumulate
void IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad);
void IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad);
void SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad);

// Matrix ops
Tensor *MatMul(Context *ctx, Tensor *a, Tensor *b);
Tensor *Dot(Context *ctx, Tensor *a, Tensor *b);

// Layer ops
Tensor *DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias);
Result DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW, Tensor *dB);
BatchNormFowardResult BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon);
BatchNormBackwardResult BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon);
Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Tensor *bias, bool withBias, Tensor *t, Tensor *dest,
              Tensor *colBuffer);
Result Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels, Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer,
                      Tensor *dBias, bool withBias, u8 stride);
Result ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Dim kernel, Tensor *t, Tensor *dest);
Result ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride, Tensor *dX, Tensor *dKernels);
Result MaxPool2d(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest);
Result MaxPool2dWithIndices(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest, Tensor *indices);
Result MaxPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, Dim kernel, u8 stride, Tensor *dX);
Result MaxPool2dBackwardWithIndices(Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices, Tensor *dX);
Result AdaptiveAvgPool2d(Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor *dest);
Result AdaptiveAvgPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH, dim_t outW, Tensor *dX);

// Loss ops
TensorPair CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits);
Tensor *CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut);

// Optimizer ops
Result Sgd(Context *ctx, Array *parameters, f32 learningRate);

Result Adam(Context *ctx, AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon);

// Debug
void PrintItem(Tensor *t);
char *GetItem(Context *ctx, Tensor *t);
void PrintTensor(Tensor *tensor);

// Tensor creation
Tensor *T_Zeros(Context *ctx, Dim shape);
Tensor *T_Int(Context *ctx, Dim shape, i8 initialValues);
Tensor *T_UInt(Context *ctx, Dim shape, u8 initialValue);
Tensor *T_Float(Context *ctx, Dim shape, f32 initialValues);
Tensor *T_Float64(Context *ctx, Dim shape, f64 initialValue);
Tensor *MakeFromContigousArray(Context *ctx, Dim shape, void *values, Dtype dtype);
Tensor *MakeRandomTensor(Context *ctx, Dim shape, f32 minValue, f32 maxValue, Dtype dtype);
Tensor *T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses);
Tensor *T_Arange(Context *ctx, f32 start, f32 end, f32 step);
void SetValues(Tensor *t, Value value);

#endif
