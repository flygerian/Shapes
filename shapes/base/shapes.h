#ifndef shapes_h
#define shapes_h

#include "result.h"
#include "types.h"
#include "array.h"
#include <stddef.h>
#include <stdint.h>

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

#define SHAPE(dimensions, numberOfDimensions)           ((Dim){.dims = (dimensions), .numOfDims = (numberOfDimensions)})
#define SCALAR                                          ((Dim){.dims = (dim_t[]){1}, .numOfDims = 1})
#define SHAPE1D(dimSize)                                ((Dim){.dims = (dim_t[]){dimSize}, .numOfDims = 1})
#define SHAPE2D(dim0Size, dim1Size)                     ((Dim){.dims = (dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define SHAPE3D(dim0Size, dim1Size, dim2Size)           ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define SHAPE4D(dim0Size, dim1Size, dim2Size, dim3Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})

// Arrays
typedef olib_Array *shapes_ArrayTensor;
shapes_ArrayTensor shapes_Make_DynamicTensorArray(olib_Memory *memory);
olib_Array *shapes_MakeTensorArray(olib_Memory *memory, size_t capacity);
void shapes_ArrayAppendTensor(olib_Array *array, Tensor *tensor);
void shapes_ArrayAppendTensorArray(olib_Array *array, olib_Array *tensorArray);
static inline Tensor shapes_ArrayTensorIdx(olib_Array *array, size_t idx) {
  return *((Tensor *)olib_ArrayIdx(array, idx));
}

static inline Tensor* shapes_ArrayTensorPtrIdx(olib_Array *array, size_t idx) {
  return (Tensor *)olib_ArrayIdx(array, idx);
}

// shapes_Context
shapes_Context shapes_InitializeHostContext(size_t arenaSize, size_t minBlockSize);
shapes_Context shapes_InitializeCudaContext(size_t hostArenaSize);
shapes_Context shapes_GetScratchContext(shapes_Context *ctx, size_t bufferSize);
void shapes_DestroyContext(shapes_Context *ctx);
void shapes_FreeContext(shapes_Context *ctx);
Result shapes_Flush(shapes_Context *ctx);

Result shapes_CopyBetweenDevices(DeviceType srcType, DeviceType destType, void *restrict srcPtr, void *restrict destPtr, size_t size);

void shapes_MoveToCuda(shapes_Context *destCtx, shapes_ArrayTensor tensors);
void shapes_MoveToHost(shapes_Context *destCtx, shapes_ArrayTensor tensors);
void shapes_MoveTensorToHost(shapes_Context *destCtx, Tensor *t);

// Binary Ops
Tensor shapes_Add(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Subtract(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Divide(shapes_Context *ctx, Tensor *numerator, Tensor *denominator);
Tensor shapes_Multiply(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_GreaterThan(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_GreaterThanOrEqual(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Equal(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_LessThan(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_LessThanOrEqual(shapes_Context *ctx, Tensor *a, Tensor *b);

void shapes_AddInPlace(shapes_Context *ctx, Tensor *a, Tensor *b);
void shapes_SubtractInPlace(shapes_Context *ctx, Tensor *a, Tensor *b);
void shapes_MultiplyInPlace(shapes_Context *ctx, Tensor *a, Tensor *b);

// Access and shapes
shapes_Value *shapes_GetAt(Tensor *t, Dim dim);
Result shapes_CopyShape(Tensor *t, dim_t *destDims, u8 *numDims);
Result shapes_AssignValueAt(shapes_Context *ctx, Tensor *t, Dim dim, shapes_Value value);
Tensor shapes_IndexWithTensor(shapes_Context *ctx, Tensor *source, Tensor *indices);
Tensor shapes_IndexWithTensor2d(shapes_Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices);
Tensor shapes_Slice(shapes_Context *ctx, Tensor *source, ...);
Tensor shapes_Reshape(shapes_Context *ctx, Tensor *source, Dim newShape);
void   shapes_ReshapeBackward(shapes_Context *ctx, Tensor *node);
Tensor shapes_Transpose(shapes_Context *ctx, Tensor *source, ...);
Tensor shapes_Permute(shapes_Context *ctx, Tensor *source, Dim order);
Tensor shapes_Squeeze(shapes_Context *ctx, Tensor *t);
Tensor shapes_SqueezeDim(shapes_Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_UnSqueeze(shapes_Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_Clone(shapes_Context *ctx, Tensor *t);
void   shapes_Copy(shapes_Context *ctx, Tensor *src, Tensor *dest);
Tensor shapes_Concat(shapes_Context *ctx, Tensor *target, dim_t targetDim, shapes_ArrayTensor tensors);
Tensor shapes_Stack(shapes_Context *ctx, shapes_ArrayTensor tensors);

// Cast
Tensor Cast(shapes_Context *ctx, Tensor *source, shapes_Dtype targetDtype);

// Unary
Tensor shapes_Pow(shapes_Context *ctx, Tensor *t, f32 power);
Tensor shapes_Exp(shapes_Context *ctx, Tensor *t);
Tensor shapes_Tanh(shapes_Context *ctx, Tensor *t);
Tensor shapes_Relu(shapes_Context *ctx, Tensor *t);
Tensor shapes_ReluBackward(shapes_Context *ctx, Tensor *output, Tensor *gradOut);
void   shapes_ReluBackwardAccumulate(shapes_Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest);
Tensor shapes_Negate(shapes_Context *ctx, Tensor *t);
Tensor shapes_Log(shapes_Context *ctx, Tensor *t);
Tensor shapes_Abs(shapes_Context *ctx, Tensor *t);
Tensor shapes_Sqrt(shapes_Context *ctx, Tensor *t);
void   shapes_SqrtBackward(shapes_Context *ctx, Tensor *tensor);

// Reduction
Tensor shapes_Sum(shapes_Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_ReduceBroadcast(shapes_Context *ctx, Tensor *input, Tensor *grad);
Tensor shapes_Mean(shapes_Context *ctx, Tensor *t);
Tensor shapes_MeanDim(shapes_Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_Std(shapes_Context *ctx, Tensor *t);
Tensor shapes_Max(shapes_Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_ArgMax(shapes_Context *ctx, Tensor *t, dim_t dim);

// Accumulate
void shapes_IndexAccumulate1d(shapes_Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad);
void shapes_IndexAccumulate2d(shapes_Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad);
void shapes_SliceAccumulate(shapes_Context *ctx, Tensor *dest, shapes_Range *ranges, Tensor *srcGrad);

// Matrix ops
Tensor shapes_MatMul(shapes_Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Dot(shapes_Context *ctx, Tensor *a, Tensor *b);

// shapesnn_layer ops
Tensor shapes_DenseLinear(shapes_Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias);
Result shapes_DenseBackward(shapes_Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW, Tensor *dB);
BatchNormFowardResult shapes_BatchNormForwardTraining(shapes_Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon);
BatchNormBackwardResult shapes_BatchNormBackward(shapes_Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon);
Result shapes_Conv2d(shapes_Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Tensor *bias, bool withBias, Tensor *t, Tensor *dest,
              Tensor *colBuffer);
Result shapes_Conv2dBackward(shapes_Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels, Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer,
                      Tensor *dBias, bool withBias, u8 stride);
Result shapes_ConvTranspose2d(shapes_Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Dim kernel, Tensor *t, Tensor *dest);
Result shapes_ConvTranspose2dBackward(shapes_Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride, Tensor *dX, Tensor *dKernels);
Result shapes_MaxPool2d(shapes_Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest);
Result shapes_MaxPool2dWithIndices(shapes_Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest, Tensor *indices);
Result shapes_MaxPool2dBackward(shapes_Context *ctx, Tensor *x, Tensor *gradOut, Dim kernel, u8 stride, Tensor *dX);
Result shapes_MaxPool2dBackwardWithIndices(shapes_Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices, Tensor *dX);
Result shapes_AdaptiveAvgPool2d(shapes_Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor *dest);
Result shapes_AdaptiveAvgPool2dBackward(shapes_Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH, dim_t outW, Tensor *dX);

// Loss ops
TensorPair shapes_loss_CrossEntropyForward(shapes_Context *ctx, Tensor *yGround, Tensor *logits);
Tensor shapes_loss_CrossEntropyBackward(shapes_Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut);

// Optimizer ops
Result shapes_optimizer_Sgd(shapes_Context *ctx, olib_Array *parameters, f32 learningRate);

Result shapes_optimizer_Adam(shapes_Context *ctx, AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon);

// Debug
void shapes_PrintItem(Tensor *t);
char *shapes_GetItem(shapes_Context *ctx, Tensor *t);
void shapes_PrintTensor(Tensor *tensor);

// Tensor creation
Tensor shapes_MakeZerosTensor(shapes_Context *ctx, Dim shape);
Tensor shapes_MakeIntTensor(shapes_Context *ctx, Dim shape, i8 initialValues);
Tensor shapes_MakeUIntTensor(shapes_Context *ctx, Dim shape, u8 initialValue);
Tensor shapes_MakeFloatTensor(shapes_Context *ctx, Dim shape, f32 initialValues);
Tensor shapes_MakeFloat64Tensor(shapes_Context *ctx, Dim shape, f64 initialValue);
Tensor shapes_MakeFromContigousArray(shapes_Context *ctx, Dim shape, void *values, shapes_Dtype dtype);
Tensor shapes_MakeRandomTensor(shapes_Context *ctx, Dim shape, f32 minValue, f32 maxValue, shapes_Dtype dtype);
Tensor shapes_MakeOneHotTensor(shapes_Context *ctx, Tensor *indices, dim_t numClasses);
Tensor shapes_MakeArangeTensor(shapes_Context *ctx, f32 start, f32 end, f32 step);
void   shapes_SetValues(Tensor *t, shapes_Value value);

#endif
