#ifndef shapes_h
#define shapes_h

#include "result.h"
#include "types.h"
#include "array.h"
#include <stddef.h>
#include <stdint.h>

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

#define SHAPE(dimensions, numberOfDimensions)           ((shapes_Dim){.dims = (dimensions), .numOfDims = (numberOfDimensions)})
#define SCALAR                                          ((shapes_Dim){.dims = (shapes_dim_t[]){1}, .numOfDims = 1})
#define SHAPE1D(dimSize)                                ((shapes_Dim){.dims = (shapes_dim_t[]){dimSize}, .numOfDims = 1})
#define SHAPE2D(dim0Size, dim1Size)                     ((shapes_Dim){.dims = (shapes_dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define SHAPE3D(dim0Size, dim1Size, dim2Size)           ((shapes_Dim){.dims = (shapes_dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define SHAPE4D(dim0Size, dim1Size, dim2Size, dim3Size) ((shapes_Dim){.dims = (shapes_dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})

// Arrays
typedef olib_Array *shapes_ArrayTensor;
shapes_ArrayTensor shapes_Make_DynamicTensorArray(olib_Memory *memory);
olib_Array *shapes_MakeTensorArray(olib_Memory *memory, size_t capacity);
void shapes_ArrayAppendTensor(olib_Array *array, shapes_Tensor *tensor);
void shapes_ArrayAppendTensorArray(olib_Array *array, olib_Array *tensorArray);
static inline shapes_Tensor shapes_ArrayTensorIdx(olib_Array *array, size_t idx) {
  return *((shapes_Tensor *)olib_ArrayIdx(array, idx));
}

static inline shapes_Tensor* shapes_ArrayTensorPtrIdx(olib_Array *array, size_t idx) {
  return (shapes_Tensor *)olib_ArrayIdx(array, idx);
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
void shapes_MoveTensorToHost(shapes_Context *destCtx, shapes_Tensor *t);

// Binary Ops
shapes_Tensor shapes_Add(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_Subtract(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_Divide(shapes_Context *ctx, shapes_Tensor *numerator, shapes_Tensor *denominator);
shapes_Tensor shapes_Multiply(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_GreaterThan(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_GreaterThanOrEqual(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_Equal(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_LessThan(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_LessThanOrEqual(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);

void shapes_AddInPlace(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
void shapes_SubtractInPlace(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
void shapes_MultiplyInPlace(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);

// Access and shapes
shapes_Value *shapes_GetAt(shapes_Tensor *t, shapes_Dim dim);
Result shapes_CopyShape(shapes_Tensor *t, shapes_dim_t *destDims, u8 *numDims);
Result shapes_AssignValueAt(shapes_Context *ctx, shapes_Tensor *t, shapes_Dim dim, shapes_Value value);
shapes_Tensor shapes_IndexWithTensor(shapes_Context *ctx, shapes_Tensor *source, shapes_Tensor *indices);
shapes_Tensor shapes_IndexWithTensor2d(shapes_Context *ctx, shapes_Tensor *source, shapes_Tensor *rowIndices, shapes_Tensor *colIndices);
shapes_Tensor shapes_Slice(shapes_Context *ctx, shapes_Tensor *source, ...);
shapes_Tensor shapes_Reshape(shapes_Context *ctx, shapes_Tensor *source, shapes_Dim newShape);
void   shapes_ReshapeBackward(shapes_Context *ctx, shapes_Tensor *node);
shapes_Tensor shapes_Transpose(shapes_Context *ctx, shapes_Tensor *source, ...);
shapes_Tensor shapes_Permute(shapes_Context *ctx, shapes_Tensor *source, shapes_Dim order);
shapes_Tensor shapes_Squeeze(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_SqueezeDim(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);
shapes_Tensor shapes_UnSqueeze(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);
shapes_Tensor shapes_Clone(shapes_Context *ctx, shapes_Tensor *t);
void   shapes_Copy(shapes_Context *ctx, shapes_Tensor *src, shapes_Tensor *dest);
shapes_Tensor shapes_Concat(shapes_Context *ctx, shapes_Tensor *target, shapes_dim_t targetDim, shapes_ArrayTensor tensors);
shapes_Tensor shapes_Stack(shapes_Context *ctx, shapes_ArrayTensor tensors);

// Cast
shapes_Tensor Cast(shapes_Context *ctx, shapes_Tensor *source, shapes_Dtype targetDtype);

// Unary
shapes_Tensor shapes_Pow(shapes_Context *ctx, shapes_Tensor *t, f32 power);
shapes_Tensor shapes_Exp(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Tanh(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Relu(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_ReluBackward(shapes_Context *ctx, shapes_Tensor *output, shapes_Tensor *gradOut);
void   shapes_ReluBackwardAccumulate(shapes_Context *ctx, shapes_Tensor *output, shapes_Tensor *gradOut, shapes_Tensor *dest);
shapes_Tensor shapes_Negate(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Log(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Abs(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Sqrt(shapes_Context *ctx, shapes_Tensor *t);
void   shapes_SqrtBackward(shapes_Context *ctx, shapes_Tensor *tensor);

// Reduction
shapes_Tensor shapes_Sum(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);
shapes_Tensor shapes_ReduceBroadcast(shapes_Context *ctx, shapes_Tensor *input, shapes_Tensor *grad);
shapes_Tensor shapes_Mean(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_MeanDim(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);
shapes_Tensor shapes_Std(shapes_Context *ctx, shapes_Tensor *t);
shapes_Tensor shapes_Max(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);
shapes_Tensor shapes_ArgMax(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim);

// Accumulate
void shapes_IndexAccumulate1d(shapes_Context *ctx, shapes_Tensor *dest, shapes_Tensor *indices, shapes_Tensor *srcGrad);
void shapes_IndexAccumulate2d(shapes_Context *ctx, shapes_Tensor *dest, shapes_Tensor *rowIndices, shapes_Tensor *colIndices, shapes_Tensor *srcGrad);
void shapes_SliceAccumulate(shapes_Context *ctx, shapes_Tensor *dest, shapes_Range *ranges, shapes_Tensor *srcGrad);

// Matrix ops
shapes_Tensor shapes_MatMul(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);
shapes_Tensor shapes_Dot(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);

// shapesnn_layer ops
shapes_Tensor shapes_DenseLinear(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *w, shapes_Tensor *b, bool withBias);
Result shapes_DenseBackward(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *w, shapes_Tensor *gradOut, shapes_Tensor *dX, shapes_Tensor *dW, shapes_Tensor *dB);
shapes_BatchNormFowardResult shapes_BatchNormForwardTraining(shapes_Context *ctx, shapes_Tensor *x2d, shapes_Tensor *gamma, shapes_Tensor *beta, f32 epsilon);
shapes_BatchNormBackwardResult shapes_BatchNormBackward(shapes_Context *ctx, shapes_Tensor *x2d, shapes_Tensor *grad2d, shapes_Tensor *gamma, f32 epsilon);
Result shapes_Conv2d(shapes_Context *ctx, size_t inChannels, size_t outChannels, u8 stride, shapes_Tensor *kernels, shapes_Tensor *bias, bool withBias, shapes_Tensor *t, shapes_Tensor *dest,
              shapes_Tensor *colBuffer);
Result shapes_Conv2dBackward(shapes_Context *ctx, shapes_Tensor *input, shapes_Tensor *dInput, shapes_Tensor *kernels, shapes_Tensor *dKernels, shapes_Tensor *outputGrad, shapes_Tensor *colBuffer,
                      shapes_Tensor *dBias, bool withBias, u8 stride);
Result shapes_ConvTranspose2d(shapes_Context *ctx, size_t inChannels, size_t outChannels, u8 stride, shapes_Tensor *kernels, shapes_Dim kernel, shapes_Tensor *t, shapes_Tensor *dest);
Result shapes_ConvTranspose2dBackward(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *kernels, shapes_Tensor *gradOut, u8 stride, shapes_Tensor *dX, shapes_Tensor *dKernels);
Result shapes_MaxPool2d(shapes_Context *ctx, shapes_Tensor *x, shapes_Dim kernel, u8 stride, shapes_Tensor *dest);
Result shapes_MaxPool2dWithIndices(shapes_Context *ctx, shapes_Tensor *x, shapes_Dim kernel, u8 stride, shapes_Tensor *dest, shapes_Tensor *indices);
Result shapes_MaxPool2dBackward(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *gradOut, shapes_Dim kernel, u8 stride, shapes_Tensor *dX);
Result shapes_MaxPool2dBackwardWithIndices(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *gradOut, shapes_Tensor *indices, shapes_Tensor *dX);
Result shapes_AdaptiveAvgPool2d(shapes_Context *ctx, shapes_Tensor *x, shapes_dim_t outH, shapes_dim_t outW, shapes_Tensor *dest);
Result shapes_AdaptiveAvgPool2dBackward(shapes_Context *ctx, shapes_Tensor *x, shapes_Tensor *gradOut, shapes_dim_t outH, shapes_dim_t outW, shapes_Tensor *dX);

// Loss ops
shapes_TensorPair shapes_loss_CrossEntropyForward(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *logits);
shapes_Tensor shapes_loss_CrossEntropyBackward(shapes_Context *ctx, shapes_Tensor *yGround, shapes_Tensor *probs, shapes_Tensor *gradOut);

// Optimizer ops
Result shapes_optimizer_Sgd(shapes_Context *ctx, olib_Array *parameters, f32 learningRate);

Result shapes_optimizer_Adam(shapes_Context *ctx, shapes_AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon);

// Debug
void shapes_PrintItem(shapes_Tensor *t);
char *shapes_GetItem(shapes_Context *ctx, shapes_Tensor *t);
void shapes_PrintTensor(shapes_Tensor *tensor);

// shapes_Tensor creation
shapes_Tensor shapes_MakeZerosTensor(shapes_Context *ctx, shapes_Dim shape);
shapes_Tensor shapes_MakeIntTensor(shapes_Context *ctx, shapes_Dim shape, i8 initialValues);
shapes_Tensor shapes_MakeUIntTensor(shapes_Context *ctx, shapes_Dim shape, u8 initialValue);
shapes_Tensor shapes_MakeFloatTensor(shapes_Context *ctx, shapes_Dim shape, f32 initialValues);
shapes_Tensor shapes_MakeFloat64Tensor(shapes_Context *ctx, shapes_Dim shape, f64 initialValue);
shapes_Tensor shapes_MakeFromContigousArray(shapes_Context *ctx, shapes_Dim shape, void *values, shapes_Dtype dtype);
shapes_Tensor shapes_MakeRandomTensor(shapes_Context *ctx, shapes_Dim shape, f32 minValue, f32 maxValue, shapes_Dtype dtype);
shapes_Tensor shapes_MakeOneHotTensor(shapes_Context *ctx, shapes_Tensor *indices, shapes_dim_t numClasses);
shapes_Tensor shapes_MakeArangeTensor(shapes_Context *ctx, f32 start, f32 end, f32 step);
void   shapes_SetValues(shapes_Tensor *t, shapes_Value value);

#endif
