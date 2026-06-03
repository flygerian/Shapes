#ifndef shapes_h
#define shapes_h

#include "result.h"
#include "types.h"
#include "array.h"
#include <stddef.h>
#include <stdint.h>
#include "shapes_internal.h"

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

#define SHAPE(dimensions, numberOfDimensions)           ((Dim){.dims = (dimensions), .numOfDims = (numberOfDimensions)})
#define SCALAR                                          ((Dim){.dims = (dim_t[]){1}, .numOfDims = 1})
#define SHAPE1D(dimSize)                                ((Dim){.dims = (dim_t[]){dimSize}, .numOfDims = 1})
#define SHAPE2D(dim0Size, dim1Size)                     ((Dim){.dims = (dim_t[]){dim0Size, dim1Size}, .numOfDims = 2})
#define SHAPE3D(dim0Size, dim1Size, dim2Size)           ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size}, .numOfDims = 3})
#define SHAPE4D(dim0Size, dim1Size, dim2Size, dim3Size) ((Dim){.dims = (dim_t[]){dim0Size, dim1Size, dim2Size, dim3Size}, .numOfDims = 4})

// Arrays
typedef Array *Array_Tensor;
Array_Tensor shapes_Make_DynamicTensorArray(Memory *memory);
Array *shapes_Make_TensorArray(Memory *memory, size_t capacity);
void shapes_Array_AppendTensor(Array *array, Tensor *tensor);
void shapes_Array_AppendTensorArray(Array *array, Array *tensorArray);
static inline Tensor shapes_Array_TensorIdx(Array *array, size_t idx) {
  return *((Tensor *)Array_Idx(array, idx));
}

static inline Tensor* shapes_Array_TensorPtrIdx(Array *array, size_t idx) {
  return (Tensor *)Array_Idx(array, idx);
}

// Context
Context shapes_InitializeHostContext(size_t arenaSize, size_t minBlockSize);
Context shapes_InitializeCudaContext(size_t hostArenaSize);
Context shapes_GetScratchContext(Context *ctx, size_t bufferSize);
void shapes_DestroyContext(Context *ctx);
void shapes_FreeContext(Context *ctx);
Result shapes_Flush(Context *ctx);

Result shapes_CopyBetweenDevices(DeviceType srcType, DeviceType destType, void *restrict srcPtr, void *restrict destPtr, size_t size);

void shapes_MoveToCuda(Context *destCtx, Array_Tensor tensors);
void shapes_MoveToHost(Context *destCtx, Array_Tensor tensors);
void shapes_MoveTensorToHost(Context *destCtx, Tensor *t);

// Binary Ops
Tensor shapes_Add(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Subtract(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Divide(Context *ctx, Tensor *numerator, Tensor *denominator);
Tensor shapes_Multiply(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_GreaterThan(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Equal(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_LessThan(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b);

void shapes_AddInPlace(Context *ctx, Tensor *a, Tensor *b);
void shapes_SubtractInPlace(Context *ctx, Tensor *a, Tensor *b);
void shapes_MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b);

// Access and shapes
Value *shapes_GetAt(Tensor *t, Dim dim);
Result shapes_CopyShape(Tensor *t, dim_t *destDims, u8 *numDims);
Result shapes_AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value);
Tensor shapes_IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices);
Tensor shapes_IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices);
Tensor shapes_Slice(Context *ctx, Tensor *source, ...);
Tensor shapes_Reshape(Context *ctx, Tensor *source, Dim newShape);
void shapes_ReshapeBackward(Context *ctx, Tensor *node);
Tensor shapes_Transpose(Context *ctx, Tensor *source, ...);
Tensor shapes_Permute(Context *ctx, Tensor *source, Dim order);
Tensor shapes_Squeeze(Context *ctx, Tensor *t);
Tensor shapes_SqueezeDim(Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_UnSqueeze(Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_Clone(Context *ctx, Tensor *t);
void shapes_Copy(Context *ctx, Tensor *src, Tensor *dest);
Tensor shapes_Concat(Context *ctx, Tensor *target, dim_t targetDim, Array_Tensor tensors);
Tensor shapes_Stack(Context *ctx, Array_Tensor tensors);

// Cast
Tensor Cast(Context *ctx, Tensor *source, Dtype targetDtype);

// Unary
Tensor shapes_Pow(Context *ctx, Tensor *t, f32 power);
Tensor shapes_Exp(Context *ctx, Tensor *t);
Tensor shapes_Tanh(Context *ctx, Tensor *t);
Tensor shapes_Relu(Context *ctx, Tensor *t);
Tensor shapes_ReluBackward(Context *ctx, Tensor *output, Tensor *gradOut);
void shapes_ReluBackwardAccumulate(Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest);
Tensor shapes_Negate(Context *ctx, Tensor *t);
Tensor shapes_Log(Context *ctx, Tensor *t);
Tensor shapes_Abs(Context *ctx, Tensor *t);
Tensor shapes_Sqrt(Context *ctx, Tensor *t);
void shapes_SqrtBackward(Context *ctx, Tensor *node);

// Reduction
Tensor shapes_Sum(Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_ReduceBroadcast(Context *ctx, Tensor *input, Tensor *grad);
Tensor shapes_Mean(Context *ctx, Tensor *t);
Tensor shapes_MeanDim(Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_Std(Context *ctx, Tensor *t);
Tensor shapes_Max(Context *ctx, Tensor *t, dim_t dim);
Tensor shapes_ArgMax(Context *ctx, Tensor *t, dim_t dim);

// Accumulate
void shapes_IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad);
void shapes_IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices, Tensor *srcGrad);
void shapes_SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad);

// Matrix ops
Tensor shapes_MatMul(Context *ctx, Tensor *a, Tensor *b);
Tensor shapes_Dot(Context *ctx, Tensor *a, Tensor *b);

// Layer ops
Tensor shapes_layer_DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias);
Result shapes_layer_DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW, Tensor *dB);
BatchNormFowardResult shapes_layer_BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon);
BatchNormBackwardResult shapes_layer_BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon);
Result shapes_layer_Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Tensor *bias, bool withBias, Tensor *t, Tensor *dest,
              Tensor *colBuffer);
Result shapes_layer_Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels, Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer,
                      Tensor *dBias, bool withBias, u8 stride);
Result shapes_layer_ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels, Dim kernel, Tensor *t, Tensor *dest);
Result shapes_layer_ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride, Tensor *dX, Tensor *dKernels);
Result shapes_layer_MaxPool2d(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest);
Result shapes_layer_MaxPool2dWithIndices(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest, Tensor *indices);
Result shapes_layer_MaxPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, Dim kernel, u8 stride, Tensor *dX);
Result shapes_layer_MaxPool2dBackwardWithIndices(Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices, Tensor *dX);
Result shapes_layer_AdaptiveAvgPool2d(Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor *dest);
Result shapes_layer_AdaptiveAvgPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH, dim_t outW, Tensor *dX);

// Loss ops
TensorPair shapes_loss_CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits);
Tensor shapes_loss_CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut);

// Optimizer ops
Result shapes_optimizer_Sgd(Context *ctx, Array *parameters, f32 learningRate);

Result shapes_optimizer_Adam(Context *ctx, AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step, f32 a, f32 epsilon);

// Debug
void shapes_PrintItem(Tensor *t);
char *shapes_GetItem(Context *ctx, Tensor *t);
void shapes_PrintTensor(Tensor *tensor);

// Tensor creation
Tensor shapes_Make_ZerosTensor(Context *ctx, Dim shape);
Tensor shapes_Make_IntTensor(Context *ctx, Dim shape, i8 initialValues);
Tensor shapes_Make_UIntTensor(Context *ctx, Dim shape, u8 initialValue);
Tensor shapes_Make_FloatTensor(Context *ctx, Dim shape, f32 initialValues);
Tensor shapes_Make_Float64Tensor(Context *ctx, Dim shape, f64 initialValue);
Tensor shapes_Make_FromContigousArray(Context *ctx, Dim shape, void *values, Dtype dtype);
Tensor shapes_Make_RandomTensor(Context *ctx, Dim shape, f32 minValue, f32 maxValue, Dtype dtype);
Tensor shapes_Make_OneHotTensor(Context *ctx, Tensor *indices, dim_t numClasses);
Tensor shapes_Make_ArangeTensor(Context *ctx, f32 start, f32 end, f32 step);
void shapes_SetValues(Tensor *t, Value value);

#endif
