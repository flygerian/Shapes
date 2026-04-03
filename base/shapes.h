#ifndef shapes_h
#define shapes_h

#include "common.h"
#include "result/result.h"
#include <stddef.h>
#include <stdint.h>

#define MAX_SUM_N_DIMS    2
#define MAX_PARALLEL_SUMS 4

// Context
Context InitializeContext(size_t arenaSize, size_t minBlockSize, bool withCuda);
Context *CreateContext(size_t arenaSize, size_t minBlockSize, bool withCuda);
void DestroyContext(Context *ctx);
void FreeContext(Context *ctx);
Result Flush(Context *ctx);
Result copyBetweenContexts(Context *srcCtx, Context *destCtx, void *srcPtr, void *destPtr,
                           size_t size);
void *allocateOnCtx(Context *ctx, size_t size);
void freeOnCtx(Context *ctx, void *ptr);
void MoveTensors(Context *destCtx, u8 numTensors, ...);

// Binary Ops
Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination);
Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result GreaterThan(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result LessThan(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);

Result AddInPlace(Context *ctx, Tensor *a, Tensor *b);
Result SubtractInPlace(Context *ctx, Tensor *a, Tensor *b);
Result MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b);

// Access and shapes
Result GetAt(Tensor *t, Dim dim, Value *result);
Result GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor *dest);
Result GetScalar(Tensor *t, Value *result);
Result CopyShape(Tensor *t, dim_t *destDims, u8 *numDims);
Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value);
Result IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor *dest);
Result IndexWithTensor2d(Context *ctx, Tensor *source, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *dest);
Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape);
Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...);
Result Permute(Context *ctx, Tensor *source, Tensor *dest, Dim order);
Result Squeeze(Context *ctx, Tensor *t, Tensor *dest);
Result SqueezeDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Clone(Context *ctx, Tensor *t, Tensor *dest);
Result Copy(Context *ctx, Tensor *src, Tensor *dest);
Result Concat(Context *ctx, Tensor *target, dim_t targetDim, Tensor **tensors, u32 numTensorsToAdd,
              Tensor *dest);

// Cast
Result Cast(Context *ctx, Tensor *source, Tensor *dest, Dtype targetDtype);

// Unary
Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest);
Result Exp(Context *ctx, Tensor *t, Tensor *dest);
Result Tanh(Context *ctx, Tensor *t, Tensor *dest);
Result Relu(Context *ctx, Tensor *t, Tensor *dest);
Result ReluBackward(Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest);
Result ReluBackwardAccumulate(Context *ctx, Tensor *output, Tensor *gradOut, Tensor *dest);
Result Negate(Context *ctx, Tensor *t, Tensor *dest);
Result Log(Context *ctx, Tensor *t, Tensor *dest);
Result Abs(Context *ctx, Tensor *t, Tensor *dest);

// Reduction
Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Mean(Context *ctx, Tensor *t, Tensor *dest);
Result MeanDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Std(Context *ctx, Tensor *t, Tensor *dest);
Result Max(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result ArgMax(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);

// Accumulate
Result IndexAccumulate1d(Context *ctx, Tensor *dest, Tensor *indices, Tensor *srcGrad);
Result IndexAccumulate2d(Context *ctx, Tensor *dest, Tensor *rowIndices, Tensor *colIndices,
                         Tensor *srcGrad);
Result SliceAccumulate(Context *ctx, Tensor *dest, Range *ranges, Tensor *srcGrad);

// Matrix ops
Result MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor *result);
Result Dot(Context *ctx, Tensor *a, Tensor *b, Tensor *result);

// Layer ops
Result DenseLinear(Context *ctx, Tensor *x, Tensor *w, Tensor *b, bool withBias, Tensor *dest);
Result DenseBackward(Context *ctx, Tensor *x, Tensor *w, Tensor *gradOut, Tensor *dX, Tensor *dW,
                     Tensor *dB);
Result BatchNormForwardTraining(Context *ctx, Tensor *x2d, Tensor *gamma, Tensor *beta, f32 epsilon,
                                Tensor *out, Tensor *mean, Tensor *variance);
Result BatchNormBackward(Context *ctx, Tensor *x2d, Tensor *grad2d, Tensor *gamma, f32 epsilon,
                         Tensor *dX, Tensor *dGamma, Tensor *dBeta);
Result Conv2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride, Tensor *kernels,
              Tensor *bias, bool withBias, Tensor *t, Tensor *dest, Tensor *colBuffer);
Result Conv2dBackward(Context *ctx, Tensor *input, Tensor *dInput, Tensor *kernels,
                      Tensor *dKernels, Tensor *outputGrad, Tensor *colBuffer, Tensor *dBias,
                      bool withBias, u8 stride);
Result ConvTranspose2d(Context *ctx, size_t inChannels, size_t outChannels, u8 stride,
                       Tensor *kernels, Dim kernel, Tensor *t, Tensor *dest);
Result ConvTranspose2dBackward(Context *ctx, Tensor *x, Tensor *kernels, Tensor *gradOut, u8 stride,
                               Tensor *dX, Tensor *dKernels);
Result MaxPool2d(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest);
Result MaxPool2dWithIndices(Context *ctx, Tensor *x, Dim kernel, u8 stride, Tensor *dest,
                            Tensor *indices);
Result MaxPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, Dim kernel, u8 stride,
                         Tensor *dX);
Result MaxPool2dBackwardWithIndices(Context *ctx, Tensor *x, Tensor *gradOut, Tensor *indices,
                                    Tensor *dX);
Result AdaptiveAvgPool2d(Context *ctx, Tensor *x, dim_t outH, dim_t outW, Tensor *dest);
Result AdaptiveAvgPool2dBackward(Context *ctx, Tensor *x, Tensor *gradOut, dim_t outH, dim_t outW,
                                 Tensor *dX);

// Loss ops
Result CrossEntropyForward(Context *ctx, Tensor *yGround, Tensor *logits, Tensor *loss,
                           Tensor *probs);
Result CrossEntropyBackward(Context *ctx, Tensor *yGround, Tensor *probs, Tensor *gradOut,
                            Tensor *dLogits);

// Optimizer ops
Result Sgd(Context *ctx, Tensor **parameters, Tensor **parameterGrads, size_t numParameters,
           f32 learningRate);


Result Adam(Context *ctx, AdamData *triplets, size_t numTriplets, f32 b1, f32 b2, size_t step,
            f32 a, f32 epsilon);

// Debug
void PrintItem(Tensor *t);
char *GetItem(Context *ctx, Tensor *t);

// Tensor creation
Tensor *T_Zeros(Context *ctx, Dim shape);
Tensor *T_Int(Context *ctx, Dim shape, i8 initialValues);
Tensor *T_UInt(Context *ctx, Dim shape, u8 initialValue);
Tensor *T_Float(Context *ctx, Dim shape, f32 initialValues);
Tensor *T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses);
Tensor *T_Arange(Context *ctx, f32 start, f32 end, f32 step);
void SetValues(Tensor *t, Value value);

// Tensor destruction
Result FreeTensor(Context *ctx, Tensor *t);
Result FreeViewTensor(Context *ctx, Tensor *t);
Result FreeTensors(Context *ctx, Tensor **tensors, int numTensors);

#endif
