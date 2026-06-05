#ifndef shapes_cuda_h
#define shapes_cuda_h

#include "array.h"
#include "olib.h"
#include <stddef.h>
#include "shapes_common_types.h"

#ifdef SHAPES_HAS_CUDA
#include <cublas_v2.h>
#endif

typedef struct {
  void *ptr;
  size_t size;
} shapescuda_Block;

typedef struct shapescuda_Memory {
  Array *blocks;
  size_t allocationPointer;
  i64 allocationCheckpoint;
} shapescuda_Memory;

shapescuda_Block shapescuda_Allocate(shapescuda_Memory *cudaMemory, Memory *hostMemory, size_t size);
void shapescuda_ReleaseBlocks(shapescuda_Memory *cudaMemory);
shapescuda_Memory shapescuda_Make_Memory(Memory *hostMemory);
shapescuda_Memory shapescuda_GetMemoryScratchCheckPoint(shapescuda_Memory *cudaMemory);
void shapescuda_RewindMemory(shapescuda_Memory *cudaMemory);
void shapescuda_FreeScratchMemory(shapescuda_Memory *cudaMemory);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SHAPES_HAS_CUDA
void shapescuda_Gemm(cublasHandle_t handle, shapes_Dtype dtype, cublasOperation_t transA, cublasOperation_t transB, int m, int n, int k, const void *a, int lda,
                 const void *b, int ldb, bool accumulate, void *c, int ldc);
#endif

Result shapescuda_BinaryOp(shapes_Dtype dtype, shapes_OpType opType, const void *a, const void *b, void *dest, size_t n);
Result shapescuda_BroadcastBinaryOp(shapes_Dtype dtype, shapes_OpType opType, void *larger,
                                void *smaller, void *dest, size_t outerDimSize,
                                size_t broadcastDimSize, size_t innerDimSize);
Result shapescuda_UnaryOp(shapes_Dtype dtype, shapes_UnaryOpType opType, const void *src, void *dest, size_t n, f32 param);
Result shapescuda_ReluBackward(shapes_Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result shapescuda_ReluBackwardAccumulate(shapes_Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result shapescuda_ReduceDim(shapes_Dtype inputDtype, shapes_Dtype outputDtype, shapes_ReductionOpType opType, const void *src, void *dest,
                        size_t numBeforeDim, size_t numAfterDim, size_t reduce);
Result shapescuda_ReduceAll(shapes_Dtype dtype, shapes_ReductionOpType opType, const void *src, void *dest, size_t n);
Result shapescuda_Std(shapes_Dtype dtype, const void *src, void *dest, size_t n);
Result shapescuda_IndexAccumulate1d(shapes_Dtype dtype, void *dest, const void *indices, shapes_Dtype indexDtype, const void *srcGrad,
                                size_t numIndices, size_t sliceSize);
Result shapescuda_IndexAccumulate2d(shapes_Dtype dtype, void *dest, size_t destDim1, const void *rowIndices, shapes_Dtype rowIndexDtype,
                                const void *colIndices, shapes_Dtype colIndexDtype, const void *srcGrad, size_t numIndices, size_t sliceSize);
Result shapescuda_SliceAccumulate(shapes_Dtype dtype, void *dest, size_t destNumDims, const size_t *destMultipliers,
                              const shapes_Range *ranges, const void *srcGrad, const size_t *srcDims, size_t srcNumDims, size_t srcSize);
Result shapescuda_Cast(shapes_Dtype sourceDtype, const void *src, shapes_Dtype targetDtype, void *dest, size_t n);
Result shapescuda_FillTensor(shapes_Dtype dtype, void *dest, size_t n, shapes_Value value);
Result shapescuda_Arange(f32 start, f32 step, void *dest, size_t n);
Result shapescuda_OneHot(shapes_Dtype indexDtype, const void *indices, size_t n, size_t numClasses, void *dest);
Result shapescuda_IndexSelect1d(shapes_Dtype dtype, const void *src, const void *indices, shapes_Dtype indexDtype, void *dest, size_t numIndices,
                            size_t sliceSize);
Result shapescuda_IndexSelect2d(shapes_Dtype dtype, const void *src, size_t sourceDim1, const void *rowIndices, shapes_Dtype rowIndexDtype,
                            const void *colIndices, shapes_Dtype colIndexDtype, void *dest, size_t numIndices, size_t sliceSize);
Result shapescuda_Sgd(shapes_Dtype dtype, void *param, const void *grad, size_t n, f32 learningRate);

Result shapescuda_Im2col(shapes_Dtype dtype, const void *input, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *colBuffer);
Result shapescuda_Col2imAccumulate(shapes_Dtype dtype, void *dest, const void *colBuffer, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride);

Result shapescuda_ConvBiasAdd(shapes_Dtype dtype, void *output, const void *bias, size_t numValues, size_t channels);
Result shapescuda_MaxPool2d(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output);
Result shapescuda_MaxPool2dWithIndices(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output, void *indices);
Result shapescuda_MaxPool2dBackward(shapes_Dtype dtype, const void *input, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *dX);
Result shapescuda_MaxPool2dBackwardWithIndices(shapes_Dtype dtype, const void *gradOut, const void *indices, size_t numGradValues, void *dX);
Result shapescuda_AdaptiveAvgPool2d(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *output);
Result shapescuda_AdaptiveAvgPool2dBackward(shapes_Dtype dtype, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *dX);

Result shapescuda_CrossEntropyForward(shapes_Dtype dtype, const void *yGround, const void *logits, size_t rows, size_t classCount, void *probs, void *loss);
Result shapescuda_CrossEntropyBackward(shapes_Dtype dtype, const void *yGround, const void *probs, const void *gradOut, size_t rows, size_t classCount, bool scalarGradOut, void *dLogits);
#ifdef __cplusplus
}
#endif
#endif
