#ifndef shapes_cuda_h
#define shapes_cuda_h

#include "array.h"
#include "olib.h"
#include <stddef.h>
#include "shapes_common_types.h"

#ifdef SHAPES_ENABLE_CUDA 
#include <cublas_v2.h>
#endif

typedef struct CudaMemory {
  Array *blocks;
  size_t allocationPointer;
  i64 allocationCheckpoint;
} CudaMemory;

CudaBlock AllocateOnCuda(CudaMemory *cudaMemory, Memory *hostMemory, size_t size);
void ReleaseCudaBlocks(CudaMemory *cudaMemory);
CudaMemory Make_CudaMemory(Memory *hostMemory);
CudaMemory GetCudaMemoryScratchCheckPoint(CudaMemory *cudaMemory);
void Rewind(CudaMemory *cudaMemory);
void FreeCudaScratchMemory(CudaMemory *cudaMemory);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SHAPES_ENABLE_CUDA 
void runCudaGemm(cublasHandle_t handle, Dtype dtype, cublasOperation_t transA, cublasOperation_t transB, int m, int n, int k, const void *a, int lda,
                 const void *b, int ldb, bool accumulate, void *c, int ldc);
#endif

Result runCudaBinaryOp(Dtype dtype, OpType opType, const void *a, const void *b, void *dest, size_t n);
Result runCudaBroadcastBinaryOp(Dtype dtype, OpType opType, void *larger,
                                void *smaller, void *dest, size_t outerDimSize,
                                size_t broadcastDimSize, size_t innerDimSize);
Result runCudaUnaryOp(Dtype dtype, UnaryOpType opType, const void *src, void *dest, size_t n, f32 param);
Result runCudaReluBackward(Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result runCudaReluBackwardAccumulate(Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result runCudaReduceDim(Dtype inputDtype, Dtype outputDtype, ReductionOpType opType, const void *src, void *dest,
                        size_t numBeforeDim, size_t numAfterDim, size_t reduce);
Result runCudaReduceAll(Dtype dtype, ReductionOpType opType, const void *src, void *dest, size_t n);
Result runCudaStd(Dtype dtype, const void *src, void *dest, size_t n);
Result runCudaIndexAccumulate1d(Dtype dtype, void *dest, const void *indices, Dtype indexDtype, const void *srcGrad,
                                size_t numIndices, size_t sliceSize);
Result runCudaIndexAccumulate2d(Dtype dtype, void *dest, size_t destDim1, const void *rowIndices, Dtype rowIndexDtype,
                                const void *colIndices, Dtype colIndexDtype, const void *srcGrad, size_t numIndices, size_t sliceSize);
Result runCudaSliceAccumulate(Dtype dtype, void *dest, size_t destNumDims, const size_t *destMultipliers,
                              const Range *ranges, const void *srcGrad, const size_t *srcDims, size_t srcNumDims, size_t srcSize);
Result runCudaCast(Dtype sourceDtype, const void *src, Dtype targetDtype, void *dest, size_t n);
Result runCudaFillTensor(Dtype dtype, void *dest, size_t n, Value value);
Result runCudaArange(f32 start, f32 step, void *dest, size_t n);
Result runCudaOneHot(Dtype indexDtype, const void *indices, size_t n, size_t numClasses, void *dest);
Result runCudaIndexSelect1d(Dtype dtype, const void *src, const void *indices, Dtype indexDtype, void *dest, size_t numIndices,
                            size_t sliceSize);
Result runCudaIndexSelect2d(Dtype dtype, const void *src, size_t sourceDim1, const void *rowIndices, Dtype rowIndexDtype,
                            const void *colIndices, Dtype colIndexDtype, void *dest, size_t numIndices, size_t sliceSize);
Result runCudaSgd(Dtype dtype, void *param, const void *grad, size_t n, f32 learningRate);

Result runCudaIm2col(Dtype dtype, const void *input, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *colBuffer);
Result runCudaCol2imAccumulate(Dtype dtype, void *dest, const void *colBuffer, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride);

Result runCudaConvBiasAdd(Dtype dtype, void *output, const void *bias, size_t numValues, size_t channels);
Result runCudaMaxPool2d(Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output);
Result runCudaMaxPool2dWithIndices(Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output, void *indices);
Result runCudaMaxPool2dBackward(Dtype dtype, const void *input, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *dX);
Result runCudaMaxPool2dBackwardWithIndices(Dtype dtype, const void *gradOut, const void *indices, size_t numGradValues, void *dX);
Result runCudaAdaptiveAvgPool2d(Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *output);
Result runCudaAdaptiveAvgPool2dBackward(Dtype dtype, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *dX);

Result runCudaCrossEntropyForward(Dtype dtype, const void *yGround, const void *logits, size_t rows, size_t classCount, void *probs, void *loss);
Result runCudaCrossEntropyBackward(Dtype dtype, const void *yGround, const void *probs, const void *gradOut, size_t rows, size_t classCount, bool scalarGradOut, void *dLogits);
#ifdef __cplusplus
}
#endif
#endif
