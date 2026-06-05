#ifndef shapes_cuda_h
#define shapes_cuda_h

#include "array.h"
#include "olib.h"
#include <stddef.h>
#include "shapes_common_types.h"

#ifdef SHAPES_HAS_CUDA
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

#ifdef SHAPES_HAS_CUDA
void runCudaGemm(cublasHandle_t handle, shapes_Dtype dtype, cublasOperation_t transA, cublasOperation_t transB, int m, int n, int k, const void *a, int lda,
                 const void *b, int ldb, bool accumulate, void *c, int ldc);
#endif

Result runCudaBinaryOp(shapes_Dtype dtype, shapes_OpType opType, const void *a, const void *b, void *dest, size_t n);
Result runCudaBroadcastBinaryOp(shapes_Dtype dtype, shapes_OpType opType, void *larger,
                                void *smaller, void *dest, size_t outerDimSize,
                                size_t broadcastDimSize, size_t innerDimSize);
Result runCudaUnaryOp(shapes_Dtype dtype, shapes_UnaryOpType opType, const void *src, void *dest, size_t n, f32 param);
Result runCudaReluBackward(shapes_Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result runCudaReluBackwardAccumulate(shapes_Dtype dtype, const void *output, const void *gradOut, void *dest, size_t n);
Result runCudaReduceDim(shapes_Dtype inputDtype, shapes_Dtype outputDtype, shapes_ReductionOpType opType, const void *src, void *dest,
                        size_t numBeforeDim, size_t numAfterDim, size_t reduce);
Result runCudaReduceAll(shapes_Dtype dtype, shapes_ReductionOpType opType, const void *src, void *dest, size_t n);
Result runCudaStd(shapes_Dtype dtype, const void *src, void *dest, size_t n);
Result runCudaIndexAccumulate1d(shapes_Dtype dtype, void *dest, const void *indices, shapes_Dtype indexDtype, const void *srcGrad,
                                size_t numIndices, size_t sliceSize);
Result runCudaIndexAccumulate2d(shapes_Dtype dtype, void *dest, size_t destDim1, const void *rowIndices, shapes_Dtype rowIndexDtype,
                                const void *colIndices, shapes_Dtype colIndexDtype, const void *srcGrad, size_t numIndices, size_t sliceSize);
Result runCudaSliceAccumulate(shapes_Dtype dtype, void *dest, size_t destNumDims, const size_t *destMultipliers,
                              const shapes_Range *ranges, const void *srcGrad, const size_t *srcDims, size_t srcNumDims, size_t srcSize);
Result runCudaCast(shapes_Dtype sourceDtype, const void *src, shapes_Dtype targetDtype, void *dest, size_t n);
Result runCudaFillTensor(shapes_Dtype dtype, void *dest, size_t n, shapes_Value value);
Result runCudaArange(f32 start, f32 step, void *dest, size_t n);
Result runCudaOneHot(shapes_Dtype indexDtype, const void *indices, size_t n, size_t numClasses, void *dest);
Result runCudaIndexSelect1d(shapes_Dtype dtype, const void *src, const void *indices, shapes_Dtype indexDtype, void *dest, size_t numIndices,
                            size_t sliceSize);
Result runCudaIndexSelect2d(shapes_Dtype dtype, const void *src, size_t sourceDim1, const void *rowIndices, shapes_Dtype rowIndexDtype,
                            const void *colIndices, shapes_Dtype colIndexDtype, void *dest, size_t numIndices, size_t sliceSize);
Result runCudaSgd(shapes_Dtype dtype, void *param, const void *grad, size_t n, f32 learningRate);

Result runCudaIm2col(shapes_Dtype dtype, const void *input, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *colBuffer);
Result runCudaCol2imAccumulate(shapes_Dtype dtype, void *dest, const void *colBuffer, size_t batch, size_t inChannels, size_t h, size_t w, size_t kH, size_t kW, u8 stride);

Result runCudaConvBiasAdd(shapes_Dtype dtype, void *output, const void *bias, size_t numValues, size_t channels);
Result runCudaMaxPool2d(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output);
Result runCudaMaxPool2dWithIndices(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *output, void *indices);
Result runCudaMaxPool2dBackward(shapes_Dtype dtype, const void *input, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t kH, size_t kW, u8 stride, void *dX);
Result runCudaMaxPool2dBackwardWithIndices(shapes_Dtype dtype, const void *gradOut, const void *indices, size_t numGradValues, void *dX);
Result runCudaAdaptiveAvgPool2d(shapes_Dtype dtype, const void *input, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *output);
Result runCudaAdaptiveAvgPool2dBackward(shapes_Dtype dtype, const void *gradOut, size_t batch, size_t channels, size_t h, size_t w, size_t outH, size_t outW, void *dX);

Result runCudaCrossEntropyForward(shapes_Dtype dtype, const void *yGround, const void *logits, size_t rows, size_t classCount, void *probs, void *loss);
Result runCudaCrossEntropyBackward(shapes_Dtype dtype, const void *yGround, const void *probs, const void *gradOut, size_t rows, size_t classCount, bool scalarGradOut, void *dLogits);
#ifdef __cplusplus
}
#endif
#endif
