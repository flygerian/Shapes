#include "result.h"
#include "shapes.h"
#include "shapes_internal.h"
#include "shapescuda.h"

// ---- shapescuda_Memory functions ----

shapescuda_Memory Make_CudaMemory(Memory *restrict hostMemory) {
  return (shapescuda_Memory){.blocks = MakeDynamicArray(hostMemory, sizeof(CudaBlock *)),
                      .allocationPointer = 0,
                      .allocationCheckpoint = -1};
}

shapescuda_Memory GetCudaMemoryScratchCheckPoint(shapescuda_Memory *restrict cudaMemory) {
  size_t currentAllocPoint = cudaMemory->blocks->size - 1;
  return (shapescuda_Memory){.allocationPointer = currentAllocPoint,
                      .allocationCheckpoint = currentAllocPoint,
                      .blocks = cudaMemory->blocks};
}

void Rewind(shapescuda_Memory *restrict cudaMemory) {
  cudaMemory->allocationPointer = cudaMemory->allocationCheckpoint;
}

void ReleaseCudaBlocks(shapescuda_Memory *restrict cudaMemory) {
  (void)cudaMemory;
}

void FreeCudaScratchMemory(shapescuda_Memory *restrict cudaMemory) {
  (void)cudaMemory;
}

CudaBlock AllocateOnCuda(shapescuda_Memory *restrict cudaMemory, Memory *restrict hostMemory,
                         size_t size) {
  (void)cudaMemory;
  (void)hostMemory;
  (void)size;
  PANIC_WITH_MSG_IF(1, "CUDA support is not compiled in");
  return (CudaBlock){0};
}

// ---- Tensor ops ----

#ifdef SHAPES_HAS_CUDA 
void runCudaGemm(cublasHandle_t handle, shapes_Dtype dtype, cublasOperation_t transA, cublasOperation_t transB,
                 int m, int n, int k, const void *a, int lda, const void *b, int ldb,
                 bool accumulate, void *c, int ldc) {
  
  (void)dtype;
  (void)transA;
  (void)transB;
  (void)m;
  (void)n;
  (void)k;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  (void)accumulate;
  (void)c;
  (void)ldc;
  PANIC_WITH_MSG_IF(1, "CUDA support is not compiled in");
}
#endif

Result runCudaBinaryOp(shapes_Dtype shapes_OpType shapes_OpType opType, const void *a, const void *b,
                       void *dest, tensor_size_t n) {
  
  (void)dtype;
  (void)opType;
  (void)a;
  (void)b;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaBroadcastBinaryOp(shapes_Dtype shapes_OpType shapes_OpType opType, void *larger,
                                void *smaller, void *dest, size_t outerDimSize,
                                size_t broadcastDimSize, size_t innerDimSize) {
  
  (void)dtype;
  (void)opType;
  (void)larger;
  (void)smaller;
  (void)dest;
  (void)outerDimSize;
  (void)broadcastDimSize;
  (void)innerDimSize;
  return ERR_NO_OP;
}

Result runCudaUnaryOp(shapes_Dtype shapes_UnaryOpTypeyOpType opType, const void *src, void *dest,
                      tensor_size_t n, f32 param) {
  
  (void)dtype;
  (void)opType;
  (void)src;
  (void)dest;
  (void)n;
  (void)param;
  return ERR_NO_OP;
}

Result runCudaReluBackward(shapes_Dtype dtype, const void *output, const void *gradOut,
                           void *dest, tensor_size_t n) {
  
  (void)dtype;
  (void)output;
  (void)gradOut;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaReluBackwardAccumulate(shapes_Dtype dtype, const void *output,
                                     const void *gradOut, void *dest, tensor_size_t n) {
  
  (void)dtype;
  (void)output;
  (void)gradOut;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaReduceDim(shapes_Dtype inputDtype, shapes_Dtypeshapes_ReductionOpTypeeductionOpType opType,
                        const void *src, void *dest, tensor_size_t numBeforeDim,
                        tensor_size_t numAfterDim, dim_t reduce) {
  
  (void)inputDtype;
  (void)outputDtype;
  (void)opType;
  (void)src;
  (void)dest;
  (void)numBeforeDim;
  (void)numAfterDim;
  (void)reduce;
  return ERR_NO_OP;
}

Result runCudaReduceAll(shapes_Dtype shapes_ReductionOpTypenOpType opType, const void *src,
                        void *dest, tensor_size_t n) {
  
  (void)dtype;
  (void)opType;
  (void)src;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaStd(shapes_Dtype dtype, const void *src, void *dest, tensor_size_t n) {
  
  (void)dtype;
  (void)src;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaIndexAccumulate1d(shapes_Dtype dtype, void *dest, const void *indices,
                                 shapes_Dtype indexDtype, const void *srcGrad,
                                 tensor_size_t numIndices, tensor_size_t sliceSize) {
  
  (void)dtype;
  (void)dest;
  (void)indices;
  (void)indexDtype;
  (void)srcGrad;
  (void)numIndices;
  (void)sliceSize;
  return ERR_NO_OP;
}

Result runCudaIndexAccumulate2d(shapes_Dtype dtype, void *dest, dim_t destDim1,
                                 const void *rowIndices, shapes_Dtype rowIndexDtype,
                                 const void *colIndices, shapes_Dtype colIndexDtype,
                                 const void *srcGrad, tensor_size_t numIndices,
                                 tensor_size_t sliceSize) {
  
  (void)dtype;
  (void)dest;
  (void)destDim1;
  (void)rowIndices;
  (void)rowIndexDtype;
  (void)colIndices;
  (void)colIndexDtype;
  (void)srcGrad;
  (void)numIndices;
  (void)sliceSize;
  return ERR_NO_OP;
}

Result runCudaSliceAccumulate(shapes_Dtype dtype, void *dest,
                               tensor_size_t destNumDims, const multiplier_t *destMultipliers,
                               const Range *ranges, const void *srcGrad, const dim_t *srcDims,
                               tensor_size_t srcNumDims, tensor_size_t srcSize) {
  
  (void)dtype;
  (void)dest;
  (void)destNumDims;
  (void)destMultipliers;
  (void)ranges;
  (void)srcGrad;
  (void)srcDims;
  (void)srcNumDims;
  (void)srcSize;
  return ERR_NO_OP;
}

Result runCudaCast(shapes_Dtype sourceDtype, const void *src, shapes_Dtype targetDtype, void *dest,
                   tensor_size_t n) {
  
  (void)sourceDtype;
  (void)src;
  (void)targetDtype;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaFillTensor(shapes_Dtype dtype, void *dest, tensor_size_t n, Value value) {
  
  (void)dtype;
  (void)dest;
  (void)n;
  (void)value;
  return ERR_NO_OP;
}

Result runCudaArange(f32 start, f32 step, void *dest, tensor_size_t n) {
  
  (void)start;
  (void)step;
  (void)dest;
  (void)n;
  return ERR_NO_OP;
}

Result runCudaOneHot(shapes_Dtype indexDtype, const void *indices, tensor_size_t n,
                     dim_t numClasses, void *dest) {
  
  (void)indexDtype;
  (void)indices;
  (void)n;
  (void)numClasses;
  (void)dest;
  return ERR_NO_OP;
}

Result runCudaIndexSelect1d(shapes_Dtype dtype, const void *src, const void *indices,
                              shapes_Dtype indexDtype, void *dest, tensor_size_t numIndices,
                              tensor_size_t sliceSize) {
  
  (void)dtype;
  (void)src;
  (void)indices;
  (void)indexDtype;
  (void)dest;
  (void)numIndices;
  (void)sliceSize;
  return ERR_NO_OP;
}

Result runCudaIndexSelect2d(shapes_Dtype dtype, const void *src, dim_t sourceDim1,
                              const void *rowIndices, shapes_Dtype rowIndexDtype,
                              const void *colIndices, shapes_Dtype colIndexDtype, void *dest,
                              tensor_size_t numIndices, tensor_size_t sliceSize) {
  
  (void)dtype;
  (void)src;
  (void)sourceDim1;
  (void)rowIndices;
  (void)rowIndexDtype;
  (void)colIndices;
  (void)colIndexDtype;
  (void)dest;
  (void)numIndices;
  (void)sliceSize;
  return ERR_NO_OP;
}

Result runCudaSgd(shapes_Dtype dtype, void *param, const void *grad, tensor_size_t n,
                   f32 learningRate) {
  
  (void)dtype;
  (void)param;
  (void)grad;
  (void)n;
  (void)learningRate;
  return ERR_NO_OP;
}

// ---- Layer ops ----

Result runCudaIm2col(shapes_Dtype dtype, const void *input, dim_t batch, dim_t inChannels,
                     dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, void *colBuffer) {
  
  (void)dtype;
  (void)input;
  (void)batch;
  (void)inChannels;
  (void)h;
  (void)w;
  (void)kH;
  (void)kW;
  (void)stride;
  (void)colBuffer;
  return ERR_NO_OP;
}

Result runCudaCol2imAccumulate(shapes_Dtype dtype, void *dest, const void *colBuffer,
                                dim_t batch, dim_t inChannels, dim_t h, dim_t w, dim_t kH,
                                dim_t kW, u8 stride) {
  
  (void)dtype;
  (void)dest;
  (void)colBuffer;
  (void)batch;
  (void)inChannels;
  (void)h;
  (void)w;
  (void)kH;
  (void)kW;
  (void)stride;
  return ERR_NO_OP;
}

Result runCudaMaxPool2d(shapes_Dtype dtype, const void *input, dim_t batch,
                        dim_t channels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride,
                        void *output) {
  
  (void)dtype;
  (void)input;
  (void)batch;
  (void)channels;
  (void)h;
  (void)w;
  (void)kH;
  (void)kW;
  (void)stride;
  (void)output;
  return ERR_NO_OP;
}

Result runCudaMaxPool2dWithIndices(shapes_Dtype dtype, const void *input, dim_t batch,
                                   dim_t channels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride,
                                   void *output, void *indices) {
  
  (void)dtype;
  (void)input;
  (void)batch;
  (void)channels;
  (void)h;
  (void)w;
  (void)kH;
  (void)kW;
  (void)stride;
  (void)output;
  (void)indices;
  return ERR_NO_OP;
}

Result runCudaMaxPool2dBackward(shapes_Dtype dtype, const void *input, const void *gradOut,
                                dim_t batch, dim_t channels, dim_t h, dim_t w, dim_t kH, dim_t kW,
                                u8 stride, void *dX) {
  
  (void)dtype;
  (void)input;
  (void)gradOut;
  (void)batch;
  (void)channels;
  (void)h;
  (void)w;
  (void)kH;
  (void)kW;
  (void)stride;
  (void)dX;
  return ERR_NO_OP;
}

Result runCudaMaxPool2dBackwardWithIndices(shapes_Dtype dtype, const void *gradOut,
                                           const void *indices, tensor_size_t numGradValues,
                                           void *dX) {
  
  (void)dtype;
  (void)gradOut;
  (void)indices;
  (void)numGradValues;
  (void)dX;
  return ERR_NO_OP;
}

Result runCudaAdaptiveAvgPool2d(shapes_Dtype dtype, const void *input, dim_t batch,
                                dim_t channels, dim_t h, dim_t w, dim_t outH, dim_t outW,
                                void *output) {
  
  (void)dtype;
  (void)input;
  (void)batch;
  (void)channels;
  (void)h;
  (void)w;
  (void)outH;
  (void)outW;
  (void)output;
  return ERR_NO_OP;
}

Result runCudaAdaptiveAvgPool2dBackward(shapes_Dtype dtype, const void *gradOut,
                                        dim_t batch, dim_t channels, dim_t h, dim_t w, dim_t outH,
                                        dim_t outW, void *dX) {
  
  (void)dtype;
  (void)gradOut;
  (void)batch;
  (void)channels;
  (void)h;
  (void)w;
  (void)outH;
  (void)outW;
  (void)dX;
  return ERR_NO_OP;
}

// ---- Loss ops ----

Result runCudaCrossEntropyForward(shapes_Dtype dtype, const void *yGround,
                                  const void *logits, tensor_size_t rows, dim_t classCount,
                                  void *probs, void *loss) {
  
  (void)dtype;
  (void)yGround;
  (void)logits;
  (void)rows;
  (void)classCount;
  (void)probs;
  (void)loss;
  return ERR_NO_OP;
}

Result runCudaCrossEntropyBackward(shapes_Dtype dtype, const void *yGround,
                                   const void *probs, const void *gradOut, tensor_size_t rows,
                                   dim_t classCount, bool scalarGradOut, void *dLogits) {
  
  (void)dtype;
  (void)yGround;
  (void)probs;
  (void)gradOut;
  (void)rows;
  (void)classCount;
  (void)scalarGradOut;
  (void)dLogits;
  return ERR_NO_OP;
}

// ---- Conv bias (declared locally in layer/conv.c) ----

Result runCudaConvBiasAdd(shapes_Dtype dtype, void *output, const void *bias,
                          tensor_size_t numValues, dim_t channels) {
  
  (void)dtype;
  (void)output;
  (void)bias;
  (void)numValues;
  (void)channels;
  return ERR_NO_OP;
}

Result runCudaConvBiasBackward(shapes_Dtype dtype, const void *outputGrad, void *dBias,
                               tensor_size_t numValues, dim_t channels) {
  
  (void)dtype;
  (void)outputGrad;
  (void)dBias;
  (void)numValues;
  (void)channels;
  return ERR_NO_OP;
}
