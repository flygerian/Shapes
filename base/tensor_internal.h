#ifndef shapes_tensor_internal_h
#define shapes_tensor_internal_h

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
  #define TRANSPOSE enum CBLAS_TRANSPOSE
#else
  #include "cblas.h"
  #define TRANSPOSE ( CBLAS_TRANSPOSE ) 
#endif




#include "nn/nn.h"
#include "result/result.h"
#include "utils_lib/array.h"
#include <stddef.h>

typedef struct {
  Tensor *tensor;
  bool ownsTensor;
} TensorArg;

#define NUM_DIMS(tensor) ((tensor)->shape.numOfDims)

static inline Tensor tensorView(Context *ctx, Memory *metadataMemory, void *values, tensor_size_t size, Dtype dtype, Dim shape, Range *boundary,
                                bool isContigous) {
  return (Tensor){.context = ctx,
                  .metadataMemory = metadataMemory,
                  .dtype = dtype,
                  .values = values,
                  .size = size,
                  .isContigous = isContigous,
                  .isView = true,
                  .boundary = boundary,
                  .shape = shape};
}

Result readTensorValueAtFlatIndex(Tensor *t, u64 idx, Value *result);
Result writeTensorValueAtFlatIndex(Tensor *t, u64 idx, Value value);

u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx);
Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type);
Tensor *t_Empty(Context *ctx, Dim shape, Dtype type);
Tensor *t_Reduced(Context *ctx, Tensor *source, dim_t dim, Dtype type);
Tensor *copyToContiguous(Context *ctx, Tensor *source);
bool isSameContext(Context *a, Context *b);
Tensor *materializeTensorOnContext(Context *ctx, Tensor *src);
Result clearTensorValues(Tensor *t);
bool areBroadcastable(Tensor *a, Tensor *b);
TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b);

sizeAndMultipliers calculateSizeAndMultipliers(Context *ctx, dim_t *dims, u8 numOfDims);
Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result);

void accumulateStridedByDtype(Dtype dtype, void *destValues, u64 destBase, u64 destStep, void *srcValues, u64 srcBase, u64 srcStep, u64 count);

void shapesnn_Array_AppendLayer(Array *array, Layer *layer);
Layer *shapesnn_Array_LayerIdx(Array *array, size_t idx);

void runGemm(Context *ctx, Dtype dtype, TRANSPOSE transA, TRANSPOSE transB, int m, int n, int k, const void *a, int lda, const void *b,
             int ldb, bool accumulate, void *c, int ldc);

#ifdef __cplusplus
extern "C" {
#endif
void runCudaGemm(Context *ctx, Dtype dtype, cublasOperation_t transA, cublasOperation_t transB, int m, int n, int k, const void *a, int lda,
                 const void *b, int ldb, bool accumulate, void *c, int ldc);
Result runCudaBinaryOp(Context *ctx, Dtype dtype, OpType opType, const void *a, const void *b, void *dest, tensor_size_t n);
Result runCudaBroadcastBinaryOp(Context *ctx, Dtype dtype, OpType opType, void *larger,
                                void *smaller, void *dest, size_t outerDimSize,
                                size_t broadcastDimSize, size_t innerDimSize);
Result runCudaUnaryOp(Context *ctx, Dtype dtype, UnaryOpType opType, const void *src, void *dest, tensor_size_t n, f32 param);
Result runCudaReluBackward(Context *ctx, Dtype dtype, const void *output, const void *gradOut, void *dest, tensor_size_t n);
Result runCudaReluBackwardAccumulate(Context *ctx, Dtype dtype, const void *output, const void *gradOut, void *dest, tensor_size_t n);
Result runCudaReduceDim(Context *ctx, Dtype inputDtype, Dtype outputDtype, ReductionOpType opType, const void *src, void *dest,
                        tensor_size_t numBeforeDim, tensor_size_t numAfterDim, dim_t reduce);
Result runCudaReduceAll(Context *ctx, Dtype dtype, ReductionOpType opType, const void *src, void *dest, tensor_size_t n);
Result runCudaStd(Context *ctx, Dtype dtype, const void *src, void *dest, tensor_size_t n);
Result runCudaIndexAccumulate1d(Context *ctx, Dtype dtype, void *dest, const void *indices, Dtype indexDtype, const void *srcGrad,
                                tensor_size_t numIndices, tensor_size_t sliceSize);
Result runCudaIndexAccumulate2d(Context *ctx, Dtype dtype, void *dest, dim_t destDim1, const void *rowIndices, Dtype rowIndexDtype,
                                const void *colIndices, Dtype colIndexDtype, const void *srcGrad, tensor_size_t numIndices, tensor_size_t sliceSize);
Result runCudaSliceAccumulate(Context *ctx, Dtype dtype, void *dest, tensor_size_t destNumDims, const multiplier_t *destMultipliers,
                              const Range *ranges, const void *srcGrad, const dim_t *srcDims, tensor_size_t srcNumDims, tensor_size_t srcSize);
Result runCudaCast(Context *ctx, Dtype sourceDtype, const void *src, Dtype targetDtype, void *dest, tensor_size_t n);
Result runCudaFillTensor(Context *ctx, Dtype dtype, void *dest, tensor_size_t n, Value value);
Result runCudaArange(Context *ctx, f32 start, f32 step, void *dest, tensor_size_t n);
Result runCudaOneHot(Context *ctx, Dtype indexDtype, const void *indices, tensor_size_t n, dim_t numClasses, void *dest);
Result runCudaIndexSelect1d(Context *ctx, Dtype dtype, const void *src, const void *indices, Dtype indexDtype, void *dest, tensor_size_t numIndices,
                            tensor_size_t sliceSize);
Result runCudaIndexSelect2d(Context *ctx, Dtype dtype, const void *src, dim_t sourceDim1, const void *rowIndices, Dtype rowIndexDtype,
                            const void *colIndices, Dtype colIndexDtype, void *dest, tensor_size_t numIndices, tensor_size_t sliceSize);
Result runCudaSgd(Context *ctx, Dtype dtype, void *param, const void *grad, tensor_size_t n, f32 learningRate);

static inline bool isInvalidTensor(Tensor *t) {
  if (t == NULL || t->values == NULL) {
    return true;
  }
  // 0-dimensional tensors have shape.dims == NULL, which is valid
  if (t->shape.numOfDims > 0 && t->shape.dims == NULL) {
    return true;
  }
  return false;
}

static inline bool isIntType(Tensor *t) {
  return t->dtype != I8 && t->dtype != I16 && t->dtype != I32 && t->dtype != I64 && t->dtype != U8 && t->dtype != U16 && t->dtype != U32 &&
         t->dtype != U64;
}

static inline bool isNotFloatType(Tensor *t) {
  return t->dtype != F16 && t->dtype != F32 && t->dtype != F64;
}

static inline bool isSameShape(Tensor *a, Tensor *b) {
  if (a->shape.numOfDims != b->shape.numOfDims) {
    return false;
  }

  for (RANGE(i, a->shape.numOfDims)) {
    if (a->shape.dims[i] != b->shape.dims[i]) {
      return false;
    }
  }

  return true;
}

#ifdef __cplusplus
}
#endif
#endif
