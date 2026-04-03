#ifndef shapes_tensor_internal_h
#define shapes_tensor_internal_h

#include "../shapes.h"
#include "cblas.h"
#include "../common.h"
#include "result/result.h"
#include "value.h"
#include <stddef.h>

typedef struct {
  Tensor *a;
  Tensor *b;
} TensorPair;

typedef struct {
  Tensor *tensor;
  bool ownsTensor;
} TensorArg;

static inline Result ensureAllocated(const void *ptr) {
  return ptr != NULL ? OK : ERR_OUT_OF_MEMORY;
}

static inline Tensor singleValueTensor(Context *ctx, Value value) {
  size_t valueBytes = getBytesForDtype(value.dtype);
  void *values = allocateOnCtx(ctx, valueBytes);
  if (values != NULL) {
    if (copyBetweenContexts(NULL, ctx, &value.as, values, valueBytes) != OK) {
      freeOnCtx(ctx, values);
      values = NULL;
    }
  }

  return (Tensor){.context = ctx,
                  .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                  .dtype = value.dtype,
                  .values = values,
                  .size = 1,
                  .isContigous = true,
                  .isView = false,
                  .boundary = NULL,
                  .shape = {.dims = NULL, .numOfDims = 0, .multipliers = NULL}};
}

static inline Tensor tensorView(Context *ctx, Memory *metadataMemory, void *values,
                                tensor_size_t size, Dtype dtype, Dim shape, Range *boundary,
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

Result initTensor(Context *ctx, Tensor *dest, Dim shape, Dtype dtype);
Result init1DTensor(Context *ctx, Tensor *dest, dim_t size, Dtype dtype);
Result init2DTensor(Context *ctx, Tensor *dest, dim_t rows, dim_t cols, Dtype dtype);
Result init4DTensor(Context *ctx, Tensor *dest, dim_t d0, dim_t d1, dim_t d2, dim_t d3,
                    Dtype dtype);
Result initTensorLike(Context *ctx, Tensor *dest, Tensor *src, Dtype dtype);
Result readTensorValueAtFlatIndex(Tensor *t, u64 idx, Value *result);
Result writeTensorValueAtFlatIndex(Tensor *t, u64 idx, Value value);

u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx);
void unravel_index(tensor_size_t flatIdx, Dim *shape, dim_t *destCoords);
Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type);
Tensor *copyToContiguous(Context *ctx, Tensor *source);
bool isSameContext(Context *a, Context *b);
Tensor* materializeTensorOnContext(Context *ctx, Tensor *src);
void releaseTensorArg(Context *fallbackCtx, TensorArg *arg);
void freeIfContingousCopy(Context *ctx, Tensor* tensor);
Result clearTensorValues(Tensor *t);
bool areBroadcastable(Tensor *a, Tensor *b);
TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b);

bool isInvalidTensor(Tensor *t);
tensor_size_t calculateNumValuesAndMultipliers(Dim shape, multiplier_t *multipliers);
Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result);

void accumulateStridedByDtype(Dtype dtype, void *destValues, u64 destBase, u64 destStep,
                              void *srcValues, u64 srcBase, u64 srcStep, u64 count);

Result moveTensor(Context *srcCtx, Context *destCtx, Tensor *t);

bool isIntType(Tensor *t);
bool isNotFloatType(Tensor *t);

Result powValue(Value *v, f32 power);
Result sqrtValue(Value *v);

Result freeTensorBuffers(Context *ctx, Tensor *t);
void runGemm(Context *ctx, Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m,
             int n, int k, const void *a, int lda, const void *b, int ldb, bool accumulate, void *c,
             int ldc);
#ifdef __cplusplus
extern "C" {
#endif
void runCudaGemm(Context *ctx, Dtype dtype, cublasOperation_t transA, cublasOperation_t transB,
                 int m, int n, int k, const void *a, int lda, const void *b, int ldb,
                 bool accumulate, void *c, int ldc);
Result runCudaBinaryOp(Context *ctx, Dtype dtype, OpType opType, const void *a, const void *b,
                       void *dest, tensor_size_t n);
Result runCudaUnaryOp(Context *ctx, Dtype dtype, UnaryOpType opType, const void *src, void *dest,
                      tensor_size_t n, f32 param);
Result runCudaReluBackward(Context *ctx, Dtype dtype, const void *output, const void *gradOut,
                           void *dest, tensor_size_t n);
Result runCudaReluBackwardAccumulate(Context *ctx, Dtype dtype, const void *output,
                                     const void *gradOut, void *dest, tensor_size_t n);
Result runCudaReduceDim(Context *ctx, Dtype inputDtype, Dtype outputDtype, ReductionOpType opType,
                        const void *src, void *dest, tensor_size_t numBeforeDim,
                        tensor_size_t numAfterDim, dim_t reduce);
Result runCudaReduceAll(Context *ctx, Dtype dtype, ReductionOpType opType, const void *src,
                        void *dest, tensor_size_t n);
Result runCudaStd(Context *ctx, Dtype dtype, const void *src, void *dest, tensor_size_t n);
Result runCudaIndexAccumulate1d(Context *ctx, Dtype dtype, void *dest, const void *indices,
                                Dtype indexDtype, const void *srcGrad, tensor_size_t numIndices,
                                tensor_size_t sliceSize);
Result runCudaIndexAccumulate2d(Context *ctx, Dtype dtype, void *dest, dim_t destDim1,
                                const void *rowIndices, Dtype rowIndexDtype, const void *colIndices,
                                Dtype colIndexDtype, const void *srcGrad, tensor_size_t numIndices,
                                tensor_size_t sliceSize);
Result runCudaSliceAccumulate(Context *ctx, Dtype dtype, void *dest, tensor_size_t destNumDims,
                              const multiplier_t *destMultipliers, const Range *ranges,
                              const void *srcGrad, const dim_t *srcDims, tensor_size_t srcNumDims,
                              tensor_size_t srcSize);
Result runCudaCast(Context *ctx, Dtype sourceDtype, const void *src, Dtype targetDtype, void *dest,
                   tensor_size_t n);
Result runCudaFillTensor(Context *ctx, Dtype dtype, void *dest, tensor_size_t n, Value value);
Result runCudaArange(Context *ctx, f32 start, f32 step, void *dest, tensor_size_t n);
Result runCudaOneHot(Context *ctx, Dtype indexDtype, const void *indices, tensor_size_t n,
                     dim_t numClasses, void *dest);
Result runCudaIndexSelect1d(Context *ctx, Dtype dtype, const void *src, const void *indices,
                            Dtype indexDtype, void *dest, tensor_size_t numIndices,
                            tensor_size_t sliceSize);
Result runCudaIndexSelect2d(Context *ctx, Dtype dtype, const void *src, dim_t sourceDim1,
                            const void *rowIndices, Dtype rowIndexDtype, const void *colIndices,
                            Dtype colIndexDtype, void *dest, tensor_size_t numIndices,
                            tensor_size_t sliceSize);
Result runCudaSgd(Context *ctx, Dtype dtype, void *param, const void *grad, tensor_size_t n,
                  f32 learningRate);
#ifdef __cplusplus
}
#endif
#endif
