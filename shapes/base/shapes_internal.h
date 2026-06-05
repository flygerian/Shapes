#ifndef shapes_tensor_internal_h
#define shapes_tensor_internal_h

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
  #define TRANSPOSE enum CBLAS_TRANSPOSE
#else
  #include "cblas.h"
  #define TRANSPOSE CBLAS_TRANSPOSE
#endif

#include "result.h"
#include "types.h"
#include <stddef.h>

typedef struct {
  Tensor *tensor;
  bool ownsTensor;
} TensorArg;

#define NUM_DIMS(tensor) ((tensor)->shape.numOfDims)

u64 nextNodeId(void);

static inline Tensor tensorView(Context *ctx, olib_Memory *metadataMemory, void *values, tensor_size_t size, shapes_Dtype dtype, Dim shape, shapes_Range *boundary,
                                bool isContigous) {
  return (Tensor){.context = ctx,
                  .metadataMemory = metadataMemory,
                  .dtype = dtype,
                  .values = values,
                  .size = size,
                  .isContigous = isContigous,
                  .isView = true,
                  .boundary = boundary,
                  .shape = shape,
                  .nodeId = nextNodeId()};
}

void attachHostDevice(Context *ctx);
void attachCudaDevice(Context *ctx);
Result readTensorValueAtFlatIndex(Tensor *t, u64 idx, shapes_Value *result);
Result writeTensorValueAtFlatIndex(Tensor *t, u64 idx, shapes_Value value);

dim_t indexValueToDim(shapes_Value idxVal, shapes_Dtype dtype);
u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx);
Tensor t_Zeros(Context *ctx, Dim shape, shapes_Dtype type);
Tensor t_Empty(Context *ctx, Dim shape, shapes_Dtype type);
Tensor t_Reduced(Context *ctx, Tensor *source, dim_t dim, shapes_Dtype type);
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

void accumulateStridedByDtype(shapes_Dtype dtype, void *destValues, u64 destBase, u64 destStep, void *srcValues, u64 srcBase, u64 srcStep, u64 count);


void runGemm(Context *ctx, shapes_Dtype dtype, TRANSPOSE transA, TRANSPOSE transB, int m, int n, int k, const void *a, int lda, const void *b,
             int ldb, bool accumulate, void *c, int ldc);


void im2colNchwF32(const f32 *input, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW, f32 *colBuffer);

Tensor *im2colF32(Context *ctx, Tensor *t, dim_t kernelHeight, dim_t kernelWidth, u8 stride);
Tensor *im2colF64(Context *ctx, Tensor *t, dim_t kernelHeight, dim_t kernelWidth, u8 stride);

void col2imAccumulateF32(Tensor *dInput, f32 *dColBuffer, dim_t kernelHeight, dim_t kernelWidth, u8 stride);
void col2imAccumulateF64(Tensor *dInput, f64 *dColBuffer, dim_t kernelHeight, dim_t kernelWidth, u8 stride);

void im2colNchwF64(const f64 *input, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW, f64 *colBuffer);

void col2imNchwAddF32(const f32 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW, f32 *dest);
void col2imNchwAddF64(const f64 *colBuffer, dim_t inChannels, dim_t h, dim_t w, dim_t kH, dim_t kW, u8 stride, dim_t outH, dim_t outW, f64 *dest);

#ifdef __cplusplus
extern "C" {
#endif

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
