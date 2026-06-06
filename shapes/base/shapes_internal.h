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
  shapes_Tensor *tensor;
  bool ownsTensor;
} TensorArg;

#define NUM_DIMS(tensor) ((tensor)->shape.numOfDims)

u64 nextNodeId(void);

static inline shapes_Tensor tensorView(shapes_Context *ctx, olib_Memory *metadataMemory, void *values, shapes_tensor_size_t size, shapes_Dtype dtype, shapes_Dim shape, shapes_Range *boundary,
                                bool isContigous) {
  return (shapes_Tensor){.context = ctx,
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

void attachHostDevice(shapes_Context *ctx);
void attachCudaDevice(shapes_Context *ctx);
Result readTensorValueAtFlatIndex(shapes_Tensor *t, u64 idx, shapes_Value *result);
Result writeTensorValueAtFlatIndex(shapes_Tensor *t, u64 idx, shapes_Value value);

shapes_dim_t indexValueToDim(shapes_Value idxVal, shapes_Dtype dtype);
u64 getContigousIdxFromCoord(shapes_Tensor *t, shapes_dim_t *idx);
shapes_Tensor t_Zeros(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype type);
shapes_Tensor t_Empty(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype type);
shapes_Tensor t_Reduced(shapes_Context *ctx, shapes_Tensor *source, shapes_dim_t dim, shapes_Dtype type);
shapes_Tensor *copyToContiguous(shapes_Context *ctx, shapes_Tensor *source);
bool isSameContext(shapes_Context *a, shapes_Context *b);
shapes_Tensor *materializeTensorOnContext(shapes_Context *ctx, shapes_Tensor *src);
Result clearTensorValues(shapes_Tensor *t);
bool areBroadcastable(shapes_Tensor *a, shapes_Tensor *b);
shapes_TensorPair padSmallerTensor(shapes_Context *ctx, shapes_Tensor *a, shapes_Tensor *b);

sizeAndMultipliers calculateSizeAndMultipliers(shapes_Context *ctx, shapes_dim_t *dims, u8 numOfDims);
Result calculateNumElementsBeforeDim(shapes_Tensor *t, shapes_dim_t dim, shapes_tensor_size_t *result);
Result calculateNumElementsAfterDim(shapes_Tensor *t, shapes_dim_t dim, shapes_tensor_size_t *result);
Result getDimsBefore(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim, shapes_Dim *result);

void accumulateStridedByDtype(shapes_Dtype dtype, void *destValues, u64 destBase, u64 destStep, void *srcValues, u64 srcBase, u64 srcStep, u64 count);


void runGemm(shapes_Context *ctx, shapes_Dtype dtype, TRANSPOSE transA, TRANSPOSE transB, int m, int n, int k, const void *a, int lda, const void *b,
             int ldb, bool accumulate, void *c, int ldc);


void im2colNchwF32(const f32 *input, shapes_dim_t inChannels, shapes_dim_t h, shapes_dim_t w, shapes_dim_t kH, shapes_dim_t kW, u8 stride, shapes_dim_t outH, shapes_dim_t outW, f32 *colBuffer);

shapes_Tensor *im2colF32(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride);
shapes_Tensor *im2colF64(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride);

void col2imAccumulateF32(shapes_Tensor *dInput, f32 *dColBuffer, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride);
void col2imAccumulateF64(shapes_Tensor *dInput, f64 *dColBuffer, shapes_dim_t kernelHeight, shapes_dim_t kernelWidth, u8 stride);

void im2colNchwF64(const f64 *input, shapes_dim_t inChannels, shapes_dim_t h, shapes_dim_t w, shapes_dim_t kH, shapes_dim_t kW, u8 stride, shapes_dim_t outH, shapes_dim_t outW, f64 *colBuffer);

void col2imNchwAddF32(const f32 *colBuffer, shapes_dim_t inChannels, shapes_dim_t h, shapes_dim_t w, shapes_dim_t kH, shapes_dim_t kW, u8 stride, shapes_dim_t outH, shapes_dim_t outW, f32 *dest);
void col2imNchwAddF64(const f64 *colBuffer, shapes_dim_t inChannels, shapes_dim_t h, shapes_dim_t w, shapes_dim_t kH, shapes_dim_t kW, u8 stride, shapes_dim_t outH, shapes_dim_t outW, f64 *dest);

#ifdef __cplusplus
extern "C" {
#endif

static inline bool isInvalidTensor(shapes_Tensor *t) {
  if (t == NULL || t->values == NULL) {
    return true;
  }
  // 0-dimensional tensors have shape.dims == NULL, which is valid
  if (t->shape.numOfDims > 0 && t->shape.dims == NULL) {
    return true;
  }
  return false;
}

static inline bool isIntType(shapes_Tensor *t) {
  return t->dtype != I8 && t->dtype != I16 && t->dtype != I32 && t->dtype != I64 && t->dtype != U8 && t->dtype != U16 && t->dtype != U32 &&
         t->dtype != U64;
}

static inline bool isNotFloatType(shapes_Tensor *t) {
  return t->dtype != F16 && t->dtype != F32 && t->dtype != F64;
}

static inline bool isSameShape(shapes_Tensor *a, shapes_Tensor *b) {
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
