#ifndef shapes_tensor_internal_h
#define shapes_tensor_internal_h

#include "../shapes.h"
#include "common.h"
#include "value.h"
#include <stddef.h>

typedef struct {
  Tensor *a;
  Tensor *b;
} TensorPair;

static inline Tensor singleValueTensor(Context *ctx, Value value) {
  void *values = allocate(ctx->memory, getBytesForDtype(value.dtype));
  VALUE_SET(values, 0, value);

  return (Tensor){.dtype = value.dtype,
                  .values = values,
                  .size = 1,
                  .isContigous = true,
                  .isView = false,
                  .boundary = NULL,
                  .shape = {.dims = NULL, .numOfDims = 0, .multipliers = NULL}};
}

Result init1DTensor(Context *ctx, Tensor *dest, dim_t size, Dtype dtype);
Result init2DTensor(Context *ctx, Tensor *dest, dim_t rows, dim_t cols, Dtype dtype);
Result init4DTensor(Context *ctx, Tensor *dest, dim_t d0, dim_t d1, dim_t d2, dim_t d3,
                    Dtype dtype);
Result initTensorLike(Context *ctx, Tensor *dest, Tensor *src, Dtype dtype);

u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx);
void unravel_index(tensor_size_t flatIdx, Dim *shape, dim_t *destCoords);
Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type);
Tensor *copyToContiguous(Context *ctx, Tensor *source);
bool areBroadcastable(Tensor *a, Tensor *b);
TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b);

bool isInvalidTensor(Tensor *t);
tensor_size_t calculateNumValuesAndMultipliers(Dim shape, u8 *multipliers);
Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result);

void accumulateStridedByDtype(Dtype dtype, void *destValues, u64 destBase, u64 destStep,
                              void *srcValues, u64 srcBase, u64 srcStep, u64 count);

bool isIntType(Tensor *t);
bool isNotFloatType(Tensor *t);

Result powValue(Value *v, f32 power);
Result sqrtValue(Value *v);

Result freeTensorBuffers(Context *ctx, Tensor *t);

#endif
