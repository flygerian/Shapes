#ifndef shapes_tensor_internal_h
#define shapes_tensor_internal_h

#include "tensor.h"
#include <stddef.h>

typedef struct {
  Tensor *a;
  Tensor *b;
} TensorPair;

u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx);
void unravel_index(tensor_size_t flatIdx, Dim *shape, dim_t *destCoords);
Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type);
Tensor *copyToContiguous(Context *ctx, Tensor *source);
bool areBroadcastable(Tensor *a, Tensor *b);
TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b);

bool isInvalidTensor(Tensor *t);
u64 calculateNumValuesAndMultipliers(Dim shape, u8 *multipliers);
Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result);
Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result);

bool isIntType(Tensor *t);

#endif
