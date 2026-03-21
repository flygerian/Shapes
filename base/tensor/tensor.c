#include <stddef.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "../shapes.h"
#include "common.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "value.h"
#include "../memory.h"

void PrintItem(Tensor *t) {
  dim_t zero[1] = {0};
  Dim zeroIdx = {.dims = zero, .numOfDims = 1};

  Value val;
  GetAt(t, zeroIdx, &val);

  PRINT_VALUE(val);
}


char *GetItem(Context *ctx, Tensor *t) {
  dim_t zero[1] = {0};
  Dim zeroIdx = {.dims = zero, .numOfDims = 1};

  Value val;
  GetAt(t, zeroIdx, &val);

  size_t size = sizeof(char) * 32;
  char *valueStr = allocate(ctx->memory, size);
  VALUE_TO_STRING(val, valueStr, size);

  return valueStr;
}


tensor_size_t calculateNumValuesAndMultipliers(Dim shape, multiplier_t *multipliers) {
  u64 numberOfValues = 1;

  for (int x = shape.numOfDims - 1; x >= 0; x--) {
    if (multipliers != NULL) {
      multipliers[x] = numberOfValues;
    }
    numberOfValues *= shape.dims[x];
  }

  return numberOfValues;
}

bool isSameContext(Context *a, Context *b) {
  return a == b;
}

Result clearTensorValues(Tensor *t) {
  if (t == NULL || t->values == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  if (t->context == NULL || t->context->device == NULL || t->context->device->type == CPU) {
    memset(t->values, 0, valueBytes);
    return OK;
  }

  cudaMemset(t->values, 0, valueBytes);
  return OK;
}

Result initTensor(Context *ctx, Tensor *dest, Dim shape, Dtype dtype) {
  tensor_size_t size = calculateNumValuesAndMultipliers(shape, shape.multipliers);

  *dest = (Tensor){.context = ctx,
                   .dtype = dtype,
                   .values = allocateOnCtx(ctx, size * getBytesForDtype(dtype)),
                   .size = size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

Result initTensorLike(Context *ctx, Tensor *dest, Tensor *src, Dtype dtype) {
  u8 numDims = src->shape.numOfDims;
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * numDims);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * numDims);

  for (u8 i = 0; i < numDims; i++) {
    dims[i] = src->shape.dims[i];
  }

  Dim shape = {.dims = dims, .numOfDims = numDims, .multipliers = multipliers};
  calculateNumValuesAndMultipliers(shape, multipliers);

  return initTensor(ctx, dest, shape, dtype);
}

Result init1DTensor(Context *ctx, Tensor *dest, dim_t size, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t));
  dims[0] = size;
  multipliers[0] = 1;

  return initTensor(ctx, dest, (Dim){.dims = dims, .numOfDims = 1, .multipliers = multipliers},
                    dtype);
}

Result init4DTensor(Context *ctx, Tensor *dest, dim_t d0, dim_t d1, dim_t d2, dim_t d3,
                    Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 4);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * 4);

  dims[0] = d0;
  dims[1] = d1;
  dims[2] = d2;
  dims[3] = d3;

  Dim shape = {.dims = dims, .numOfDims = 4, .multipliers = multipliers};
  calculateNumValuesAndMultipliers(shape, multipliers);

  return initTensor(ctx, dest, shape, dtype);
}

u64 getContigousIdxFromCoord(Tensor *t, dim_t *idx) {
  u64 result = 0;

  for (u8 x = 0; x < t->shape.numOfDims; x++) {
    u64 coord = idx[x];
    if (t->isView && t->boundary) {
      coord += t->boundary[x].start;
    }
    result += coord * t->shape.multipliers[x];
  }

  return result;
}

bool isInvalidTensor(Tensor *t) {
  if (t == NULL || t->values == NULL) {
    return true;
  }
  // 0-dimensional tensors have shape.dims == NULL, which is valid
  if (t->shape.numOfDims > 0 && t->shape.dims == NULL) {
    return true;
  }
  return false;
}

bool isIntType(Tensor *t) {
  return t->dtype != I8 && t->dtype != I16 && t->dtype != I32 && t->dtype != I64 &&
         t->dtype != U8 && t->dtype != U16 && t->dtype != U32 && t->dtype != U64;
}

bool isNotFloatType(Tensor *t) {
  return t->dtype != F16 && t->dtype != F32 && t->dtype != F64;
}

void unravel_index(tensor_size_t flatIdx, Dim *shape, dim_t *destCoords) {
  for (int d = shape->numOfDims - 1; d >= 0; d--) {
    destCoords[d] = flatIdx % shape->dims[d];
    flatIdx /= shape->dims[d];
  }
}

Tensor *copyToContiguous(Context *ctx, Tensor *source) {
  Tensor *copy = t_Zeros(ctx, source->shape, source->dtype);

  size_t *indices = allocate(ctx->memory, sizeof(size_t) * source->shape.numOfDims);
  memset(indices, 0, sizeof(size_t) * source->shape.numOfDims);

  for (tensor_size_t i = 0; i < source->size; i++) {
    Dim idx = {.dims = indices, .numOfDims = source->shape.numOfDims};
    Value val;
    GetAt(source, idx, &val);
    VALUE_SET(copy->values, i, val);

    for (int d = source->shape.numOfDims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < source->shape.dims[d]) {
        break;
      }
      indices[d] = 0;
    }
  }

  freeAlloc(ctx->memory, indices);
  return copy;
}

Result materializeTensorOnContext(Context *ctx, Tensor *src, bool requireContiguous,
                                  TensorArg *arg) {
  if (ctx == NULL || src == NULL || arg == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  arg->tensor = NULL;
  arg->ownsTensor = false;

  if (isSameContext(src->context, ctx) && (!requireContiguous || src->isContigous)) {
    arg->tensor = src;
    return OK;
  }

  Tensor *working = src;
  bool ownsWorking = false;

  if (requireContiguous && !src->isContigous) {
    Context *materializeCtx = src->context != NULL ? src->context : ctx;
    working = copyToContiguous(materializeCtx, src);
    ownsWorking = true;
  }

  if (isSameContext(working->context, ctx)) {
    arg->tensor = working;
    arg->ownsTensor = ownsWorking;
    return OK;
  }

  Tensor *copy = allocate(ctx->memory, sizeof(Tensor));
  Result initResult = initTensorLike(ctx, copy, working, working->dtype);
  if (initResult != OK) {
    if (ownsWorking) {
      FreeTensor(working->context != NULL ? working->context : ctx, working);
    }
    return initResult;
  }

  size_t valueBytes = working->size * getBytesForDtype(working->dtype);
  Result copyResult =
      copyBetweenContexts(working->context, ctx, working->values, copy->values, valueBytes);
  if (copyResult != OK) {
    FreeTensor(ctx, copy);
    if (ownsWorking) {
      FreeTensor(working->context != NULL ? working->context : ctx, working);
    }
    return copyResult;
  }

  if (ownsWorking) {
    FreeTensor(working->context != NULL ? working->context : ctx, working);
  }

  arg->tensor = copy;
  arg->ownsTensor = true;
  return OK;
}

void releaseTensorArg(Context *fallbackCtx, TensorArg *arg) {
  if (arg == NULL || !arg->ownsTensor || arg->tensor == NULL) {
    return;
  }

  Context *freeCtx = arg->tensor->context != NULL ? arg->tensor->context : fallbackCtx;
  FreeTensor(freeCtx, arg->tensor);
  arg->tensor = NULL;
  arg->ownsTensor = false;
}

bool areBroadcastable(Tensor *a, Tensor *b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;

  for (int d = 0; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;

    dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;

    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return false;
    }
  }

  return true;
}

TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b) {
  Tensor *smaller = a->shape.numOfDims < b->shape.numOfDims ? a : b;
  Tensor *larger = a->shape.numOfDims < b->shape.numOfDims ? b : a;
  u8 diff = larger->shape.numOfDims - smaller->shape.numOfDims;

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * larger->shape.numOfDims);
  for (u8 i = 0; i < diff; i++) {
    newDims[i] = 1;
  }
  for (u8 i = 0; i < smaller->shape.numOfDims; i++) {
    newDims[diff + i] = smaller->shape.dims[i];
  }

  Dim newShape = {.dims = newDims, .numOfDims = larger->shape.numOfDims};
  Tensor *reshapedSmaller = allocate(ctx->memory, sizeof(Tensor));
  Reshape(ctx, smaller, reshapedSmaller, newShape);

  if (smaller == a) {
    return (TensorPair){.a = reshapedSmaller, .b = b};
  }
  return (TensorPair){.a = a, .b = reshapedSmaller};
}

Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result) {
  if (dim == 0) {
    *result = 1;
    return OK;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t totalElements = t->shape.dims[0];
  for (dim_t d = 1; d < t->shape.numOfDims; d++) {
    if (d == dim) {
      break;
    }
    totalElements *= t->shape.dims[d];
  }

  *result = totalElements;
  return OK;
}

Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result) {
  if (dim == 0) {
    return OK;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  for (dim_t d = 0; d < dim; d++) {
    result->dims[d] = t->shape.dims[d];
  }

  return OK;
}

Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result) {
  if (dim == t->shape.numOfDims - 1) {
    *result = 1;
    return OK;
  }

  if (dim >= t->size) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t numElements = t->shape.dims[dim + 1];
  for (dim_t d = dim + 2; d < t->shape.numOfDims; d++) {
    if (d >= t->shape.numOfDims) {
      break;
    }
    numElements *= t->shape.dims[d];
  }

  *result = numElements;
  return OK;
}

void accumulateStridedByDtype(Dtype dtype, void *destValues, u64 destBase, u64 destStep,
                              void *srcValues, u64 srcBase, u64 srcStep, u64 count) {
  switch (dtype) {
    case BOOL: {
      bool *d = (bool *)destValues;
      bool *s = (bool *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] = d[destBase + i * destStep] || s[srcBase + i * srcStep];
      }
      break;
    }
    case U8: {
      u8 *d = (u8 *)destValues;
      u8 *s = (u8 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U16: {
      u16 *d = (u16 *)destValues;
      u16 *s = (u16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U32: {
      u32 *d = (u32 *)destValues;
      u32 *s = (u32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U64: {
      u64 *d = (u64 *)destValues;
      u64 *s = (u64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I8: {
      i8 *d = (i8 *)destValues;
      i8 *s = (i8 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I16: {
      i16 *d = (i16 *)destValues;
      i16 *s = (i16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I32: {
      i32 *d = (i32 *)destValues;
      i32 *s = (i32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I64: {
      i64 *d = (i64 *)destValues;
      i64 *s = (i64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F16: {
      f16 *d = (f16 *)destValues;
      f16 *s = (f16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F32: {
      f32 *d = (f32 *)destValues;
      f32 *s = (f32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F64: {
      f64 *d = (f64 *)destValues;
      f64 *s = (f64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
  }
}

Result init2DTensor(Context *ctx, Tensor *dest, dim_t rows, dim_t cols, Dtype dtype) {
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t) * 2);
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * 2);

  dims[0] = rows;
  dims[1] = cols;

  Dim shape = {.dims = dims, .numOfDims = 2, .multipliers = multipliers};
  calculateNumValuesAndMultipliers(shape, multipliers);

  return initTensor(ctx, dest, shape, dtype);
}

Result moveTensor(Context *srcCtx, Context *destCtx, Tensor *t) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (destCtx == NULL) {
    return ERR_COPY_CTX_DEVICE_IS_NULL;
  }

  Context *tensorCtx = t->context != NULL ? t->context : srcCtx;
  if (tensorCtx == NULL) {
    return ERR_COPY_CTX_DEVICE_IS_NULL;
  }

  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  void *locationOnTarget = allocateOnCtx(destCtx, valueBytes);
  Result copyResult =
      copyBetweenContexts(tensorCtx, destCtx, t->values, locationOnTarget, valueBytes);
  if (copyResult != OK) {
    return copyResult;
  }

  freeOnCtx(tensorCtx, t->values);
  t->values = locationOnTarget;
  t->context = destCtx;

  return OK;
}
