#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/value.h"
#include "tensor_internal.h"
#include <complex.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void seedRandomOnce(void) {
  static bool seeded = false;
  if (seeded) {
    return;
  }

  seeded = true;
  srand((unsigned int)(time(NULL) ^ (time_t)clock()));
}

static u64 nextRandomBits(void) {
  u64 a = (u64)(unsigned int)rand();
  u64 b = (u64)(unsigned int)rand();
  u64 c = (u64)(unsigned int)rand();
  return (a << 42) ^ (b << 21) ^ c;
}

static f64 nextRandomUnit(void) {
  return (f64)nextRandomBits() / (f64)UINT64_MAX;
}

static Value randomValueForRange(f32 minValue, f32 maxValue, Dtype dtype) {
  switch (dtype) {
    case BOOL: {
      bool minBool = minValue != 0.0f;
      bool maxBool = maxValue != 0.0f;
      if (minBool == maxBool) {
        return (Value){.dtype = BOOL, .as.boolean = minBool};
      }
      return (Value){.dtype = BOOL, .as.boolean = nextRandomBits() % 2 == 0};
    }
    case U8: {
      u8 min = (u8)minValue;
      u8 max = (u8)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (Value){.dtype = U8, .as.u8 = (u8)(min + (u8)(nextRandomBits() % range))};
    }
    case U16: {
      u16 min = (u16)minValue;
      u16 max = (u16)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (Value){.dtype = U16, .as.u16 = (u16)(min + (u16)(nextRandomBits() % range))};
    }
    case U32: {
      u32 min = (u32)minValue;
      u32 max = (u32)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (Value){.dtype = U32, .as.u32 = (u32)(min + (u32)(nextRandomBits() % range))};
    }
    case U64: {
      u64 min = (u64)minValue;
      u64 max = (u64)maxValue;
      u64 range = max - min;
      if (range == UINT64_MAX) {
        return (Value){.dtype = U64, .as.u64 = nextRandomBits()};
      }
      return (Value){.dtype = U64, .as.u64 = min + (nextRandomBits() % (range + 1))};
    }
    case I8: {
      i8 min = (i8)minValue;
      i8 max = (i8)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (Value){.dtype = I8, .as.i8 = (i8)(min + (i8)(nextRandomBits() % (u64)range))};
    }
    case I16: {
      i16 min = (i16)minValue;
      i16 max = (i16)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (Value){.dtype = I16, .as.i16 = (i16)(min + (i16)(nextRandomBits() % (u64)range))};
    }
    case I32: {
      i32 min = (i32)minValue;
      i32 max = (i32)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (Value){.dtype = I32, .as.i32 = (i32)(min + (i32)(nextRandomBits() % (u64)range))};
    }
    case I64: {
      i64 min = (i64)minValue;
      i64 max = (i64)maxValue;
      u64 range = (u64)max - (u64)min;
      if (range == UINT64_MAX) {
        return (Value){.dtype = I64, .as.i64 = (i64)nextRandomBits()};
      }
      return (Value){.dtype = I64, .as.i64 = min + (i64)(nextRandomBits() % (range + 1))};
    }
    case F16: {
      f64 scale = (f64)maxValue - (f64)minValue;
      return (Value){.dtype = F16, .as.f16 = (f16)(minValue + (f32)(nextRandomUnit() * scale))};
    }
    case F32: {
      f64 scale = (f64)maxValue - (f64)minValue;
      return (Value){.dtype = F32, .as.f32 = minValue + (f32)(nextRandomUnit() * scale)};
    }
    case F64: {
      f64 min = (f64)minValue;
      f64 max = (f64)maxValue;
      f64 scale = max - min;
      return (Value){.dtype = F64, .as.f64 = min + nextRandomUnit() * scale};
    }
  }

  return VALUE(dtype, minValue);
}

static Result initTensor(Context *ctx, Tensor *dest, Dim shape, Dtype dtype) {
  if (dest == NULL) {
    return ERR_NULL_PTR;
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, shape.dims, shape.numOfDims);
  shape.multipliers = snm.multipliers;

  void *values = allocateOnCtx(ctx, snm.size * getBytesForDtype(dtype));
  Result allocRes = ensureAllocated(values);
  if (allocRes != OK) {
    return allocRes;
  }

  *dest = (Tensor){.context = ctx,
                   .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                   .dtype = dtype,
                   .values = values,
                   .size = snm.size,
                   .shape = shape,
                   .isView = false,
                   .isContigous = true,
                   .boundary = NULL};

  return OK;
}

Tensor *t_Empty(Context *ctx, Dim shape, Dtype type) {
  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(t == NULL, ALLOCATION_FAILED);
  PANIC_IF(initTensor(ctx, t, shape, type) != OK, ALLOCATION_FAILED);
  return t;
}

static Tensor *zeroTensorWithGrad(Context *ctx, Dim shape, Dtype type, bool withGrad) {
  Dim tShape = {.numOfDims = shape.numOfDims};
  if (shape.numOfDims > 0) {
    tShape.dims = allocate(ctx->memory, sizeof(dim_t) * shape.numOfDims);
    PANIC_IF(tShape.dims == NULL, ALLOCATION_FAILED);
    memcpy(tShape.dims, shape.dims, sizeof(dim_t) * shape.numOfDims);
  } else {
    tShape.dims = NULL;
  }

  Tensor *t = t_Empty(ctx, tShape, type);
  PANIC_IF(clearTensorValues(t) != OK, ALLOCATION_FAILED);
  if (withGrad) {
    t->grad = zeroTensorWithGrad(ctx, tShape, F32, false);
  }
  return t;
}

Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type) {
  return zeroTensorWithGrad(ctx, shape, type, true);
}

Tensor *t_Reduced(Context *ctx, Tensor *source, dim_t dim, Dtype type) {
  PANIC_IF(source == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim >= source->shape.numOfDims, ERR_OUT_OF_BOUNDS);

  dim_t *dims = NULL;
  if (source->shape.numOfDims > 0) {
    dims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
    PANIC_IF(dims == NULL, ALLOCATION_FAILED);
    memcpy(dims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);
    dims[dim] = 1;
  }

  Tensor *t = t_Empty(ctx, SHAPE(dims, source->shape.numOfDims), type);
  t->grad = zeroTensorWithGrad(ctx, t->shape, F32, false);
  return t;
}

Tensor *T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, F32);
}

Result Clone(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Tensor *source = t;
  if (!t->isContigous) {
    source = copyToContiguous(ctx, t);
    if (source == NULL) {
      return ERR_OUT_OF_MEMORY;
    }
  }

  size_t valueBytes = getBytesForDtype(source->dtype) * source->size;
  void *newValues = allocateOnCtx(ctx, valueBytes);
  Result allocRes = ensureAllocated(newValues);
  if (allocRes != OK) {
    if (!t->isContigous) {
      FreeTensor(ctx, source);
    }
    return allocRes;
  }
  Result valueCopyRes =
      copyBetweenContexts(source->context, ctx, source->values, newValues, valueBytes);
  if (valueCopyRes != OK) {
    freeOnCtx(ctx, newValues);
    if (!t->isContigous) {
      FreeTensor(ctx, source);
    }
    return valueCopyRes;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  allocRes = ensureAllocated(newDims);
  if (allocRes != OK) {
    freeOnCtx(ctx, newValues);
    if (!t->isContigous) {
      FreeTensor(ctx, source);
    }
    return allocRes;
  }
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  allocRes = ensureAllocated(newMultipliers);
  if (allocRes != OK) {
    freeAlloc(ctx->memory, newDims);
    freeOnCtx(ctx, newValues);
    if (!t->isContigous) {
      FreeTensor(ctx, source);
    }
    return allocRes;
  }
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  *dest = (Tensor){.context = ctx,
                   .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                   .dtype = source->dtype,
                   .values = newValues,
                   .size = source->size,
                   .isContigous = true,
                   .isView = false,
                   .boundary = NULL,
                   .shape = {.dims = newDims,
                             .numOfDims = source->shape.numOfDims,
                             .multipliers = newMultipliers}};
  if (!t->isContigous) {
    FreeTensor(ctx, source);
  }
  return OK;
}

Result Copy(Context *ctx, Tensor *src, Tensor *dest) {
  if (isInvalidTensor(src) || isInvalidTensor(dest)) {
    return ERR_COPY_REQUIRES_INITIALIZED_TENSORS;
  }

  if (dest->isView) {
    return ERR_COPY_DESTINATION_VIEW;
  }

  if (src->size != dest->size) {
    return ERR_COPY_REQUIRES_TENSORS_OF_THE_SAME_SIZE;
  }

  if (src->dtype != dest->dtype) {
    return ERR_COPY_SAME_DTYPE;
  }

  Tensor *srcContigous;
  if (!src->isContigous) {
    srcContigous = copyToContiguous(ctx, src);
    if (srcContigous == NULL) {
      return ERR_OUT_OF_MEMORY;
    }
  } else {
    srcContigous = src;
  }

  Result copyRes =
      copyBetweenContexts(srcContigous->context, dest->context, srcContigous->values, dest->values,
                          srcContigous->size * getBytesForDtype(srcContigous->dtype));
  if (copyRes != OK) {
    if (!src->isContigous) {
      FreeTensor(ctx, srcContigous);
    }
    return copyRes;
  }

  if (!src->isContigous) {
    // Free the intermediate contigous tensor
    FreeTensor(ctx, srcContigous);
  }

  return OK;
}

void SetValues(Tensor *t, Value value) {
  if (t->context != NULL && t->context->device != NULL && t->context->device->type == CUDA) {
    Result result = runCudaFillTensor(t->context, t->dtype, t->values, t->size, value);
    if (result == OK) {
      return;
    }
  }

  for (tensor_size_t i = 0; i < t->size; i++) {
    VALUE_SET(t->values, i, value);
  }
}

Tensor *T_Int(Context *ctx, Dim shape, i8 initialValue) {
  Tensor *init = t_Zeros(ctx, shape, I8);
  Value v = (Value){.dtype = I8, .as.i8 = initialValue};
  SetValues(init, v);
  return init;
}

Tensor *T_UInt(Context *ctx, Dim shape, u8 initialValue) {
  Tensor *init = t_Zeros(ctx, shape, U8);
  Value v = (Value){.dtype = U8, .as.u8 = initialValue};
  SetValues(init, v);
  return init;
}

Tensor *T_Float(Context *ctx, Dim shape, f32 initialValue) {
  Tensor *init = t_Zeros(ctx, shape, F32);
  Value v = (Value){.dtype = F32, .as.f32 = initialValue};
  SetValues(init, v);
  return init;
}

Tensor *MakeFromContigousArray(Context *ctx, Dim shape, void *values, tensor_size_t numElements, Dtype dtype) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(values == NULL, ERR_NULL_PTR);
  PANIC_IF(numElements == 0, ERR_NO_OP);
  PANIC_IF(ctx->device != NULL && ctx->device->type == CUDA, ERR_NO_OP);

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, shape.dims, shape.numOfDims);

  PANIC_IF(snm.size != numElements, ERR_RESHAPE_DIM_MISMATCH);

  Tensor *tensor = t_Zeros(ctx, shape, dtype);
  PANIC_IF(tensor == NULL, ALLOCATION_FAILED);

  memcpy(tensor->values, values, numElements * getBytesForDtype(dtype));

  return tensor;
}

Tensor *MakeRandomTensor(Context *ctx, Dim shape, f32 minValue, f32 maxValue, Dtype dtype) {
  PANIC_IF(minValue > maxValue, ERR_INVALID_RANGE);
  PANIC_IF(ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA, ERR_NO_OP);

  Tensor *tensor = t_Zeros(ctx, shape, dtype);
  PANIC_IF(tensor == NULL, ALLOCATION_FAILED);

  seedRandomOnce();
  for (tensor_size_t i = 0; i < tensor->size; i++) {
    VALUE_SET(tensor->values, i, randomValueForRange(minValue, maxValue, dtype));
  }

  return tensor;
}

Tensor *T_Arange(Context *ctx, f32 start, f32 end, f32 step) {
  if (step == 0.0f) {
    step = 1.0f;
  }

  // Calculate number of elements
  // Use a small epsilon to handle floating-point precision issues
  const f32 eps = 1e-6f;
  tensor_size_t n = 0;
  if (step > 0) {
    if (start >= end) {
      return NULL;
    }
    n = (tensor_size_t)((end - start + eps) / step);
    // Ensure we don't include values >= end
    while (n > 0 && start + (n - 1) * step >= end - eps) {
      n--;
    }
  } else {
    if (start <= end) {
      return NULL;
    }
    n = (tensor_size_t)((start - end + eps) / (-step));
    // Ensure we don't include values <= end
    while (n > 0 && start + (n - 1) * step <= end + eps) {
      n--;
    }
  }

  if (n <= 0) {
    return NULL;
  }

  Tensor *t = t_Zeros(ctx, SHAPE1D(n), F32);
  PANIC_IF(t == NULL, ALLOCATION_FAILED);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    Result result = runCudaArange(ctx, start, step, t->values, n);
    if (result == OK) {
      return t;
    }
  }

  f32 *values = (f32 *)t->values;
  for (tensor_size_t i = 0; i < n; i++) {
    values[i] = start + (f32)i * step;
  }

  return t;
}

Tensor *T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses) {
  if (isInvalidTensor(indices)) {
    return NULL;
  }

  Tensor *source = materializeTensorOnContext(ctx, indices);

  // Build output shape: input shape + [numClasses]
  u8 outNumDims = source->shape.numOfDims + 1;
  dim_t *outDims = allocate(ctx->memory, sizeof(dim_t) * outNumDims);
  PANIC_IF(outDims == NULL, ALLOCATION_FAILED);


  for (u8 i = 0; i < source->shape.numOfDims; i++) {
    outDims[i] = source->shape.dims[i];
  }
  outDims[outNumDims - 1] = numClasses;

  // Create output tensor filled with zeros
  Tensor *out = T_Zeros(ctx, SHAPE(outDims, outNumDims)); 

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    Result result =
        runCudaOneHot(ctx, source->dtype, source->values, source->size, numClasses, out->values);
    freeIfContingousCopy(ctx, source);
    PANIC_IF(result != OK, result);     
    return out;
  }

  // Set one-hot values
  // For each element in indices, set the corresponding position to 1.0
  dim_t lastDimStride = numClasses;
  for (tensor_size_t i = 0; i < source->size; i++) {
    Value idxVal;
    Result readResult = readTensorValueAtFlatIndex(source, i, &idxVal);
    PANIC_IF(readResult != OK, readResult); 

    // Convert index to i64 for bounds checking
    i64 classIdx = 0;
    switch (idxVal.dtype) {
      case I8: classIdx = (i64)idxVal.as.i8; break;
      case I16: classIdx = (i64)idxVal.as.i16; break;
      case I32: classIdx = (i64)idxVal.as.i32; break;
      case I64: classIdx = (i64)idxVal.as.i64; break;
      case U8: classIdx = (i64)idxVal.as.u8; break;
      case U16: classIdx = (i64)idxVal.as.u16; break;
      case U32: classIdx = (i64)idxVal.as.u32; break;
      case U64: classIdx = (i64)idxVal.as.u64; break;
      default: classIdx = (i64)idxVal.as.f32; break; // For float types
    }

    // Skip if out of bounds (could also error, but we'll skip)
    if (classIdx < 0 || classIdx >= (i64)numClasses) {
      continue;
    }

    tensor_size_t outIdx = i * lastDimStride + (tensor_size_t)classIdx;
    Result writeResult =
        writeTensorValueAtFlatIndex(out, outIdx, (Value){.dtype = F32, .as.f32 = 1.0f});
    PANIC_IF(writeResult != OK, ERR_NO_OP);
  }

  freeIfContingousCopy(ctx, source);
  return out;
}
