#include "olib.h"
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "value.h"
#include "shapes_internal.h"

#ifdef SHAPES_HAS_CUDA 
#include "shapescuda.h"
#endif
#include <complex.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static u64 nodeIdCounter = 0;

u64 nextNodeId(void) {
  return ++nodeIdCounter;
}

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

static shapes_Value randomValueForRange(f32 minValue, f32 maxValue, shapes_Dtype dtype) {
  switch (dtype) {
    case BOOL: {
      bool minBool = minValue != 0.0f;
      bool maxBool = maxValue != 0.0f;
      if (minBool == maxBool) {
        return (shapes_Value){.dtype = BOOL, .as.boolean = minBool};
      }
      return (shapes_Value){.dtype = BOOL, .as.boolean = nextRandomBits() % 2 == 0};
    }
    case U8: {
      u8 min = (u8)minValue;
      u8 max = (u8)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (shapes_Value){.dtype = U8, .as.u8 = (u8)(min + (u8)(nextRandomBits() % range))};
    }
    case U16: {
      u16 min = (u16)minValue;
      u16 max = (u16)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (shapes_Value){.dtype = U16, .as.u16 = (u16)(min + (u16)(nextRandomBits() % range))};
    }
    case U32: {
      u32 min = (u32)minValue;
      u32 max = (u32)maxValue;
      u64 range = (u64)max - (u64)min + 1;
      return (shapes_Value){.dtype = U32, .as.u32 = (u32)(min + (u32)(nextRandomBits() % range))};
    }
    case U64: {
      u64 min = (u64)minValue;
      u64 max = (u64)maxValue;
      u64 range = max - min;
      if (range == UINT64_MAX) {
        return (shapes_Value){.dtype = U64, .as.u64 = nextRandomBits()};
      }
      return (shapes_Value){.dtype = U64, .as.u64 = min + (nextRandomBits() % (range + 1))};
    }
    case I8: {
      i8 min = (i8)minValue;
      i8 max = (i8)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (shapes_Value){.dtype = I8, .as.i8 = (i8)(min + (i8)(nextRandomBits() % (u64)range))};
    }
    case I16: {
      i16 min = (i16)minValue;
      i16 max = (i16)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (shapes_Value){.dtype = I16, .as.i16 = (i16)(min + (i16)(nextRandomBits() % (u64)range))};
    }
    case I32: {
      i32 min = (i32)minValue;
      i32 max = (i32)maxValue;
      i64 range = (i64)max - (i64)min + 1;
      return (shapes_Value){.dtype = I32, .as.i32 = (i32)(min + (i32)(nextRandomBits() % (u64)range))};
    }
    case I64: {
      i64 min = (i64)minValue;
      i64 max = (i64)maxValue;
      u64 range = (u64)max - (u64)min;
      if (range == UINT64_MAX) {
        return (shapes_Value){.dtype = I64, .as.i64 = (i64)nextRandomBits()};
      }
      return (shapes_Value){.dtype = I64, .as.i64 = min + (i64)(nextRandomBits() % (range + 1))};
    }
    case F16: {
      f64 scale = (f64)maxValue - (f64)minValue;
      return (shapes_Value){.dtype = F16, .as.f16 = (f16)(minValue + (f32)(nextRandomUnit() * scale))};
    }
    case F32: {
      f64 scale = (f64)maxValue - (f64)minValue;
      return (shapes_Value){.dtype = F32, .as.f32 = minValue + (f32)(nextRandomUnit() * scale)};
    }
    case F64: {
      f64 min = (f64)minValue;
      f64 max = (f64)maxValue;
      f64 scale = max - min;
      return (shapes_Value){.dtype = F64, .as.f64 = min + nextRandomUnit() * scale};
    }
  }

  return VALUE(dtype, minValue);
}

static inline void *allocateTensorValues(shapes_Context *ctx, size_t size) {

  #ifdef SHAPES_HAS_CUDA 
  if (ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA) {
     shapescuda_Block block =  shapescuda_Allocate(&ctx->cudaMemory, ctx->cudaMetadataMemory, size);
    return block.ptr;
  }
  #endif
  return olib_Allocate(ctx->memory, size);
}

static Result initTensor(shapes_Context *ctx, shapes_Tensor *dest, shapes_Dim shape, shapes_Dtype dtype) {
  if (dest == NULL) {
    return ERR_NULL_PTR;
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, shape.dims, shape.numOfDims);
  shape.multipliers = snm.multipliers;

  void *values = allocateTensorValues(ctx, snm.size * getBytesForDtype(dtype));

  *dest = (shapes_Tensor){
      .context = ctx,
      .metadataMemory = ctx != NULL ? ctx->memory : NULL,
      .dtype = dtype,
      .values = values,
      .size = snm.size,
      .shape = shape,
      .isView = false,
      .isContigous = true,
      .boundary = NULL,
      .inputs = NULL,
      .opType = OP_NONE,
      .nodeId = nextNodeId(),
  };

  return OK;
}

shapes_Tensor t_Empty(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype type) {
  shapes_Tensor t = {};
  PANIC_IF(initTensor(ctx, &t, shape, type) != OK, ALLOCATION_FAILED);
  return t;
}

static shapes_Tensor zeroTensorWithGrad(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype type, bool withGrad) {
  shapes_Dim tShape = {.numOfDims = shape.numOfDims};
  if (shape.numOfDims > 0) {
    tShape.dims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * shape.numOfDims);
    PANIC_IF(tShape.dims == NULL, ALLOCATION_FAILED);
    memcpy(tShape.dims, shape.dims, sizeof(shapes_dim_t) * shape.numOfDims);
  } else {
    tShape.dims = NULL;
  }

  shapes_Tensor t = t_Empty(ctx, tShape, type);
  PANIC_IF(clearTensorValues(&t) != OK, ALLOCATION_FAILED);
  if (withGrad) {
    shapes_Tensor *gradPtr = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
    PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
    *gradPtr = zeroTensorWithGrad(ctx, tShape, F32, false);
    t.grad = gradPtr;
  }
  return t;
}

shapes_Tensor t_Zeros(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype type) {
  return zeroTensorWithGrad(ctx, shape, type, true);
}

shapes_Tensor t_Reduced(shapes_Context *ctx, shapes_Tensor *source, shapes_dim_t dim, shapes_Dtype type) {
  PANIC_IF(source == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim >= source->shape.numOfDims, ERR_OUT_OF_BOUNDS);

  shapes_dim_t *dims = NULL;
  if (source->shape.numOfDims > 0) {

    dims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * source->shape.numOfDims);
    PANIC_IF(dims == NULL, ALLOCATION_FAILED);
    memcpy(dims, source->shape.dims, sizeof(shapes_dim_t) * source->shape.numOfDims);
    dims[dim] = 1;
  }

  shapes_Tensor t = t_Empty(ctx, SHAPE(dims, source->shape.numOfDims), type);
  shapes_Tensor *gradPtr = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
  *gradPtr = zeroTensorWithGrad(ctx, t.shape, F32, false);
  t.grad = gradPtr;
  return t;
}

shapes_Tensor shapes_MakeZerosTensor(shapes_Context *ctx, shapes_Dim shape) {
  return t_Zeros(ctx, shape, F32);
}

shapes_Tensor shapes_MakeZerosTensorWithDtype(shapes_Context *ctx, shapes_Dim shape, shapes_Dtype dtype) {
  return t_Zeros(ctx, shape, dtype); 
}

shapes_Tensor shapes_Clone(shapes_Context *ctx, shapes_Tensor *t) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor *source = t;
  if (!t->isContigous) {
    source = copyToContiguous(ctx, t);
    PANIC_IF(source == NULL, ERR_OUT_OF_MEMORY);
  }

  size_t valueBytes = getBytesForDtype(source->dtype) * source->size;
  void *newValues = allocateTensorValues(ctx, valueBytes);
  PANIC_IF(newValues == NULL, ALLOCATION_FAILED);

  Result valueCopyRes = shapes_CopyBetweenDevices(source->context->device->type, ctx->device->type, source->values, newValues, valueBytes);
  PANIC_IF(valueCopyRes != OK, valueCopyRes);

  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * source->shape.numOfDims);
  PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
  memcpy(newDims, source->shape.dims, sizeof(shapes_dim_t) * source->shape.numOfDims);

  shapes_multiplier_t *newMultipliers = olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t) * source->shape.numOfDims);
  PANIC_IF(newMultipliers == NULL, ALLOCATION_FAILED);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(shapes_multiplier_t) * source->shape.numOfDims);

  shapes_Tensor dest = (shapes_Tensor){.context = ctx,
                         .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                         .dtype = source->dtype,
                         .values = newValues,
                         .size = source->size,
                         .isContigous = true,
                         .isView = false,
                         .boundary = NULL,
                         .shape = {.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
                         .nodeId = nextNodeId()};
  return dest;
}

void shapes_Copy(shapes_Context *ctx, shapes_Tensor *src, shapes_Tensor *dest) {
  PANIC_IF(isInvalidTensor(src) || isInvalidTensor(dest), ERR_COPY_REQUIRES_INITIALIZED_TENSORS);
  PANIC_IF(dest->isView, ERR_COPY_DESTINATION_VIEW);
  PANIC_IF(src->size != dest->size, ERR_COPY_REQUIRES_TENSORS_OF_THE_SAME_SIZE);
  PANIC_IF(src->dtype != dest->dtype, ERR_COPY_SAME_DTYPE);

  shapes_Tensor *srcContigous;
  if (!src->isContigous) {
    srcContigous = copyToContiguous(ctx, src);
    PANIC_IF(srcContigous == NULL, ERR_OUT_OF_MEMORY);
  } else {
    srcContigous = src;
  }

  Result copyRes =
      shapes_CopyBetweenDevices(srcContigous->context->device->type, dest->context->device->type, srcContigous->values, dest->values, srcContigous->size * getBytesForDtype(srcContigous->dtype));
  PANIC_IF(copyRes != OK, copyRes);
}

void shapes_SetValues(shapes_Tensor *t, shapes_Value value) {
  if (t->context != NULL && t->context->device != NULL && t->context->device->type == CUDA) {
    Result result = shapescuda_FillTensor(t->dtype, t->values, t->size, value);
    if (result == OK) {
      return;
    }
  }

  for (shapes_tensor_size_t i = 0; i < t->size; i++) {
    VALUE_SET(t->values, i, value);
  }
}

shapes_Tensor shapes_MakeIntTensor(shapes_Context *ctx, shapes_Dim shape, i8 initialValue) {
  shapes_Tensor init = t_Zeros(ctx, shape, I8);
  shapes_Value v = (shapes_Value){.dtype = I8, .as.i8 = initialValue};
  shapes_SetValues(&init, v);
  return init;
}

shapes_Tensor shapes_MakeUIntTensor(shapes_Context *ctx, shapes_Dim shape, u8 initialValue) {
  shapes_Tensor init = t_Zeros(ctx, shape, U8);
  shapes_Value v = (shapes_Value){.dtype = U8, .as.u8 = initialValue};
  shapes_SetValues(&init, v);
  return init;
}

shapes_Tensor shapes_MakeFloatTensor(shapes_Context *ctx, shapes_Dim shape, f32 initialValue) {
  shapes_Tensor init = t_Zeros(ctx, shape, F32);
  shapes_Value v = (shapes_Value){.dtype = F32, .as.f32 = initialValue};
  shapes_SetValues(&init, v);
  return init;
}

shapes_Tensor shapes_MakeFloat64Tensor(shapes_Context *ctx, shapes_Dim shape, f64 initialValue) {
  shapes_Tensor init = t_Zeros(ctx, shape, F64);
  shapes_Value v = (shapes_Value){.dtype = F64, .as.f64 = initialValue};
  shapes_SetValues(&init, v);
  return init;
}

shapes_Tensor shapes_MakeFromContigousArray(shapes_Context *ctx, shapes_Dim shape, void *values, shapes_Dtype dtype) {
  PANIC_IF(ctx == NULL, ERR_NULL_PTR);
  PANIC_IF(values == NULL, ERR_NULL_PTR);

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, shape.dims, shape.numOfDims);

  shapes_Tensor tensor = t_Zeros(ctx, shape, dtype);

  size_t valueBytes = snm.size * getBytesForDtype(dtype);

  // Assume the values buffer is created on the host then copy it to the devices on the context, which is where the tensor will get created as well
  Result copyRes = shapes_CopyBetweenDevices(CPU, ctx->device->type, values, tensor.values, valueBytes);
  PANIC_IF(copyRes != OK, copyRes);

  return tensor;
}

shapes_Tensor shapes_MakeRandomTensor(shapes_Context *ctx, shapes_Dim shape, f32 minValue, f32 maxValue, shapes_Dtype dtype) {
  PANIC_IF(minValue > maxValue, ERR_INVALID_RANGE);

  shapes_Tensor tensor = t_Zeros(ctx, shape, dtype);

  seedRandomOnce();

  for (RANGE(i, tensor.size)) {
    shapes_Value nextValue = randomValueForRange(minValue, maxValue, dtype);
    VALUE_SET(tensor.values, i, nextValue);
  }

  return tensor;
}

shapes_Tensor shapes_MakeArangeTensor(shapes_Context *ctx, f32 start, f32 end, f32 step) {
  if (step == 0.0f) {
    step = 1.0f;
  }

  // Calculate number of elements
  // Use a small epsilon to handle floating-point precision issues
  const f32 eps = 1e-6f;
  shapes_tensor_size_t n = 0;
  if (step > 0) {
    PANIC_IF(start >= end, ERR_INVALID_RANGE);
    n = (shapes_tensor_size_t)((end - start + eps) / step);
    while (n > 0 && start + (n - 1) * step >= end - eps) {
      n--;
    }
  } else {
    PANIC_IF(start <= end, ERR_INVALID_RANGE);
    n = (shapes_tensor_size_t)((start - end + eps) / (-step));
    while (n > 0 && start + (n - 1) * step <= end + eps) {
      n--;
    }
  }

  PANIC_IF(n == 0, ERR_INVALID_RANGE);

  shapes_Tensor t = t_Zeros(ctx, SHAPE1D(n), F32);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    Result result = shapescuda_Arange(start, step, t.values, n);
    if (result == OK) {
      return t;
    }
  }

  f32 *values = (f32 *)t.values;
  for (shapes_tensor_size_t i = 0; i < n; i++) {
    values[i] = start + (f32)i * step;
  }

  return t;
}

shapes_Tensor shapes_MakeOneHotTensor(shapes_Context *ctx, shapes_Tensor *indices, shapes_dim_t numClasses) {
  PANIC_IF(isInvalidTensor(indices), ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor *source = materializeTensorOnContext(ctx, indices);

  // Build output shape: input shape + [numClasses]
  u8 outNumDims = source->shape.numOfDims + 1;
  shapes_dim_t *outDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * outNumDims);
  PANIC_IF(outDims == NULL, ALLOCATION_FAILED);

  for (u8 i = 0; i < source->shape.numOfDims; i++) {
    outDims[i] = source->shape.dims[i];
  }
  outDims[outNumDims - 1] = numClasses;

  // Create output tensor filled with zeros
  shapes_Tensor out = shapes_MakeZerosTensor(ctx, SHAPE(outDims, outNumDims));

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    Result result = shapescuda_OneHot(source->dtype, source->values, source->size, numClasses, out.values);
    PANIC_IF(result != OK, result);
    return out;
  }

  // Set one-hot values
  // For each element in indices, set the corresponding position to 1.0
  shapes_dim_t lastDimStride = numClasses;
  for (shapes_tensor_size_t i = 0; i < source->size; i++) {
    shapes_Value idxVal;
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

    shapes_tensor_size_t outIdx = i * lastDimStride + (shapes_tensor_size_t)classIdx;
    Result writeResult = writeTensorValueAtFlatIndex(&out, outIdx, (shapes_Value){.dtype = F32, .as.f32 = 1.0f});
    PANIC_IF(writeResult != OK, ERR_NO_OP);
  }

  return out;
}
