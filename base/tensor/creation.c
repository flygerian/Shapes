#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/value.h"
#include "tensor_internal.h"
#include "../memory.h"
#include <complex.h>
#include <stddef.h>
#include <string.h>


Tensor *t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims = shape.numOfDims};

  // Handle 0-dimensional tensor (scalar)
  if (shape.numOfDims == 0) {
    tShape.dims = NULL;
    tShape.multipliers = NULL;

    Tensor *t = allocate(ctx->memory, sizeof(Tensor));
    initTensor(ctx, t, tShape, type);
    clearTensorValues(t);

    return t;
  }

  tShape.dims = allocate(ctx->memory, sizeof(dim_t) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(multiplier_t) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(dim_t) * shape.numOfDims);
  tensor_size_t size = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);
  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  initTensor(ctx, t, tShape, type);
  clearTensorValues(t);

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
  }

  size_t valueBytes = getBytesForDtype(source->dtype) * source->size;
  void *newValues = allocateOnCtx(ctx, valueBytes);
  Result valueCopyRes =
      copyBetweenContexts(source->context, ctx, source->values, newValues, valueBytes);
  if (valueCopyRes != OK) {
    return valueCopyRes;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
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
  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  if (t->context != NULL && t->context->device != NULL && t->context->device->type == CUDA) {
    void *hostValues = allocate(t->context->memory, valueBytes);
    for (tensor_size_t i = 0; i < t->size; i++) {
      VALUE_SET(hostValues, i, value);
    }

    copyBetweenContexts(NULL, t->context, hostValues, t->values, valueBytes);
    freeAlloc(t->context->memory, hostValues);
    return;
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

  // Create 1D shape
  dim_t *dims = allocate(ctx->memory, sizeof(dim_t));
  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t));
  *dims = (dim_t)n;
  *multipliers = 1;

  Dim shape = {.dims = dims, .numOfDims = 1, .multipliers = multipliers};

  // Allocate and fill values
  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  size_t bytesRequired = n * sizeof(f32);
  initTensor(ctx, t, shape, F32);

  if (ctx->device != NULL && ctx->device->type == CUDA) {
    f32 *hostValues = allocate(ctx->memory, bytesRequired);
    for (tensor_size_t i = 0; i < n; i++) {
      hostValues[i] = start + (f32)i * step;
    }
    copyBetweenContexts(NULL, ctx, hostValues, t->values, bytesRequired);
    freeAlloc(ctx->memory, hostValues);
  } else {
    f32 *values = (f32 *)t->values;
    for (tensor_size_t i = 0; i < n; i++) {
      values[i] = start + (f32)i * step;
    }
  }

  return t;
}

Tensor *T_OneHot(Context *ctx, Tensor *indices, dim_t numClasses) {
  if (isInvalidTensor(indices)) {
    return NULL;
  }

  TensorArg sourceArg = {0};
  if (materializeTensorOnContext(ctx, indices, true, &sourceArg) != OK) {
    return NULL;
  }

  Tensor *source = sourceArg.tensor;
  // Build output shape: input shape + [numClasses]
  u8 outNumDims = source->shape.numOfDims + 1;
  dim_t *outDims = allocate(ctx->memory, sizeof(dim_t) * outNumDims);
  multiplier_t *outMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * outNumDims);

  for (u8 i = 0; i < source->shape.numOfDims; i++) {
    outDims[i] = source->shape.dims[i];
  }
  outDims[outNumDims - 1] = numClasses;

  Dim outShape = {.dims = outDims, .numOfDims = outNumDims, .multipliers = outMultipliers};
  tensor_size_t outSize = calculateNumValuesAndMultipliers(outShape, outMultipliers);

  // Create output tensor filled with zeros
  Tensor *out = allocate(ctx->memory, sizeof(Tensor));
  initTensor(ctx, out, outShape, F32);
  clearTensorValues(out);

  // Set one-hot values
  // For each element in indices, set the corresponding position to 1.0
  dim_t lastDimStride = numClasses;
  for (tensor_size_t i = 0; i < source->size; i++) {
    Value idxVal;
    Result readResult = readTensorValueAtFlatIndex(source, i, &idxVal);
    if (readResult != OK) {
      releaseTensorArg(ctx, &sourceArg);
      FreeTensor(ctx, out);
      return NULL;
    }

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
    if (writeResult != OK) {
      releaseTensorArg(ctx, &sourceArg);
      FreeTensor(ctx, out);
      return NULL;
    }
  }

  releaseTensorArg(ctx, &sourceArg);
  return out;
}
