#include <stdarg.h>
#include <string.h>
#include "tensor_internal.h"
#include "../memory.h"

Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...) {
  Range *ranges = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  va_list args;
  va_start(args, dest);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    ranges[x] = va_arg(args, Range);

    if (ranges[x].end < ranges[x].start) {
      va_end(args);
      return ERR_INVALID_RANGE;
    }

    if (ranges[x].start < 0 || ranges[x].start > source->shape.dims[x] || ranges[x].end < 0 ||
        ranges[x].end > source->shape.dims[x]) {
      va_end(args);
      return ERR_DIM_MISMATCH;
    }
  }
  va_end(args);

  Dim newShape = {.dims = allocate(ctx->memory, sizeof(Dim) * source->shape.numOfDims),
                  .numOfDims = source->shape.numOfDims,
                  .multipliers = source->shape.multipliers};
  Range *boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    Range r = ranges[x];
    u32 dimsize = (r.end - r.start);
    newShape.dims[x] = dimsize;

    if (source->isView) {
      boundary[x] = (Range){.start = source->boundary->start + ranges[x].start,
                            .end = source->boundary->start + ranges[x].end};
    } else {
      boundary[x] = ranges[x];
    }
  }

  tensor_size_t size = calculateNumValuesAndMultipliers(newShape, NULL);
  *dest = ((Tensor){
      .isView = true,
      .values = source->values,
      .shape = newShape,
      .dtype = source->dtype,
      .boundary = boundary,
      .size = size,
      .isContigous = false,
  });

  return OK;
}

Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (newShape.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
  }

  u8 *multipliers = allocate(ctx->memory, sizeof(u8) * newShape.numOfDims);
  tensor_size_t proposedSize = calculateNumValuesAndMultipliers(newShape, multipliers);

  if (proposedSize != source->size) {
    return ERR_RESHAPE_DIM_MISMATCH;
  }

  void *values;
  bool isView = source->isView;
  Range *boundary = source->boundary;

  if (!source->isContigous) {
    Tensor *contiguous = copyToContiguous(ctx, source);
    values = contiguous->values;
    isView = false;
    boundary = NULL;
    freeAlloc(ctx->memory, contiguous->shape.dims);
    freeAlloc(ctx->memory, contiguous->shape.multipliers);
    freeAlloc(ctx->memory, contiguous);
  } else {
    values = source->values;
  }

  *dest = ((Tensor){.isView = isView,
                    .values = values,
                    .dtype = source->dtype,
                    .boundary = boundary,
                    .size = source->size,
                    .isContigous = true});
  dest->shape =
      (Dim){.dims = newShape.dims, .numOfDims = newShape.numOfDims, .multipliers = multipliers};
  return OK;
}

Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...) {
  dim_t transposeDims[2];
  u8 expectedDims = 2;

  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->size < 2) {
    return ERR_NO_OP;
  }

  va_list args;
  va_start(args, dest);

  for (u8 x = 0; x < expectedDims; x++) {
    transposeDims[x] = va_arg(args, dim_t);
  }
  va_end(args);

  if (transposeDims[0] >= source->shape.numOfDims || transposeDims[1] >= source->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  dim_t temp = newDims[transposeDims[0]];
  newDims[transposeDims[0]] = newDims[transposeDims[1]];
  newDims[transposeDims[1]] = temp;

  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  multiplier_t tempMultiplier = newMultipliers[transposeDims[0]];
  newMultipliers[transposeDims[0]] = newMultipliers[transposeDims[1]];
  newMultipliers[transposeDims[1]] = tempMultiplier;

  *dest = (Tensor){.dtype = source->dtype,
                   .values = source->values,
                   .size = source->size,
                   .isView = true,
                   .isContigous = false,
                   .shape = {.dims = newDims,
                             .numOfDims = source->shape.numOfDims,
                             .multipliers = newMultipliers},
                   .boundary = source->boundary};

  return OK;
}

Result Squeeze(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  u8 newNumDims = 0;
  for (u8 i = 0; i < t->shape.numOfDims; i++) {
    if (t->shape.dims[i] != 1) {
      newNumDims++;
    }
  }

  if (newNumDims == 0) {
    newNumDims = 1;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  u8 destIdx = 0;

  if (newNumDims == 1 && t->shape.dims[0] == 1) {
    bool allOnes = true;
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (t->shape.dims[i] != 1) {
        allOnes = false;
        break;
      }
    }
    if (allOnes) {
      newDims[0] = 1;
      destIdx = 1;
    }
  }

  if (destIdx == 0) {
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (t->shape.dims[i] != 1) {
        newDims[destIdx++] = t->shape.dims[i];
      }
    }
  }

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  calculateNumValuesAndMultipliers((Dim){.dims = newDims, .numOfDims = newNumDims}, newMultipliers);

  *dest =
      (Tensor){.dtype = t->dtype,
               .values = t->values,
               .size = t->size,
               .isContigous = t->isContigous,
               .isView = true,
               .boundary = t->boundary,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers}};

  return OK;
}

Result SqueezeDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (t->shape.dims[dim] != 1) {
    return ERR_DIM_MISMATCH;
  }

  if (t->shape.numOfDims == 1) {
    *dest = *t;
    dest->isView = true;
    return OK;
  }

  u8 newNumDims = t->shape.numOfDims - 1;
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);

  u8 destIdx = 0;
  for (u8 i = 0; i < t->shape.numOfDims; i++) {
    if (i == dim) {
      continue;
    }
    newDims[destIdx++] = t->shape.dims[i];
  }

  calculateNumValuesAndMultipliers((Dim){.dims = newDims, .numOfDims = newNumDims}, newMultipliers);

  *dest =
      (Tensor){.dtype = t->dtype,
               .values = t->values,
               .size = t->size,
               .isContigous = t->isContigous,
               .isView = true,
               .boundary = t->boundary,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers}};

  return OK;
}

Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim > t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  u8 newNumDims = t->shape.numOfDims + 1;
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);

  for (u8 i = 0; i < newNumDims; i++) {
    if (i < dim) {
      newDims[i] = t->shape.dims[i];
      newMultipliers[i] = t->shape.multipliers[i];
      continue;
    }

    if (i == dim) {
      newDims[i] = 1;
      if (dim < t->shape.numOfDims) {
        newMultipliers[i] = t->shape.multipliers[dim];
      } else {
        newMultipliers[i] = 1;
      }
      continue;
    }

    newDims[i] = t->shape.dims[i - 1];
    newMultipliers[i] = t->shape.multipliers[i - 1];
  }

  *dest =
      (Tensor){.dtype = t->dtype,
               .values = t->values,
               .size = t->size,
               .isContigous = t->isContigous,
               .isView = true,
               .boundary = t->boundary,
               .shape = {.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers}};

  return OK;
}
