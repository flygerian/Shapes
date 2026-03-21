#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "result/result.h"
#include "shapes.h"
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
      freeAlloc(ctx->memory, ranges);
      return ERR_INVALID_RANGE;
    }

    if (ranges[x].start < 0 || ranges[x].start > source->shape.dims[x] || ranges[x].end < 0 ||
        ranges[x].end > source->shape.dims[x]) {
      va_end(args);
      freeAlloc(ctx->memory, ranges);
      return ERR_DIM_MISMATCH;
    }
  }
  va_end(args);

  Dim newShape = {.dims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims),
                  .numOfDims = source->shape.numOfDims,
                  .multipliers =
                      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims)};
  Range *boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    Range r = ranges[x];
    u32 dimsize = (r.end - r.start);
    newShape.dims[x] = dimsize;

    if (source->isView && source->boundary) {
      boundary[x] = (Range){.start = source->boundary[x].start + ranges[x].start,
                            .end = source->boundary[x].start + ranges[x].end};
    } else {
      boundary[x] = ranges[x];
    }
  }

  // Preserve source strides for views so boundary-adjusted indexing maps into
  // the same underlying storage layout (including sliced/transposed sources).
  memcpy(newShape.multipliers, source->shape.multipliers,
         sizeof(multiplier_t) * source->shape.numOfDims);
  tensor_size_t size = calculateNumValuesAndMultipliers(
      (Dim){.dims = newShape.dims, .numOfDims = newShape.numOfDims}, NULL);
  freeAlloc(ctx->memory, ranges);
  *dest =
      tensorView(source->context, ctx->memory, source->values, size, source->dtype, newShape,
                 boundary, false);

  return OK;
}

Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (newShape.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
  }

  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * newShape.numOfDims);
  tensor_size_t proposedSize = calculateNumValuesAndMultipliers(newShape, multipliers);

  if (proposedSize != source->size) {
    return ERR_RESHAPE_DIM_MISMATCH;
  }

  void *values;
  bool isView = true;
  Range *boundary = NULL;

  if (!source->isContigous) {
    Tensor *contiguous = copyToContiguous(ctx, source);
    values = contiguous->values;
    isView = false;
    freeAlloc(contiguous->metadataMemory, contiguous->shape.dims);
    freeAlloc(contiguous->metadataMemory, contiguous->shape.multipliers);
    freeAlloc(contiguous->metadataMemory, contiguous);
  } else {
    values = source->values;
    if (source->boundary != NULL) {
      boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
      memcpy(boundary, source->boundary, sizeof(Range) * source->shape.numOfDims);
    }
  }

  *dest = isView ? tensorView(source->context, ctx->memory, values, source->size, source->dtype,
                              (Dim){0}, boundary, true)
                 : (Tensor){.context = ctx,
                            .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                            .isView = false,
                            .values = values,
                            .dtype = source->dtype,
                            .boundary = boundary,
                            .size = source->size,
                            .isContigous = true};
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

  // Deep-copy and permute boundary to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
    memcpy(newBoundary, source->boundary, sizeof(Range) * source->shape.numOfDims);
    Range tmp = newBoundary[transposeDims[0]];
    newBoundary[transposeDims[0]] = newBoundary[transposeDims[1]];
    newBoundary[transposeDims[1]] = tmp;
  }

  *dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);

  return OK;
}

Result Permute(Context *ctx, Tensor *source, Tensor *dest, Dim order) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (order.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
  }

  if (order.numOfDims != source->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  bool *seen = allocate(ctx->memory, sizeof(bool) * source->shape.numOfDims);
  memset(seen, 0, sizeof(bool) * source->shape.numOfDims);

  for (u8 i = 0; i < order.numOfDims; i++) {
    dim_t sourceDim = order.dims[i];
    if (sourceDim >= source->shape.numOfDims || seen[sourceDim]) {
      freeAlloc(ctx->memory, seen);
      return ERR_DIM_MISMATCH;
    }
    seen[sourceDim] = true;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);

  for (u8 i = 0; i < order.numOfDims; i++) {
    dim_t sourceDim = order.dims[i];
    newDims[i] = source->shape.dims[sourceDim];
    newMultipliers[i] = source->shape.multipliers[sourceDim];
  }

  Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
    for (u8 i = 0; i < order.numOfDims; i++) {
      dim_t sourceDim = order.dims[i];
      newBoundary[i] = source->boundary[sourceDim];
    }
  }

  freeAlloc(ctx->memory, seen);

  *dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);

  return OK;
}

Result Squeeze(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  // Scalars have no singleton dimensions to remove, so preserve the 0-D shape.
  if (t->shape.numOfDims == 0) {
    *dest =
        tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                   (Dim){.dims = NULL, .numOfDims = 0, .multipliers = NULL}, NULL, t->isContigous);
    return OK;
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

  // Deep-copy boundary for surviving dims only to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    u8 bIdx = 0;
    if (newNumDims == 1 && t->shape.dims[0] == 1) {
      // All-ones edge case: kept first dim
      newBoundary[0] = t->boundary[0];
    } else {
      for (u8 i = 0; i < t->shape.numOfDims; i++) {
        if (t->shape.dims[i] != 1) {
          newBoundary[bIdx++] = t->boundary[i];
        }
      }
    }
  }

  *dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                     (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     newBoundary, t->isContigous);

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
    dim_t *newDims = allocate(ctx->memory, sizeof(dim_t));
    newDims[0] = 1;
    multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t));
    newMultipliers[0] = 1;

    Range *newBoundary = NULL;
    if (t->boundary != NULL) {
      newBoundary = allocate(ctx->memory, sizeof(Range));
      newBoundary[0] = t->boundary[0];
    }

    *dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                       (Dim){.dims = newDims, .numOfDims = 1, .multipliers = newMultipliers},
                       newBoundary, t->isContigous);
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

  // Deep-copy boundary excluding the squeezed dim to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    u8 bIdx = 0;
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (i != dim) {
        newBoundary[bIdx++] = t->boundary[i];
      }
    }
  }

  *dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                     (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     newBoundary, t->isContigous);

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

  // Deep-copy boundary with the new dimension inserted to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    for (u8 i = 0; i < newNumDims; i++) {
      if (i < dim) {
        newBoundary[i] = t->boundary[i];
      } else if (i == dim) {
        newBoundary[i] = (Range){.start = 0, .end = 1};
      } else {
        newBoundary[i] = t->boundary[i - 1];
      }
    }
  }

  *dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                     (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     newBoundary, t->isContigous);

  return OK;
}

Result Concat(Context *ctx, Tensor *target, dim_t targetDim, Tensor **tensors, u32 numTensorsToAdd,
              Tensor *dest) {
  TensorArg targetArg = {0};
  TensorArg *tensorArgs = NULL;
  Result result = OK;

  if (isInvalidTensor(target)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (target->shape.numOfDims < 1) {
    return ERR_CONCAT_SOURCE_TENSOR_CANNOT_HAVE_ZERO_DIMS;
  }

  if (targetDim > target->shape.numOfDims - 1) {
    return ERR_CONCAT_TARGET_DIM_IS_OUT_OF_BOUNDS;
  }

  result = materializeTensorOnContext(ctx, target, true, &targetArg);
  if (result != OK) {
    return result;
  }

  Tensor *workingTarget = targetArg.tensor;
  tensor_size_t numElementsBeforeTargetDim;
  result = calculateNumElementsBeforeDim(workingTarget, targetDim, &numElementsBeforeTargetDim);
  if (result != OK) {
    goto cleanup_concat;
  }

  tensor_size_t numElementsAfterTargetDim;
  result = calculateNumElementsAfterDim(workingTarget, targetDim, &numElementsAfterTargetDim);
  if (result != OK) {
    goto cleanup_concat;
  }

  if (numTensorsToAdd > 0) {
    tensorArgs = allocate(ctx->memory, sizeof(TensorArg) * numTensorsToAdd);
    memset(tensorArgs, 0, sizeof(TensorArg) * numTensorsToAdd);
  }

  for (tensor_size_t it = 0; it < numTensorsToAdd; it++) {
    Tensor *t = tensors[it];

    if (isInvalidTensor(t)) {
      result = ERR_CONCAT_TENSOR_IS_NULL;
      goto cleanup_concat;
    }

    if (t->shape.numOfDims != workingTarget->shape.numOfDims) {
      result = ERR_CONCAT_TENSORS_UNEQUAL_DIMS;
      goto cleanup_concat;
    }

    for (dim_t id = 0; id < workingTarget->shape.numOfDims; id++) {
      if (id != targetDim && workingTarget->shape.dims[id] != t->shape.dims[id]) {
        result = ERR_CONCAT_TENSORS_UNEQUAL_DIMS;
        goto cleanup_concat;
      }
    }

    if (t->dtype != workingTarget->dtype) {
      result = ERR_CONCAT_TENSOR_NOT_SAME_DTYPE;
      goto cleanup_concat;
    }

    if (!t->isContigous) {
      result = ERR_CONCAT_TENSOR_NOT_CONTIGOUS;
      goto cleanup_concat;
    }

    result = materializeTensorOnContext(ctx, t, true, &tensorArgs[it]);
    if (result != OK) {
      goto cleanup_concat;
    }
  }

  dim_t *outputDims = allocate(ctx->memory, sizeof(dim_t) * workingTarget->shape.numOfDims);
  dim_t dimsToAdd = 0;

  for (tensor_size_t ist = 0; ist < numTensorsToAdd; ist++) {
    dimsToAdd += tensorArgs[ist].tensor->shape.dims[targetDim];
  }

  for (dim_t io = 0; io < workingTarget->shape.numOfDims; io++) {
    if (io == targetDim) {
      dim_t targetDimCurrSize = workingTarget->shape.dims[io];
      outputDims[io] = targetDimCurrSize + dimsToAdd;
      continue;
    }

    outputDims[io] = workingTarget->shape.dims[io];
  }

  Dim outputShape = {.dims = outputDims, .numOfDims = workingTarget->shape.numOfDims};

  multiplier_t *multipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * workingTarget->shape.numOfDims);
  calculateNumValuesAndMultipliers(outputShape, multipliers);
  outputShape.multipliers = multipliers;

  result = initTensor(ctx, dest, outputShape, workingTarget->dtype);
  if (result != OK) {
    goto cleanup_concat;
  }

  dim_t currDimSize = workingTarget->shape.dims[targetDim];
  dim_t newDimSize = outputShape.dims[targetDim];
  size_t bytesPerElem = getBytesForDtype(workingTarget->dtype);

  for (tensor_size_t inb = 0; inb < numElementsBeforeTargetDim; inb++) {
    tensor_size_t destSliceOffset = inb * newDimSize * numElementsAfterTargetDim;
    tensor_size_t srcSliceOffset = inb * currDimSize * numElementsAfterTargetDim;

    result = copyBetweenContexts(workingTarget->context, dest->context,
                                 (char *)workingTarget->values + srcSliceOffset * bytesPerElem,
                                 (char *)dest->values + destSliceOffset * bytesPerElem,
                                 currDimSize * numElementsAfterTargetDim * bytesPerElem);
    if (result != OK) {
      goto cleanup_concat;
    }

    dim_t dimOffset = currDimSize;
    for (dim_t ist = 0; ist < numTensorsToAdd; ist++) {
      Tensor *curr = tensorArgs[ist].tensor;
      dim_t currTargetDimSize = curr->shape.dims[targetDim];
      tensor_size_t currDestOffset = destSliceOffset + (dimOffset * numElementsAfterTargetDim);
      tensor_size_t currSrcOffset = inb * currTargetDimSize * numElementsAfterTargetDim;
      result = copyBetweenContexts(curr->context, dest->context,
                                   (char *)curr->values + currSrcOffset * bytesPerElem,
                                   (char *)dest->values + currDestOffset * bytesPerElem,
                                   currTargetDimSize * numElementsAfterTargetDim * bytesPerElem);
      if (result != OK) {
        goto cleanup_concat;
      }

      dimOffset += currTargetDimSize;
    }
  }

  result = OK;

cleanup_concat:
  if (tensorArgs != NULL) {
    for (tensor_size_t it = 0; it < numTensorsToAdd; it++) {
      releaseTensorArg(ctx, &tensorArgs[it]);
    }
  }
  releaseTensorArg(ctx, &targetArg);
  return result;
}
