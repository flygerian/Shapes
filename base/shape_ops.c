#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "result/result.h"
#include "shapes.h"
#include "types.h"
#include "tensor_internal.h"
#include <stdlib.h>

Tensor shapes_Slice(Context *ctx, Tensor *source, ...) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);

  Range *ranges = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
  PANIC_IF(ranges == NULL, ERR_OUT_OF_MEMORY);

  va_list args;
  va_start(args, source);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    ranges[x] = va_arg(args, Range);

    if (ranges[x].end < ranges[x].start) {
      va_end(args);
      PANIC_IF(true, ERR_INVALID_RANGE);
    }

    if (ranges[x].start < 0 || ranges[x].start > source->shape.dims[x] || ranges[x].end < 0 ||
        ranges[x].end > source->shape.dims[x]) {
      va_end(args);
      PANIC_IF(true, ERR_DIM_MISMATCH);
    }
  }
  va_end(args);

  Dim newShape = {.dims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims),
                  .numOfDims = source->shape.numOfDims,
                  .multipliers =
                      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims)};
  Range *boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
  PANIC_IF(newShape.dims == NULL || newShape.multipliers == NULL || boundary == NULL,
           ALLOCATION_FAILED);

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
  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newShape.dims, newShape.numOfDims);

  Tensor dest = tensorView(source->context, ctx->memory, source->values, snm.size, source->dtype,
                     newShape, boundary, false);
  return dest;
}

Tensor shapes_Reshape(Context *ctx, Tensor *source, Dim newShape) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(newShape.dims == NULL, ERR_NULL_SHAPE_PROVIDED);

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newShape.dims, newShape.numOfDims);
  PANIC_IF(snm.size != source->size, ERR_RESHAPE_DIM_MISMATCH);

  void *values;
  bool isView = true;
  Range *boundary = NULL;

  if (!source->isContigous) {
    Tensor *contiguous = copyToContiguous(ctx, source);
    PANIC_IF(contiguous == NULL, ERR_OUT_OF_MEMORY);
    values = contiguous->values;
    isView = false;
  } else {
    values = source->values;
    if (source->boundary != NULL) {
      boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
      PANIC_IF(boundary == NULL, ALLOCATION_FAILED);
      memcpy(boundary, source->boundary, sizeof(Range) * source->shape.numOfDims);
    }
  }

  Tensor dest = isView ? tensorView(source->context, ctx->memory, values, source->size, source->dtype,
                              (Dim){0}, boundary, true)
                 : (Tensor){.context = ctx,
                            .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                            .isView = false,
                            .values = values,
                            .dtype = source->dtype,
                            .boundary = boundary,
                            .size = source->size,
                            .isContigous = true};
  dim_t *copiedDims = allocate(ctx->memory, sizeof(dim_t) * newShape.numOfDims);
  PANIC_IF(copiedDims == NULL, ALLOCATION_FAILED);
  memcpy(copiedDims, newShape.dims, sizeof(dim_t) * newShape.numOfDims);

  dest.shape =
      (Dim){.dims = copiedDims, .numOfDims = newShape.numOfDims, .multipliers = snm.multipliers};

  dest.opType = OP_RESHAPE;
  dest.inputs = MakeDynamicArray(ctx->memory, sizeof(Tensor));
  shapes_Array_AppendTensor(dest.inputs, source);

  Tensor *gradPtr = allocate(ctx->memory, sizeof(Tensor));
  PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
  *gradPtr = shapes_Make_ZerosTensor(ctx, dest.shape);
  dest.grad = gradPtr;

  return dest;
}

void shapes_ReshapeBackward(Context *ctx, Tensor *node) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(node == NULL || node->inputs == NULL || node->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  Tensor *input = shapes_Array_TensorIdx(node->inputs, 0);

  Tensor gradReshaped = shapes_Reshape(ctx, node->grad, input->shape);
  shapes_AddInPlace(ctx, input->grad, &gradReshaped);
}

Tensor shapes_Transpose(Context *ctx, Tensor *source, ...) {
  dim_t transposeDims[2];
  u8 expectedDims = 2;

  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->size < 2, ERR_NO_OP);

  va_list args;
  va_start(args, source);

  for (u8 x = 0; x < expectedDims; x++) {
    transposeDims[x] = va_arg(args, dim_t);
  }
  va_end(args);

  PANIC_IF(transposeDims[0] >= source->shape.numOfDims ||
               transposeDims[1] >= source->shape.numOfDims,
           ERR_DIM_MISMATCH);

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  dim_t temp = newDims[transposeDims[0]];
  newDims[transposeDims[0]] = newDims[transposeDims[1]];
  newDims[transposeDims[1]] = temp;

  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  PANIC_IF(newMultipliers == NULL, ALLOCATION_FAILED);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  multiplier_t tempMultiplier = newMultipliers[transposeDims[0]];
  newMultipliers[transposeDims[0]] = newMultipliers[transposeDims[1]];
  newMultipliers[transposeDims[1]] = tempMultiplier;

  // Deep-copy and permute boundary to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    memcpy(newBoundary, source->boundary, sizeof(Range) * source->shape.numOfDims);
    Range tmp = newBoundary[transposeDims[0]];
    newBoundary[transposeDims[0]] = newBoundary[transposeDims[1]];
    newBoundary[transposeDims[1]] = tmp;
  }

  Tensor dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);
  return dest;
}

Tensor shapes_Permute(Context *ctx, Tensor *source, Dim order) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(order.dims == NULL, ERR_NULL_SHAPE_PROVIDED);
  PANIC_IF(order.numOfDims != source->shape.numOfDims, ERR_DIM_MISMATCH);

  bool *seen = allocate(ctx->memory, sizeof(bool) * source->shape.numOfDims);
  PANIC_IF(seen == NULL, ALLOCATION_FAILED);
  memset(seen, 0, sizeof(bool) * source->shape.numOfDims);

  for (u8 i = 0; i < order.numOfDims; i++) {
    dim_t sourceDim = order.dims[i];
    if (sourceDim >= source->shape.numOfDims || seen[sourceDim]) {
      PANIC_IF(true, ERR_DIM_MISMATCH);
    }
    seen[sourceDim] = true;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  multiplier_t *newMultipliers =
      allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  PANIC_IF(newDims == NULL || newMultipliers == NULL, ALLOCATION_FAILED);

  for (u8 i = 0; i < order.numOfDims; i++) {
    dim_t sourceDim = order.dims[i];
    newDims[i] = source->shape.dims[sourceDim];
    newMultipliers[i] = source->shape.multipliers[sourceDim];
  }

  Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    for (u8 i = 0; i < order.numOfDims; i++) {
      dim_t sourceDim = order.dims[i];
      newBoundary[i] = source->boundary[sourceDim];
    }
  }

  Tensor dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);
  return dest;
}

Tensor shapes_Squeeze(Context *ctx, Tensor *t) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);

  Tensor dest;

  // Scalars have no singleton dimensions to remove, so preserve the 0-D shape.
  if (t->shape.numOfDims == 0) {
    dest =
        tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                   (Dim){.dims = NULL, .numOfDims = 0, .multipliers = NULL}, NULL, t->isContigous);
    dest.grad = t->grad;
    dest.inputs = t->inputs;
    dest.opType = t->opType;
    return dest;
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
  PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
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

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  // Deep-copy boundary for surviving dims only to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
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

  dest =
      tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                 (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers},
                 newBoundary, t->isContigous);
  dest.grad = t->grad;
  dest.inputs = t->inputs;
  dest.opType = t->opType;

  return dest;
}

Tensor shapes_SqueezeDim(Context *ctx, Tensor *t, dim_t dim) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim >= t->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(t->shape.dims[dim] != 1, ERR_DIM_MISMATCH);

  if (t->shape.numOfDims == 1) {
    dim_t *newDims = allocate(ctx->memory, sizeof(dim_t));
    PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
    newDims[0] = 1;
    multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t));
    PANIC_IF(newMultipliers == NULL, ALLOCATION_FAILED);
    newMultipliers[0] = 1;

    Range *newBoundary = NULL;
    if (t->boundary != NULL) {
      newBoundary = allocate(ctx->memory, sizeof(Range));
      PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
      newBoundary[0] = t->boundary[0];
    }

    Tensor destSingle = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                       (Dim){.dims = newDims, .numOfDims = 1, .multipliers = newMultipliers},
                       newBoundary, t->isContigous);
    return destSingle;
  }

  u8 newNumDims = t->shape.numOfDims - 1;
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  PANIC_IF(newDims == NULL, ALLOCATION_FAILED);

  u8 destIdx = 0;
  for (u8 i = 0; i < t->shape.numOfDims; i++) {
    if (i == dim) {
      continue;
    }
    newDims[destIdx++] = t->shape.dims[i];
  }

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newDims, newNumDims);

  // Deep-copy boundary excluding the squeezed dim to avoid shared-pointer double-free.
  Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = allocate(ctx->memory, sizeof(Range) * newNumDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    u8 bIdx = 0;
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (i != dim) {
        newBoundary[bIdx++] = t->boundary[i];
      }
    }
  }

  Tensor destSqueezed =
      tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                 (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers},
                 newBoundary, t->isContigous);

  return destSqueezed;
}

Tensor shapes_UnSqueeze(Context *ctx, Tensor *t, dim_t dim) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim > t->shape.numOfDims, ERR_DIM_MISMATCH);

  u8 newNumDims = t->shape.numOfDims + 1;
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  PANIC_IF(newDims == NULL || newMultipliers == NULL, ALLOCATION_FAILED);

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
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
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

  Tensor dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                     (Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     newBoundary, t->isContigous);
  return dest;
}

Tensor shapes_Concat(Context *ctx, Tensor *target, dim_t targetDim, Tensor **tensors,
               u32 numTensorsToAdd) {
  Tensor **tensorsContig = NULL;

  PANIC_IF(isInvalidTensor(target), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(target->shape.numOfDims < 1, ERR_CONCAT_SOURCE_TENSOR_CANNOT_HAVE_ZERO_DIMS);
  PANIC_IF(targetDim > target->shape.numOfDims - 1, ERR_CONCAT_TARGET_DIM_IS_OUT_OF_BOUNDS);

  Tensor *workingTarget = materializeTensorOnContext(ctx, target);

  tensor_size_t numElementsBeforeTargetDim;
  Result result =
      calculateNumElementsBeforeDim(workingTarget, targetDim, &numElementsBeforeTargetDim);
  PANIC_IF(result != OK, result);

  tensor_size_t numElementsAfterTargetDim;
  result = calculateNumElementsAfterDim(workingTarget, targetDim, &numElementsAfterTargetDim);
  PANIC_IF(result != OK, result);

  if (numTensorsToAdd > 0) {
    tensorsContig = (Tensor **)allocate(ctx->memory, sizeof(Tensor *) * numTensorsToAdd);
    PANIC_IF(tensorsContig == NULL, ALLOCATION_FAILED);
  }

  for (tensor_size_t it = 0; it < numTensorsToAdd; it++) {
    Tensor *currentTensor = tensors[it];

    PANIC_IF(isInvalidTensor(currentTensor), ERR_CONCAT_TENSOR_IS_NULL);
    PANIC_IF(currentTensor->shape.numOfDims != workingTarget->shape.numOfDims,
             ERR_CONCAT_TENSORS_UNEQUAL_DIMS);
    PANIC_IF(currentTensor->dtype != workingTarget->dtype, ERR_CONCAT_TENSOR_NOT_SAME_DTYPE);

    for (dim_t id = 0; id < workingTarget->shape.numOfDims; id++) {
      PANIC_IF(id != targetDim && workingTarget->shape.dims[id] != currentTensor->shape.dims[id],
               ERR_CONCAT_TENSORS_UNEQUAL_DIMS);
    }

    tensorsContig[it] = materializeTensorOnContext(ctx, currentTensor);
  }

  dim_t *outputDims = allocate(ctx->memory, sizeof(dim_t) * workingTarget->shape.numOfDims);
  PANIC_IF(result != OK, result);
  dim_t dimsToAdd = 0;

  for (tensor_size_t ist = 0; ist < numTensorsToAdd; ist++) {
    dimsToAdd += tensorsContig[ist]->shape.dims[targetDim];
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

  sizeAndMultipliers snm =
      calculateSizeAndMultipliers(ctx, outputDims, workingTarget->shape.numOfDims);
  outputShape.multipliers = snm.multipliers;

  Tensor dest = t_Zeros(ctx, outputShape, workingTarget->dtype);
  dim_t currDimSize = workingTarget->shape.dims[targetDim];
  dim_t newDimSize = outputShape.dims[targetDim];
  size_t bytesPerElem = getBytesForDtype(workingTarget->dtype);

  for (tensor_size_t inb = 0; inb < numElementsBeforeTargetDim; inb++) {
    tensor_size_t destSliceOffset = inb * newDimSize * numElementsAfterTargetDim;
    tensor_size_t srcSliceOffset = inb * currDimSize * numElementsAfterTargetDim;

    memcpy((char *)dest.values + destSliceOffset * bytesPerElem,
           (char *)workingTarget->values + srcSliceOffset * bytesPerElem,
           currDimSize * numElementsAfterTargetDim * bytesPerElem);

    dim_t dimOffset = currDimSize;
    for (dim_t ist = 0; ist < numTensorsToAdd; ist++) {
      Tensor *curr = tensorsContig[ist];
      dim_t currTargetDimSize = curr->shape.dims[targetDim];
      tensor_size_t currDestOffset = destSliceOffset + (dimOffset * numElementsAfterTargetDim);
      tensor_size_t currSrcOffset = inb * currTargetDimSize * numElementsAfterTargetDim;
      memcpy((char *)dest.values + currDestOffset * bytesPerElem,
             (char *)curr->values + currSrcOffset * bytesPerElem,
             currTargetDimSize * numElementsAfterTargetDim * bytesPerElem);
      dimOffset += currTargetDimSize;
    }
  }

  return dest;
}

Tensor shapes_Stack(Context *ctx, Array_Tensor tensors) {
  PANIC_IF_NULL(ctx);
  PANIC_IF_NULL(tensors);
  PANIC_IF(tensors->size < 2, ERR_STACKING_LESS_THAN_TWO_TENSORS);

  Tensor *firstTensor = shapes_Array_TensorIdx(tensors, 0);
  size_t numToAdd = tensors->size - 1;
  Tensor *unsqueezedStorage = allocate(ctx->memory, sizeof(Tensor) * numToAdd);
  PANIC_IF(unsqueezedStorage == NULL, ALLOCATION_FAILED);
  Tensor **unsqueezedPtrs = allocate(ctx->memory, sizeof(Tensor *) * numToAdd);
  PANIC_IF(unsqueezedPtrs == NULL, ALLOCATION_FAILED);

  for (RANGE_FROM(1, tensors->size, i)) {
    Tensor *currentTensor = shapes_Array_TensorIdx(tensors, i);

    PANIC_IF_NULL(currentTensor);
    PANIC_IF(!isSameShape(firstTensor, currentTensor), ERR_DIM_MISMATCH);

    unsqueezedStorage[i - 1] = shapes_UnSqueeze(ctx, currentTensor, 0);
    unsqueezedPtrs[i - 1] = &unsqueezedStorage[i - 1];
  }

  Tensor firstTensorUnsqueezed = shapes_UnSqueeze(ctx, firstTensor, 0);
  return shapes_Concat(ctx, &firstTensorUnsqueezed, 0, unsqueezedPtrs, numToAdd);
}
