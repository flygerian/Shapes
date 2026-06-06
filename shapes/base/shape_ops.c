#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "result.h"
#include "shapes.h"
#include "types.h"
#include "olib.h"
#include <stdlib.h>
#include "shapes_internal.h"

shapes_Tensor shapes_Slice(shapes_Context *ctx, shapes_Tensor *source, ...) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);

  shapes_Range *ranges = olib_Allocate(ctx->memory, sizeof(shapes_Range) * source->shape.numOfDims);
  PANIC_IF(ranges == NULL, ERR_OUT_OF_MEMORY);

  va_list args;
  va_start(args, source);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    ranges[x] = va_arg(args, shapes_Range);

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

  shapes_Dim newShape = {.dims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * source->shape.numOfDims),
                  .numOfDims = source->shape.numOfDims,
                  .multipliers =
                      olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t) * source->shape.numOfDims)};
  shapes_Range *boundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * source->shape.numOfDims);
  PANIC_IF(newShape.dims == NULL || newShape.multipliers == NULL || boundary == NULL,
           ALLOCATION_FAILED);

  for (u8 x = 0; x < source->shape.numOfDims; x++) {
    shapes_Range r = ranges[x];
    u32 dimsize = (r.end - r.start);
    newShape.dims[x] = dimsize;

    if (source->isView && source->boundary) {
      boundary[x] = (shapes_Range){.start = source->boundary[x].start + ranges[x].start,
                            .end = source->boundary[x].start + ranges[x].end};
    } else {
      boundary[x] = ranges[x];
    }
  }

  // Preserve source strides for views so boundary-adjusted indexing maps into
  // the same underlying storage layout (including sliced/transposed sources).
  memcpy(newShape.multipliers, source->shape.multipliers,
         sizeof(shapes_multiplier_t) * source->shape.numOfDims);
  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newShape.dims, newShape.numOfDims);

  shapes_Tensor dest = tensorView(source->context, ctx->memory, source->values, snm.size, source->dtype,
                     newShape, boundary, false);
  return dest;
}

shapes_Tensor shapes_Reshape(shapes_Context *ctx, shapes_Tensor *source, shapes_Dim newShape) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(newShape.dims == NULL, ERR_NULL_SHAPE_PROVIDED);

  sizeAndMultipliers snm = calculateSizeAndMultipliers(ctx, newShape.dims, newShape.numOfDims);
  PANIC_IF(snm.size != source->size, ERR_RESHAPE_DIM_MISMATCH);

  void *values;
  bool isView = true;
  shapes_Range *boundary = NULL;

  if (!source->isContigous) {
    shapes_Tensor *contiguous = copyToContiguous(ctx, source);
    PANIC_IF(contiguous == NULL, ERR_OUT_OF_MEMORY);
    values = contiguous->values;
    isView = false;
  } else {
    values = source->values;
    if (source->boundary != NULL) {
      boundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * source->shape.numOfDims);
      PANIC_IF(boundary == NULL, ALLOCATION_FAILED);
      memcpy(boundary, source->boundary, sizeof(shapes_Range) * source->shape.numOfDims);
    }
  }

  shapes_Tensor dest = isView ? tensorView(source->context, ctx->memory, values, source->size, source->dtype,
                              (shapes_Dim){0}, boundary, true)
                 : (shapes_Tensor){.context = ctx,
                            .metadataMemory = ctx != NULL ? ctx->memory : NULL,
                            .isView = false,
                            .values = values,
                            .dtype = source->dtype,
                            .boundary = boundary,
                            .size = source->size,
                            .isContigous = true};
  shapes_dim_t *copiedDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newShape.numOfDims);
  PANIC_IF(copiedDims == NULL, ALLOCATION_FAILED);
  memcpy(copiedDims, newShape.dims, sizeof(shapes_dim_t) * newShape.numOfDims);

  dest.shape =
      (shapes_Dim){.dims = copiedDims, .numOfDims = newShape.numOfDims, .multipliers = snm.multipliers};

  dest.opType = OP_RESHAPE;
  dest.inputs = olib_MakeDynamicArray(ctx->memory, sizeof(shapes_Tensor));
  shapes_ArrayAppendTensor(dest.inputs, source);

  shapes_Tensor *gradPtr = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
  PANIC_IF(gradPtr == NULL, ALLOCATION_FAILED);
  *gradPtr = shapes_MakeZerosTensor(ctx, dest.shape);
  dest.grad = gradPtr;

  return dest;
}

void shapes_ReshapeBackward(shapes_Context *ctx, shapes_Tensor *node) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(node == NULL || node->inputs == NULL || node->grad == NULL, ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor input = shapes_ArrayTensorIdx(node->inputs, 0);

  shapes_Tensor gradReshaped = shapes_Reshape(ctx, node->grad, input.shape);
  shapes_AddInPlace(ctx, input.grad, &gradReshaped);
}

shapes_Tensor shapes_Transpose(shapes_Context *ctx, shapes_Tensor *source, ...) {
  shapes_dim_t transposeDims[2];
  u8 expectedDims = 2;

  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(source->size < 2, ERR_NO_OP);

  va_list args;
  va_start(args, source);

  for (u8 x = 0; x < expectedDims; x++) {
    transposeDims[x] = va_arg(args, shapes_dim_t);
  }
  va_end(args);

  PANIC_IF(transposeDims[0] >= source->shape.numOfDims ||
               transposeDims[1] >= source->shape.numOfDims,
           ERR_DIM_MISMATCH);

  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * source->shape.numOfDims);
  PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
  memcpy(newDims, source->shape.dims, sizeof(shapes_dim_t) * source->shape.numOfDims);

  shapes_dim_t temp = newDims[transposeDims[0]];
  newDims[transposeDims[0]] = newDims[transposeDims[1]];
  newDims[transposeDims[1]] = temp;

  shapes_multiplier_t *newMultipliers =
      olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t) * source->shape.numOfDims);
  PANIC_IF(newMultipliers == NULL, ALLOCATION_FAILED);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(shapes_multiplier_t) * source->shape.numOfDims);

  shapes_multiplier_t tempMultiplier = newMultipliers[transposeDims[0]];
  newMultipliers[transposeDims[0]] = newMultipliers[transposeDims[1]];
  newMultipliers[transposeDims[1]] = tempMultiplier;

  // Deep-copy and permute boundary to avoid shared-pointer double-free.
  shapes_Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * source->shape.numOfDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    memcpy(newBoundary, source->boundary, sizeof(shapes_Range) * source->shape.numOfDims);
    shapes_Range tmp = newBoundary[transposeDims[0]];
    newBoundary[transposeDims[0]] = newBoundary[transposeDims[1]];
    newBoundary[transposeDims[1]] = tmp;
  }

  shapes_Tensor dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (shapes_Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);
  return dest;
}

shapes_Tensor shapes_Permute(shapes_Context *ctx, shapes_Tensor *source, shapes_Dim order) {
  PANIC_IF(isInvalidTensor(source), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(order.dims == NULL, ERR_NULL_SHAPE_PROVIDED);
  PANIC_IF(order.numOfDims != source->shape.numOfDims, ERR_DIM_MISMATCH);

  bool *seen = olib_Allocate(ctx->memory, sizeof(bool) * source->shape.numOfDims);
  PANIC_IF(seen == NULL, ALLOCATION_FAILED);
  memset(seen, 0, sizeof(bool) * source->shape.numOfDims);

  for (u8 i = 0; i < order.numOfDims; i++) {
    shapes_dim_t sourceDim = order.dims[i];
    if (sourceDim >= source->shape.numOfDims || seen[sourceDim]) {
      PANIC_IF(true, ERR_DIM_MISMATCH);
    }
    seen[sourceDim] = true;
  }

  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * source->shape.numOfDims);
  shapes_multiplier_t *newMultipliers =
      olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t) * source->shape.numOfDims);
  PANIC_IF(newDims == NULL || newMultipliers == NULL, ALLOCATION_FAILED);

  for (u8 i = 0; i < order.numOfDims; i++) {
    shapes_dim_t sourceDim = order.dims[i];
    newDims[i] = source->shape.dims[sourceDim];
    newMultipliers[i] = source->shape.multipliers[sourceDim];
  }

  shapes_Range *newBoundary = NULL;
  if (source->boundary != NULL) {
    newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * source->shape.numOfDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    for (u8 i = 0; i < order.numOfDims; i++) {
      shapes_dim_t sourceDim = order.dims[i];
      newBoundary[i] = source->boundary[sourceDim];
    }
  }

  shapes_Tensor dest = tensorView(
      source->context, ctx->memory, source->values, source->size, source->dtype,
      (shapes_Dim){.dims = newDims, .numOfDims = source->shape.numOfDims, .multipliers = newMultipliers},
      newBoundary, false);
  return dest;
}

shapes_Tensor shapes_Squeeze(shapes_Context *ctx, shapes_Tensor *t) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);

  shapes_Tensor dest;

  // Scalars have no singleton dimensions to remove, so preserve the 0-D shape.
  if (t->shape.numOfDims == 0) {
    dest =
        tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                   (shapes_Dim){.dims = NULL, .numOfDims = 0, .multipliers = NULL}, NULL, t->isContigous);
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

  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newNumDims);
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
  shapes_Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * newNumDims);
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
                 (shapes_Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers},
                 newBoundary, t->isContigous);
  dest.grad = t->grad;
  dest.inputs = t->inputs;
  dest.opType = t->opType;

  return dest;
}

shapes_Tensor shapes_SqueezeDim(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim >= t->shape.numOfDims, ERR_DIM_MISMATCH);
  PANIC_IF(t->shape.dims[dim] != 1, ERR_DIM_MISMATCH);

  if (t->shape.numOfDims == 1) {
    shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t));
    PANIC_IF(newDims == NULL, ALLOCATION_FAILED);
    newDims[0] = 1;
    shapes_multiplier_t *newMultipliers = olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t));
    PANIC_IF(newMultipliers == NULL, ALLOCATION_FAILED);
    newMultipliers[0] = 1;

    shapes_Range *newBoundary = NULL;
    if (t->boundary != NULL) {
      newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range));
      PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
      newBoundary[0] = t->boundary[0];
    }

    shapes_Tensor destSingle = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                       (shapes_Dim){.dims = newDims, .numOfDims = 1, .multipliers = newMultipliers},
                       newBoundary, t->isContigous);
    return destSingle;
  }

  u8 newNumDims = t->shape.numOfDims - 1;
  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newNumDims);
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
  shapes_Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * newNumDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    u8 bIdx = 0;
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (i != dim) {
        newBoundary[bIdx++] = t->boundary[i];
      }
    }
  }

  shapes_Tensor destSqueezed =
      tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                 (shapes_Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = snm.multipliers},
                 newBoundary, t->isContigous);

  return destSqueezed;
}

shapes_Tensor shapes_UnSqueeze(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dim > t->shape.numOfDims, ERR_DIM_MISMATCH);

  u8 newNumDims = t->shape.numOfDims + 1;
  shapes_dim_t *newDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * newNumDims);
  shapes_multiplier_t *newMultipliers = olib_Allocate(ctx->memory, sizeof(shapes_multiplier_t) * newNumDims);
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
  shapes_Range *newBoundary = NULL;
  if (t->boundary != NULL) {
    newBoundary = olib_Allocate(ctx->memory, sizeof(shapes_Range) * newNumDims);
    PANIC_IF(newBoundary == NULL, ALLOCATION_FAILED);
    for (u8 i = 0; i < newNumDims; i++) {
      if (i < dim) {
        newBoundary[i] = t->boundary[i];
      } else if (i == dim) {
        newBoundary[i] = (shapes_Range){.start = 0, .end = 1};
      } else {
        newBoundary[i] = t->boundary[i - 1];
      }
    }
  }

  shapes_Tensor dest = tensorView(t->context, ctx->memory, t->values, t->size, t->dtype,
                     (shapes_Dim){.dims = newDims, .numOfDims = newNumDims, .multipliers = newMultipliers},
                     newBoundary, t->isContigous);
  return dest;
}

shapes_Tensor shapes_Concat(shapes_Context *ctx, shapes_Tensor *target, shapes_dim_t targetDim, shapes_ArrayTensor tensors) {
  shapes_ArrayTensor tensorsContig = NULL;

  PANIC_IF(tensors->size <= 0, ERR_NO_OP);
  PANIC_IF(isInvalidTensor(target), ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(target->shape.numOfDims < 1, ERR_CONCAT_SOURCE_TENSOR_CANNOT_HAVE_ZERO_DIMS);
  PANIC_IF(targetDim > target->shape.numOfDims - 1, ERR_CONCAT_TARGET_DIM_IS_OUT_OF_BOUNDS);

  shapes_Tensor *workingTarget = materializeTensorOnContext(ctx, target);

  shapes_tensor_size_t numElementsBeforeTargetDim;
  Result result =
      calculateNumElementsBeforeDim(workingTarget, targetDim, &numElementsBeforeTargetDim);
  PANIC_IF(result != OK, result);

  shapes_tensor_size_t numElementsAfterTargetDim;
  result = calculateNumElementsAfterDim(workingTarget, targetDim, &numElementsAfterTargetDim);
  PANIC_IF(result != OK, result);

  tensorsContig = shapes_MakeTensorArray(ctx->memory, tensors->size); 
  PANIC_IF(tensorsContig == NULL, ALLOCATION_FAILED);

  for (RANGE(it, tensors->size)) {
    shapes_Tensor currentTensor = shapes_ArrayTensorIdx(tensors, it); //tensors[it];

    PANIC_IF(currentTensor.shape.numOfDims != workingTarget->shape.numOfDims,
             ERR_CONCAT_TENSORS_UNEQUAL_DIMS);
    PANIC_IF(currentTensor.dtype != workingTarget->dtype, ERR_CONCAT_TENSOR_NOT_SAME_DTYPE);

    for (shapes_dim_t id = 0; id < workingTarget->shape.numOfDims; id++) {
      PANIC_IF(id != targetDim && workingTarget->shape.dims[id] != currentTensor.shape.dims[id],
               ERR_CONCAT_TENSORS_UNEQUAL_DIMS);
    }

    shapes_Tensor *currentTensorContig = materializeTensorOnContext(ctx, &currentTensor);
    shapes_ArrayAppendTensor(tensorsContig, currentTensorContig);
  }

  shapes_dim_t *outputDims = olib_Allocate(ctx->memory, sizeof(shapes_dim_t) * workingTarget->shape.numOfDims);
  PANIC_IF(result != OK, result);
  shapes_dim_t dimsToAdd = 0;

  for (RANGE(ist, tensors->size)) {
    dimsToAdd += shapes_ArrayTensorIdx(tensorsContig, ist).shape.dims[targetDim]; // tensorsContig[ist]->shape.dims[targetDim];
  }

  for (RANGE(io, workingTarget->shape.numOfDims)) {
    if (io == targetDim) {
      shapes_dim_t targetDimCurrSize = workingTarget->shape.dims[io];
      outputDims[io] = targetDimCurrSize + dimsToAdd;
      continue;
    }

    outputDims[io] = workingTarget->shape.dims[io];
  }

  shapes_Dim outputShape = {.dims = outputDims, .numOfDims = workingTarget->shape.numOfDims};

  sizeAndMultipliers snm =
      calculateSizeAndMultipliers(ctx, outputDims, workingTarget->shape.numOfDims);
  outputShape.multipliers = snm.multipliers;

  shapes_Tensor dest = t_Zeros(ctx, outputShape, workingTarget->dtype);
  shapes_dim_t currDimSize = workingTarget->shape.dims[targetDim];
  shapes_dim_t newDimSize = outputShape.dims[targetDim];
  size_t bytesPerElem = getBytesForDtype(workingTarget->dtype);

  for (shapes_tensor_size_t inb = 0; inb < numElementsBeforeTargetDim; inb++) {
    shapes_tensor_size_t destSliceOffset = inb * newDimSize * numElementsAfterTargetDim;
    shapes_tensor_size_t srcSliceOffset = inb * currDimSize * numElementsAfterTargetDim;

    memcpy((char *)dest.values + destSliceOffset * bytesPerElem,
           (char *)workingTarget->values + srcSliceOffset * bytesPerElem,
           currDimSize * numElementsAfterTargetDim * bytesPerElem);

    shapes_dim_t dimOffset = currDimSize;
    for (RANGE(ist, tensors->size)) {
      shapes_Tensor curr = shapes_ArrayTensorIdx(tensorsContig, ist); //tensorsContig[ist];
      shapes_dim_t currTargetDimSize = curr.shape.dims[targetDim];
      shapes_tensor_size_t currDestOffset = destSliceOffset + (dimOffset * numElementsAfterTargetDim);
      shapes_tensor_size_t currSrcOffset = inb * currTargetDimSize * numElementsAfterTargetDim;
      memcpy((char *)dest.values + currDestOffset * bytesPerElem,
             (char *)curr.values + currSrcOffset * bytesPerElem,
             currTargetDimSize * numElementsAfterTargetDim * bytesPerElem);
      dimOffset += currTargetDimSize;
    }
  }

  return dest;
}

shapes_Tensor shapes_Stack(shapes_Context *ctx, shapes_ArrayTensor tensors) {
  PANIC_IF_NULL(ctx);
  PANIC_IF_NULL(tensors);
  PANIC_IF(tensors->size < 2, ERR_STACKING_LESS_THAN_TWO_TENSORS);

  shapes_Tensor firstTensor = shapes_ArrayTensorIdx(tensors, 0);
  olib_Array *unsqueezed = shapes_Make_DynamicTensorArray(ctx->memory);

  for (RANGE_FROM(1, tensors->size, i)) {
    shapes_Tensor currentTensor = shapes_ArrayTensorIdx(tensors, i);
    PANIC_IF(!isSameShape(&firstTensor, &currentTensor), ERR_DIM_MISMATCH);

    shapes_Tensor currentTensorUnsqueezed = shapes_UnSqueeze(ctx, &currentTensor, 0); 
    shapes_ArrayAppendTensor(unsqueezed, &currentTensorUnsqueezed);
  }

  shapes_Tensor firstTensorUnsqueezed = shapes_UnSqueeze(ctx, &firstTensor, 0);
  return shapes_Concat(ctx, &firstTensorUnsqueezed, 0, unsqueezed);
}
