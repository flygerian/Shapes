#include <stdbool.h>
#include <stddef.h>
#include "result.h"
#include "shapes.h"
#include "shapes_internal.h"
#include "value.h"

static shapes_Value *contigousSum(olib_Memory *m, void *position, shapes_tensor_size_t limit, shapes_Dtype dtype) {
  shapes_tensor_size_t x = 0;
  shapes_Value sums[MAX_PARALLEL_SUMS] = {};

  for (u8 i = 0; i < MAX_PARALLEL_SUMS; i++) {
    sums[i] = VALUE(dtype, 0);
  }

  while (x < limit) {
    u8 numComputations = 0;
    for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
      if (x + y >= limit) {
        break;
      }

      shapes_Value val;
      VALUE_GET_FROM_ARR(position, x + y, &val, dtype);
      VALUE_BINOP(sums[y], sums[y], val, +);
      numComputations++;
    }
    x += numComputations;
  }

  shapes_Value *sum = olib_Allocate(m, sizeof(shapes_Value));
  PANIC_IF(sum == NULL, ALLOCATION_FAILED);

  *sum = VALUE(dtype, 0);
  for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
    VALUE_BINOP(*sum, *sum, sums[y], +);
  }

  return sum;
}

static DeviceType getReductionDispatchDevice(shapes_Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return CPU;
  }

  return ctx->device->type;
}

static void validateReductionTensor(shapes_Tensor *t) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
}

static void validateReduceDim(shapes_Tensor *t, shapes_dim_t dim, Result dimError) {
  validateReductionTensor(t);
  PANIC_IF(dim >= t->shape.numOfDims, dimError);
  PANIC_IF(t->size == 0, ERR_NO_OP);
}

static void validateMeanLike(shapes_Tensor *t, Result invalidTypeError) {
  validateReductionTensor(t);
  switch (t->dtype) {
    case F16:
    case F32:
    case F64: return;
    default: PANIC_IF(true, invalidTypeError);
  }
}

static void validateStdTensor(shapes_Tensor *t) {
  validateReductionTensor(t);
  PANIC_IF(t->size < 2, ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES);
  PANIC_IF(isNotFloatType(t), ERR_STD_NOT_FLOAT_TYPE);
}

static Result prepareReductionGeometry(shapes_Tensor *t, shapes_dim_t dim, shapes_tensor_size_t *numBeforeDim,
                                       shapes_tensor_size_t *numAfterDim, shapes_dim_t *reduce) {
  Result result = calculateNumElementsBeforeDim(t, dim, numBeforeDim);
  if (result != OK) {
    return result;
  }

  *reduce = t->shape.dims[dim];
  return calculateNumElementsAfterDim(t, dim, numAfterDim);
}

static shapes_Tensor sumCpu(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;
  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  for (shapes_tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (shapes_tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      shapes_Value acc = VALUE(workingTensor->dtype, 0);
      for (shapes_dim_t r = 0; r < reduce; r++) {
        shapes_tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        shapes_Value value;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &value, workingTensor->dtype);
        VALUE_BINOP(acc, acc, value, +);
      }

      VALUE_SET(dest.values, outer * numAfterDim + inner, acc);
    }
  }


  return dest;
}

static shapes_Tensor meanCpu(shapes_Context *ctx, shapes_Tensor *t) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_Value *tensorSum = contigousSum(ctx->memory, input->values, input->size, input->dtype);
  shapes_Value size = VALUE(input->dtype, input->size);
  shapes_Value mean = VALUE(input->dtype, 0);
  VALUE_BINOP(mean, *tensorSum, size, /);

  shapes_Tensor dest = shapes_MakeZerosTensor(ctx, SCALAR);
  return dest;
}

static shapes_Tensor meanDimCpu(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;

  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, ALLOCATION_FAILED);

  shapes_Tensor dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  for (shapes_tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (shapes_tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      f64 sum = 0.0;
      for (shapes_dim_t r = 0; r < reduce; r++) {
        shapes_tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        shapes_Value value;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &value, workingTensor->dtype);

        switch (workingTensor->dtype) {
          case F16: sum += (f64)value.as.f16; break;
          case F32: sum += (f64)value.as.f32; break;
          case F64: sum += value.as.f64; break;
          default: break;
        }
      }

      shapes_tensor_size_t destIdx = outer * numAfterDim + inner;
      switch (workingTensor->dtype) {
        case F16: ((f16 *)dest.values)[destIdx] = (f16)(sum / (f64)reduce); break;
        case F32: ((f32 *)dest.values)[destIdx] = (f32)(sum / (f64)reduce); break;
        case F64: ((f64 *)dest.values)[destIdx] = sum / (f64)reduce; break;
        default: break;
      }
    }
  }


  return dest;
}

static shapes_Tensor stdCpu(shapes_Context *ctx, shapes_Tensor *t) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_Value *tensorSum = contigousSum(ctx->memory, input->values, input->size, input->dtype);
  f64 mean = 0.0;
  switch (input->dtype) {
    case F16: mean = (f64)tensorSum->as.f16 / (f64)input->size; break;
    case F32: mean = (f64)tensorSum->as.f32 / (f64)input->size; break;
    case F64: mean = tensorSum->as.f64 / (f64)input->size; break;
    default: break;
  }

  f64 deviationSquaredSum = 0.0;
  for (shapes_tensor_size_t i = 0; i < input->size; i++) {
    shapes_Value value;
    VALUE_GET_FROM_ARR(input->values, i, &value, input->dtype);

    f64 current = 0.0;
    switch (input->dtype) {
      case F16: current = (f64)value.as.f16; break;
      case F32: current = (f64)value.as.f32; break;
      case F64: current = value.as.f64; break;
      default: break;
    }

    f64 centered = current - mean;
    deviationSquaredSum += centered * centered;
  }

  f64 std = sqrt(deviationSquaredSum / (f64)(input->size - 1));

  shapes_Tensor dest = shapes_MakeZerosTensor(ctx, SCALAR);
  switch (input->dtype) {
    case F16: ((f16 *)dest.values)[0] = (f16)std; break;
    case F32: ((f32 *)dest.values)[0] = (f32)std; break;
    case F64: ((f64 *)dest.values)[0] = std; break;
    default: break;
  }

  return dest;
}

static shapes_Tensor maxCpu(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;

  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  for (shapes_tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (shapes_tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      shapes_tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      shapes_Value maxValue;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &maxValue, workingTensor->dtype);

      for (shapes_dim_t r = 1; r < reduce; r++) {
        shapes_tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        shapes_Value candidate;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &candidate, workingTensor->dtype);
        if (VALUE_CMP(candidate, maxValue, >, workingTensor->dtype)) {
          maxValue = candidate;
        }
      }

      VALUE_SET(dest.values, outer * numAfterDim + inner, maxValue);
    }
  }


  return dest;
}

static shapes_Tensor argMaxCpu(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;
  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, ALLOCATION_FAILED);

  shapes_Tensor dest = t_Reduced(ctx, workingTensor, dim, I64);
  i64 *destValues = (i64 *)dest.values;
  for (shapes_tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (shapes_tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      shapes_tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      shapes_Value maxValue;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &maxValue, workingTensor->dtype);

      i64 argMax = 0;
      for (shapes_dim_t r = 1; r < reduce; r++) {
        shapes_tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        shapes_Value candidate;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &candidate, workingTensor->dtype);
        if (VALUE_CMP(candidate, maxValue, >, workingTensor->dtype)) {
          maxValue = candidate;
          argMax = (i64)r;
        }
      }

      destValues[outer * numAfterDim + inner] = argMax;
    }
  }


  return dest;
}

static shapes_Tensor sumCuda(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, input, dim, input->dtype);
  result = shapescuda_ReduceDim(input->dtype, input->dtype, REDUCTION_OP_SUM, input->values,
                            dest.values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static shapes_Tensor meanCuda(shapes_Context *ctx, shapes_Tensor *t) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_Tensor dest = shapes_MakeZerosTensor(ctx, SCALAR);
  Result result = shapescuda_ReduceAll(input->dtype, REDUCTION_OP_MEAN, input->values,
                                   dest.values, input->size);
  PANIC_IF(result != OK, result);
  return dest;
}

static shapes_Tensor meanDimCuda(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, input, dim, input->dtype);
  result = shapescuda_ReduceDim(input->dtype, input->dtype, REDUCTION_OP_MEAN, input->values,
                            dest.values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static shapes_Tensor stdCuda(shapes_Context *ctx, shapes_Tensor *t) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_Tensor dest = shapes_MakeZerosTensor(ctx, SCALAR);
  Result result = shapescuda_Std(input->dtype, input->values, dest.values, input->size);
  PANIC_IF(result != OK, result);
  return dest;
}

static shapes_Tensor maxCuda(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, input, dim, input->dtype);
  result = shapescuda_ReduceDim(input->dtype, input->dtype, REDUCTION_OP_MAX, input->values,
                            dest.values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static shapes_Tensor argMaxCuda(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  shapes_Tensor *input = materializeTensorOnContext(ctx, t);

  shapes_tensor_size_t numBeforeDim = 0;
  shapes_tensor_size_t numAfterDim = 0;
  shapes_dim_t reduce = 0;

  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  shapes_Tensor dest = t_Reduced(ctx, input, dim, I64);
  result = shapescuda_ReduceDim(input->dtype, I64, REDUCTION_OP_ARGMAX, input->values,
                            dest.values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

shapes_Tensor shapes_Sum(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  validateReduceDim(t, dim, ERR_SUM_DIM_OUT_OF_BOUNDS);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return sumCuda(ctx, t, dim);
    case CPU:
    default: return sumCpu(ctx, t, dim);
  }
}

shapes_Tensor shapes_ReduceBroadcast(shapes_Context *ctx, shapes_Tensor *input, shapes_Tensor *grad) {
  validateReductionTensor(input);
  validateReductionTensor(grad);

  shapes_Tensor *current = grad;
  i32 dimDiff = (i32)grad->shape.numOfDims - (i32)input->shape.numOfDims;
  if (dimDiff < 0) {
    shapes_Tensor out = shapes_Clone(ctx, current);
    return out;
  }

  for (i32 i = 0; i < dimDiff; i++) {
    shapes_Tensor *summed = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
    PANIC_IF(summed == NULL, ALLOCATION_FAILED);
    *summed = shapes_Sum(ctx, current, 0);
    shapes_Tensor *squeezed = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
    PANIC_IF(squeezed == NULL, ALLOCATION_FAILED);
    *squeezed = shapes_SqueezeDim(ctx, summed, 0);
    current = squeezed;
  }

  for (u8 d = 0; d < input->shape.numOfDims; d++) {
    if (input->shape.dims[d] == 1 && current->shape.dims[d] > 1) {
      shapes_Tensor *summed = olib_Allocate(ctx->memory, sizeof(shapes_Tensor));
      PANIC_IF(summed == NULL, ALLOCATION_FAILED);
      *summed = shapes_Sum(ctx, current, d);
      current = summed;
    }
  }

  shapes_Tensor out = shapes_Clone(ctx, current);
  return out;
}

shapes_Tensor shapes_Mean(shapes_Context *ctx, shapes_Tensor *t) {
  validateMeanLike(t, ERR_MEAN_VALUE_NOT_FLOAT);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return meanCuda(ctx, t);
    case CPU:
    default: return meanCpu(ctx, t);
  }
}

shapes_Tensor shapes_MeanDim(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  validateMeanLike(t, ERR_MEAN_VALUE_NOT_FLOAT);
  PANIC_IF(dim >= t->shape.numOfDims, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return meanDimCuda(ctx, t, dim);
    case CPU:
    default: return meanDimCpu(ctx, t, dim);
  }
}

shapes_Tensor shapes_Std(shapes_Context *ctx, shapes_Tensor *t) {
  validateStdTensor(t);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return stdCuda(ctx, t);
    case CPU:
    default: return stdCpu(ctx, t);
  }
}

shapes_Tensor shapes_Max(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  validateReduceDim(t, dim, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return maxCuda(ctx, t, dim);
    case CPU:
    default: return maxCpu(ctx, t, dim);
  }
}

shapes_Tensor shapes_ArgMax(shapes_Context *ctx, shapes_Tensor *t, shapes_dim_t dim) {
  validateReduceDim(t, dim, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return argMaxCuda(ctx, t, dim);
    case CPU:
    default: return argMaxCpu(ctx, t, dim);
  }
}
