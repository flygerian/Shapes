#include <stdbool.h>
#include <stddef.h>
#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "value.h"

static Value *contigousSum(Memory *m, void *position, tensor_size_t limit, Dtype dtype) {
  tensor_size_t x = 0;
  Value sums[MAX_PARALLEL_SUMS] = {};

  for (u8 i = 0; i < MAX_PARALLEL_SUMS; i++) {
    sums[i] = VALUE(dtype, 0);
  }

  while (x < limit) {
    u8 numComputations = 0;
    for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
      if (x + y >= limit) {
        break;
      }

      Value val;
      VALUE_GET_FROM_ARR(position, x + y, &val, dtype);
      VALUE_BINOP(sums[y], sums[y], val, +);
      numComputations++;
    }
    x += numComputations;
  }

  Value *sum = allocate(m, sizeof(Value));
  PANIC_IF(sum == NULL, ALLOCATION_FAILED);

  *sum = VALUE(dtype, 0);
  for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
    VALUE_BINOP(*sum, *sum, sums[y], +);
  }

  return sum;
}

static DeviceType getReductionDispatchDevice(Context *ctx) {
  if (ctx == NULL || ctx->device == NULL) {
    return CPU;
  }

  return ctx->device->type;
}

static void validateReductionTensor(Tensor *t) {
  PANIC_IF(isInvalidTensor(t), ERR_NULL_TENSOR_PROVIDED);
}

static void validateReduceDim(Tensor *t, dim_t dim, Result dimError) {
  validateReductionTensor(t);
  PANIC_IF(dim >= t->shape.numOfDims, dimError);
  PANIC_IF(t->size == 0, ERR_NO_OP);
}

static void validateMeanLike(Tensor *t, Result invalidTypeError) {
  validateReductionTensor(t);
  switch (t->dtype) {
    case F16:
    case F32:
    case F64: return;
    default: PANIC_IF(true, invalidTypeError);
  }
}

static void validateStdTensor(Tensor *t) {
  validateReductionTensor(t);
  PANIC_IF(t->size < 2, ERR_STD_REQUIRES_AT_LEAST_TWO_VALUES);
  PANIC_IF(isNotFloatType(t), ERR_STD_NOT_FLOAT_TYPE);
}

static Result prepareReductionGeometry(Tensor *t, dim_t dim, tensor_size_t *numBeforeDim,
                                       tensor_size_t *numAfterDim, dim_t *reduce) {
  Result result = calculateNumElementsBeforeDim(t, dim, numBeforeDim);
  if (result != OK) {
    return result;
  }

  *reduce = t->shape.dims[dim];
  return calculateNumElementsAfterDim(t, dim, numAfterDim);
}

static Tensor *sumCpu(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;
  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      Value acc = VALUE(workingTensor->dtype, 0);
      for (dim_t r = 0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value value;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &value, workingTensor->dtype);
        VALUE_BINOP(acc, acc, value, +);
      }

      VALUE_SET(dest->values, outer * numAfterDim + inner, acc);
    }
  }


  return dest;
}

static Tensor *meanCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Value *tensorSum = contigousSum(ctx->memory, input->values, input->size, input->dtype);
  Value size = VALUE(input->dtype, input->size);
  Value mean = VALUE(input->dtype, 0);
  VALUE_BINOP(mean, *tensorSum, size, /);

  Tensor *dest = shapes_Make_ZerosTensor(ctx, SCALAR);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  return dest;
}

static Tensor *meanDimCpu(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;

  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, ALLOCATION_FAILED);

  Tensor *dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      f64 sum = 0.0;
      for (dim_t r = 0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value value;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &value, workingTensor->dtype);

        switch (workingTensor->dtype) {
          case F16: sum += (f64)value.as.f16; break;
          case F32: sum += (f64)value.as.f32; break;
          case F64: sum += value.as.f64; break;
          default: break;
        }
      }

      tensor_size_t destIdx = outer * numAfterDim + inner;
      switch (workingTensor->dtype) {
        case F16: ((f16 *)dest->values)[destIdx] = (f16)(sum / (f64)reduce); break;
        case F32: ((f32 *)dest->values)[destIdx] = (f32)(sum / (f64)reduce); break;
        case F64: ((f64 *)dest->values)[destIdx] = sum / (f64)reduce; break;
        default: break;
      }
    }
  }


  return dest;
}

static Tensor *stdCpu(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Value *tensorSum = contigousSum(ctx->memory, input->values, input->size, input->dtype);
  f64 mean = 0.0;
  switch (input->dtype) {
    case F16: mean = (f64)tensorSum->as.f16 / (f64)input->size; break;
    case F32: mean = (f64)tensorSum->as.f32 / (f64)input->size; break;
    case F64: mean = tensorSum->as.f64 / (f64)input->size; break;
    default: break;
  }

  f64 deviationSquaredSum = 0.0;
  for (tensor_size_t i = 0; i < input->size; i++) {
    Value value;
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

  Tensor *dest = shapes_Make_ZerosTensor(ctx, SCALAR);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  switch (input->dtype) {
    case F16: ((f16 *)dest->values)[0] = (f16)std; break;
    case F32: ((f32 *)dest->values)[0] = (f32)std; break;
    case F64: ((f64 *)dest->values)[0] = std; break;
    default: break;
  }

  return dest;
}

static Tensor *maxCpu(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;

  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, workingTensor, dim, workingTensor->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      Value maxValue;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &maxValue, workingTensor->dtype);

      for (dim_t r = 1; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value candidate;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &candidate, workingTensor->dtype);
        if (VALUE_CMP(candidate, maxValue, >, workingTensor->dtype)) {
          maxValue = candidate;
        }
      }

      VALUE_SET(dest->values, outer * numAfterDim + inner, maxValue);
    }
  }


  return dest;
}

static Tensor *argMaxCpu(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *workingTensor = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;
  Result result =
      prepareReductionGeometry(workingTensor, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, ALLOCATION_FAILED);

  Tensor *dest = t_Reduced(ctx, workingTensor, dim, I64);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  i64 *destValues = (i64 *)dest->values;
  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      tensor_size_t firstIdx = outer * reduce * numAfterDim + inner;
      Value maxValue;
      VALUE_GET_FROM_ARR(workingTensor->values, firstIdx, &maxValue, workingTensor->dtype);

      i64 argMax = 0;
      for (dim_t r = 1; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
        Value candidate;
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

static Tensor *sumCuda(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, input, dim, input->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  result = runCudaReduceDim(ctx, input->dtype, input->dtype, REDUCTION_OP_SUM, input->values,
                            dest->values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static Tensor *meanCuda(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *dest = shapes_Make_ZerosTensor(ctx, SCALAR);
  Result result = runCudaReduceAll(ctx, input->dtype, REDUCTION_OP_MEAN, input->values,
                                   dest->values, input->size);
  PANIC_IF(result != OK, result);
  return dest;
}

static Tensor *meanDimCuda(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, input, dim, input->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  result = runCudaReduceDim(ctx, input->dtype, input->dtype, REDUCTION_OP_MEAN, input->values,
                            dest->values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static Tensor *stdCuda(Context *ctx, Tensor *t) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  Tensor *dest = shapes_Make_ZerosTensor(ctx, SCALAR);

  Result result = runCudaStd(ctx, input->dtype, input->values, dest->values, input->size);
  PANIC_IF(result != OK, result);
  return dest;
}

static Tensor *maxCuda(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;
  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, input, dim, input->dtype);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  result = runCudaReduceDim(ctx, input->dtype, input->dtype, REDUCTION_OP_MAX, input->values,
                            dest->values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

static Tensor *argMaxCuda(Context *ctx, Tensor *t, dim_t dim) {
  Tensor *input = materializeTensorOnContext(ctx, t);

  tensor_size_t numBeforeDim = 0;
  tensor_size_t numAfterDim = 0;
  dim_t reduce = 0;

  Result result = prepareReductionGeometry(input, dim, &numBeforeDim, &numAfterDim, &reduce);
  PANIC_IF(result != OK, result);

  Tensor *dest = t_Reduced(ctx, input, dim, I64);
  PANIC_IF(dest == NULL, ALLOCATION_FAILED);

  result = runCudaReduceDim(ctx, input->dtype, I64, REDUCTION_OP_ARGMAX, input->values,
                            dest->values, numBeforeDim, numAfterDim, reduce);
  PANIC_IF(result != OK, result);
  return dest;
}

Tensor *shapes_Sum(Context *ctx, Tensor *t, dim_t dim) {
  validateReduceDim(t, dim, ERR_SUM_DIM_OUT_OF_BOUNDS);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return sumCuda(ctx, t, dim);
    case CPU:
    default: return sumCpu(ctx, t, dim);
  }
}

Tensor *shapes_ReduceBroadcast(Context *ctx, Tensor *input, Tensor *grad) {
  validateReductionTensor(input);
  validateReductionTensor(grad);

  Tensor *current = grad;
  i32 dimDiff = (i32)grad->shape.numOfDims - (i32)input->shape.numOfDims;
  if (dimDiff < 0) {
    return shapes_Clone(ctx, current);
  }

  for (i32 i = 0; i < dimDiff; i++) {
    Tensor *summed = shapes_Sum(ctx, current, 0);
    current = shapes_SqueezeDim(ctx, summed, 0);
  }

  for (u8 d = 0; d < input->shape.numOfDims; d++) {
    if (input->shape.dims[d] == 1 && current->shape.dims[d] > 1) {
      current = shapes_Sum(ctx, current, d);
    }
  }

  return shapes_Clone(ctx, current);
}

Tensor *shapes_Mean(Context *ctx, Tensor *t) {
  validateMeanLike(t, ERR_MEAN_VALUE_NOT_FLOAT);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return meanCuda(ctx, t);
    case CPU:
    default: return meanCpu(ctx, t);
  }
}

Tensor *shapes_MeanDim(Context *ctx, Tensor *t, dim_t dim) {
  validateMeanLike(t, ERR_MEAN_VALUE_NOT_FLOAT);
  PANIC_IF(dim >= t->shape.numOfDims, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return meanDimCuda(ctx, t, dim);
    case CPU:
    default: return meanDimCpu(ctx, t, dim);
  }
}

Tensor *shapes_Std(Context *ctx, Tensor *t) {
  validateStdTensor(t);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return stdCuda(ctx, t);
    case CPU:
    default: return stdCpu(ctx, t);
  }
}

Tensor *shapes_Max(Context *ctx, Tensor *t, dim_t dim) {
  validateReduceDim(t, dim, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return maxCuda(ctx, t, dim);
    case CPU:
    default: return maxCpu(ctx, t, dim);
  }
}

Tensor *shapes_ArgMax(Context *ctx, Tensor *t, dim_t dim) {
  validateReduceDim(t, dim, ERR_DIM_MISMATCH);

  switch (getReductionDispatchDevice(ctx)) {
    case CUDA: return argMaxCuda(ctx, t, dim);
    case CPU:
    default: return argMaxCpu(ctx, t, dim);
  }
}
