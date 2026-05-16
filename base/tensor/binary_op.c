#include "common.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor/types.h"
#include "tensor_internal.h"
#include "value.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include "binary_op_helpers.h"


static bool isComparisonOp(OpType opType) {
  return opType == OP_GREATER || opType == OP_GREATER_OR_EQUAL || opType == OP_LESS ||
         opType == OP_LESS_OR_EQUAL || opType == OP_EQUAL;
}

static bool isArithmeticOp(OpType opType) {
  return opType == OP_ADD || opType == OP_SUBTRACT || opType == OP_MULTIPLY;
}

static bool isCudaContext(Context *ctx) {
  return ctx != NULL && ctx->device != NULL && ctx->device->type == CUDA;
}

static bool areTensorsSameShape(Tensor *a, Tensor *b) {
  if (a->shape.numOfDims != b->shape.numOfDims) {
    return false;
  }

  for (tensor_size_t i = 0; i < a->shape.numOfDims; i++) {
    if (a->shape.dims[i] != b->shape.dims[i]) {
      return false;
    }
  }

  return true;
}

static Result straightArithBinop(Tensor *a, Tensor *b, Tensor *dest, OpType opType) {
  tensor_size_t n = dest->size;

  switch (a->dtype) {
    case BOOL: {
      const bool *restrict pa = a->values;
      const bool *restrict pb = b->values;
      bool *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_bool(pa, pb, po, n), subtract_bool(pa, pb, po, n),
                      multiply_bool(pa, pb, po, n));
      return OK;
    }
    case U8: {
      const u8 *restrict pa = a->values;
      const u8 *restrict pb = b->values;
      u8 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_u8(pa, pb, po, n), subtract_u8(pa, pb, po, n),
                      multiply_u8(pa, pb, po, n));
      return OK;
    }
    case U16: {
      const u16 *restrict pa = a->values;
      const u16 *restrict pb = b->values;
      u16 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_u16(pa, pb, po, n), subtract_u16(pa, pb, po, n),
                      multiply_u16(pa, pb, po, n));
      return OK;
    }
    case U32: {
      const u32 *restrict pa = a->values;
      const u32 *restrict pb = b->values;
      u32 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_u32(pa, pb, po, n), subtract_u32(pa, pb, po, n),
                      multiply_u32(pa, pb, po, n));
      return OK;
    }
    case U64: {
      const u64 *restrict pa = a->values;
      const u64 *restrict pb = b->values;
      u64 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_u64(pa, pb, po, n), subtract_u64(pa, pb, po, n),
                      multiply_u64(pa, pb, po, n));
      return OK;
    }
    case I8: {
      const i8 *restrict pa = a->values;
      const i8 *restrict pb = b->values;
      i8 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_i8(pa, pb, po, n), subtract_i8(pa, pb, po, n),
                      multiply_i8(pa, pb, po, n));
      return OK;
    }
    case I16: {
      const i16 *restrict pa = a->values;
      const i16 *restrict pb = b->values;
      i16 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_i16(pa, pb, po, n), subtract_i16(pa, pb, po, n),
                      multiply_i16(pa, pb, po, n));
      return OK;
    }
    case I32: {
      const i32 *restrict pa = a->values;
      const i32 *restrict pb = b->values;
      i32 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_i32(pa, pb, po, n), subtract_i32(pa, pb, po, n),
                      multiply_i32(pa, pb, po, n));
      return OK;
    }
    case I64: {
      const i64 *restrict pa = a->values;
      const i64 *restrict pb = b->values;
      i64 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_i64(pa, pb, po, n), subtract_i64(pa, pb, po, n),
                      multiply_i64(pa, pb, po, n));
      return OK;
    }
    case F16:
    case F32: {
      const f32 *restrict pa = a->values;
      const f32 *restrict pb = b->values;
      f32 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_f32(pa, pb, po, n), subtract_f32(pa, pb, po, n),
                      multiply_f32(pa, pb, po, n));
      return OK;
    }
    case F64: {
      const f64 *restrict pa = a->values;
      const f64 *restrict pb = b->values;
      f64 *restrict po = dest->values;
      SWITCH_ARITH_OP(opType, add_f64(pa, pb, po, n), subtract_f64(pa, pb, po, n),
                      multiply_f64(pa, pb, po, n));
      return OK;
    }
    default: return ERR_NOT_A_BINOP;
  }
}

static Result straightBinop(Tensor *a, Tensor *b, Tensor *dest, OpType opType) {
  if (opType == OP_ADD || opType == OP_SUBTRACT || opType == OP_MULTIPLY) {
    return straightArithBinop(a, b, dest, opType);
  }

  switch (opType) {
    case OP_GREATER:
      switch (a->dtype) {
        case BOOL: STRAIGHT_CMP_LOOP(bool, >);
        case U8: STRAIGHT_CMP_LOOP(u8, >);
        case U16: STRAIGHT_CMP_LOOP(u16, >);
        case U32: STRAIGHT_CMP_LOOP(u32, >);
        case U64: STRAIGHT_CMP_LOOP(u64, >);
        case I8: STRAIGHT_CMP_LOOP(i8, >);
        case I16: STRAIGHT_CMP_LOOP(i16, >);
        case I32: STRAIGHT_CMP_LOOP(i32, >);
        case I64: STRAIGHT_CMP_LOOP(i64, >);
        case F16: STRAIGHT_CMP_LOOP(float, >);
        case F32: STRAIGHT_CMP_LOOP(float, >);
        case F64: STRAIGHT_CMP_LOOP(double, >);
      }
      break;
    case OP_GREATER_OR_EQUAL:
      switch (a->dtype) {
        case BOOL: STRAIGHT_CMP_LOOP(bool, >=);
        case U8: STRAIGHT_CMP_LOOP(u8, >=);
        case U16: STRAIGHT_CMP_LOOP(u16, >=);
        case U32: STRAIGHT_CMP_LOOP(u32, >=);
        case U64: STRAIGHT_CMP_LOOP(u64, >=);
        case I8: STRAIGHT_CMP_LOOP(i8, >=);
        case I16: STRAIGHT_CMP_LOOP(i16, >=);
        case I32: STRAIGHT_CMP_LOOP(i32, >=);
        case I64: STRAIGHT_CMP_LOOP(i64, >=);
        case F16: STRAIGHT_CMP_LOOP(float, >=);
        case F32: STRAIGHT_CMP_LOOP(float, >=);
        case F64: STRAIGHT_CMP_LOOP(double, >=);
      }
      break;
    case OP_LESS:
      switch (a->dtype) {
        case BOOL: STRAIGHT_CMP_LOOP(bool, <);
        case U8: STRAIGHT_CMP_LOOP(u8, <);
        case U16: STRAIGHT_CMP_LOOP(u16, <);
        case U32: STRAIGHT_CMP_LOOP(u32, <);
        case U64: STRAIGHT_CMP_LOOP(u64, <);
        case I8: STRAIGHT_CMP_LOOP(i8, <);
        case I16: STRAIGHT_CMP_LOOP(i16, <);
        case I32: STRAIGHT_CMP_LOOP(i32, <);
        case I64: STRAIGHT_CMP_LOOP(i64, <);
        case F16: STRAIGHT_CMP_LOOP(float, <);
        case F32: STRAIGHT_CMP_LOOP(float, <);
        case F64: STRAIGHT_CMP_LOOP(double, <);
      }
      break;
    case OP_LESS_OR_EQUAL:
      switch (a->dtype) {
        case BOOL: STRAIGHT_CMP_LOOP(bool, <=);
        case U8: STRAIGHT_CMP_LOOP(u8, <=);
        case U16: STRAIGHT_CMP_LOOP(u16, <=);
        case U32: STRAIGHT_CMP_LOOP(u32, <=);
        case U64: STRAIGHT_CMP_LOOP(u64, <=);
        case I8: STRAIGHT_CMP_LOOP(i8, <=);
        case I16: STRAIGHT_CMP_LOOP(i16, <=);
        case I32: STRAIGHT_CMP_LOOP(i32, <=);
        case I64: STRAIGHT_CMP_LOOP(i64, <=);
        case F16: STRAIGHT_CMP_LOOP(float, <=);
        case F32: STRAIGHT_CMP_LOOP(float, <=);
        case F64: STRAIGHT_CMP_LOOP(double, <=);
      }
      break;
    case OP_EQUAL:
      switch (a->dtype) {
        case BOOL: STRAIGHT_CMP_LOOP(bool, ==);
        case U8: STRAIGHT_CMP_LOOP(u8, ==);
        case U16: STRAIGHT_CMP_LOOP(u16, ==);
        case U32: STRAIGHT_CMP_LOOP(u32, ==);
        case U64: STRAIGHT_CMP_LOOP(u64, ==);
        case I8: STRAIGHT_CMP_LOOP(i8, ==);
        case I16: STRAIGHT_CMP_LOOP(i16, ==);
        case I32: STRAIGHT_CMP_LOOP(i32, ==);
        case I64: STRAIGHT_CMP_LOOP(i64, ==);
        case F16: STRAIGHT_CMP_LOOP(float, ==);
        case F32: STRAIGHT_CMP_LOOP(float, ==);
        case F64: STRAIGHT_CMP_LOOP(double, ==);
      }
      break;
    default: return ERR_NOT_A_BINOP;
  }

  return ERR_NOT_A_BINOP;
}

static inline void unravel_index(tensor_size_t flatIdx, Dim *shape, dim_t *destCoords) {
  for (int d = shape->numOfDims - 1; d >= 0; d--) {
    destCoords[d] = flatIdx % shape->dims[d];
    flatIdx /= shape->dims[d];
  }
}

static Result broadcastBinop(Dim outputShape, Tensor *opA, Tensor *opB, Tensor *output,
                             OpType opType) {
  dim_t currentCoord[output->shape.numOfDims];
  dim_t aCoords[output->shape.numOfDims];
  dim_t bCoords[output->shape.numOfDims];

  PANIC_IF(!areBroadcastable(opA, opB), TENSORS_CANNOT_BE_BROADCASTED);

  dim_t broadcastDim = 0;
  dim_t broadcastDimNumIterations = 0;
  Tensor *widerOperand = opA->size > opB->size ? opA : opB;
  Tensor *smallerOperand = opA->size < opB->size ? opA : opB;
  Tensor *broadcastOperand = NULL;
  Tensor *nonBroadcastOperand = NULL;

  for (dim_t i = widerOperand->shape.numOfDims - 1; i >= 0; i--) {
    dim_t widerOpDim = widerOperand->shape.dims[i];
    dim_t smallerOpDim = smallerOperand->shape.dims[i];

    if (widerOpDim != smallerOpDim) {
      broadcastDim = i;
      broadcastDimNumIterations = widerOpDim > smallerOpDim ? widerOpDim : smallerOpDim;
      broadcastOperand = widerOpDim < smallerOpDim ? widerOperand : smallerOperand;
      nonBroadcastOperand = widerOpDim > smallerOpDim ? widerOperand : smallerOperand;
      break;
    }
  }

  tensor_size_t numOuterIterations;
  calculateNumElementsBeforeDim(widerOperand, broadcastDim, &numOuterIterations);

  tensor_size_t broadcastOperandOuterSize;
  calculateNumElementsBeforeDim(broadcastOperand, broadcastDim, &broadcastOperandOuterSize);

  tensor_size_t nonBroadcastOperandOuterSize;
  calculateNumElementsBeforeDim(nonBroadcastOperand, broadcastDim, &nonBroadcastOperandOuterSize);

  tensor_size_t numInnerIterations;
  calculateNumElementsAfterDim(widerOperand, broadcastDim, &numInnerIterations);

  size_t elemBytes = getBytesForDtype(opA->dtype);
  binopFn doBinaryOperation = getBinopFn(binopTable[opType], opA->dtype);

  for (tensor_size_t outerIdx = 0; outerIdx < numOuterIterations; outerIdx++) {
    for (dim_t broadcastDimIdx = 0; broadcastDimIdx < broadcastDimNumIterations;
         broadcastDimIdx++) {
      tensor_size_t bOuterIdxPresenceMultiplier =
          broadcastOperandOuterSize == numOuterIterations ? 1 : 0;
      tensor_size_t broadcastOperandIdx =
          bOuterIdxPresenceMultiplier * outerIdx * numInnerIterations;
      void *broadcastOperandPos = (u8 *)broadcastOperand->values + broadcastOperandIdx * elemBytes;

      tensor_size_t nbOuterIdxPresenceMultiplier =
          nonBroadcastOperandOuterSize == numOuterIterations ? 1 : 0;
      tensor_size_t nonBroadcastOperandIdx =
          nbOuterIdxPresenceMultiplier *
              (outerIdx * broadcastDimNumIterations * numInnerIterations) +
          (broadcastDimIdx * numInnerIterations);
      void *nonBroadcastOperandPos =
          (u8 *)nonBroadcastOperand->values + nonBroadcastOperandIdx * elemBytes;

      tensor_size_t outputIdx =
          (outerIdx * broadcastDimNumIterations + broadcastDimIdx) * numInnerIterations;
      void *aPos = broadcastOperand == opA ? broadcastOperandPos : nonBroadcastOperandPos;
      void *bPos = broadcastOperand == opA ? nonBroadcastOperandPos : broadcastOperandPos;
      doBinaryOperation(aPos, bPos, (u8 *)output->values + outputIdx * elemBytes,
                        numInnerIterations);
    }
  }


  // for (tensor_size_t x = 0; x < output->size; x++) {
  //   // unravel index into output shape
  //   // eg 7 -> (2, 2, 1)
  //   unravel_index(x, &outputShape, currentCoord);
  //
  //   for (u8 d = 0; d < output->shape.numOfDims; d++) {
  //     aCoords[d] = currentCoord[d] % opA->shape.dims[d];
  //     bCoords[d] = currentCoord[d] % opB->shape.dims[d];
  //   }
  //
  //   Value aVal;
  //   u64 idx = getContigousIdxFromCoord(opA, aCoords);
  //   VALUE_GET_FROM_ARR(opA->values, idx, &aVal, opA->dtype);
  //
  //   Value bVal;
  //   idx = getContigousIdxFromCoord(opB, bCoords);
  //   VALUE_GET_FROM_ARR(opB->values, idx, &bVal, opB->dtype);
  //
  //   Value result;
  //   switch (opType) {
  //     case OP_ADD: VALUE_BINOP(result, aVal, bVal, +); break;
  //     case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
  //     case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;
  //     case OP_GREATER: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, >, opA->dtype)); break;
  //     case OP_GREATER_OR_EQUAL: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, >=, opA->dtype));
  //     break; case OP_LESS: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, <, opA->dtype)); break;
  //     case OP_LESS_OR_EQUAL: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, <=, opA->dtype)); break;
  //
  //     default: return ERR_NOT_A_BINOP;
  //   }
  //
  //   VALUE_SET(output->values, x, result);
  // }

  return OK;
}


static Tensor *binaryOpCpu(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  Tensor *opA = materializeTensorOnContext(ctx, ops.a);
  Tensor *opB = materializeTensorOnContext(ctx, ops.b);

  Dim outputShape;
  if (opA->size > opB->size) {
    outputShape = opA->shape;
  } else {
    outputShape = opB->shape;
  }

  Dtype outputDtype = opA->dtype;
  if (isComparisonOp(opType)) {
    outputDtype = BOOL;
  }

  Tensor *output = t_Zeros(ctx, outputShape, outputDtype);
  PANIC_IF(output == NULL, ALLOCATION_FAILED);

  Result res;
  if (areTensorsSameShape(opA, opB)) {
    res = straightBinop(opA, opB, output, opType);
  } else {
    res = broadcastBinop(outputShape, opA, opB, output, opType);
  }

  PANIC_IF(res != OK, res);

  return output;
}


static Tensor *binaryOpCuda(Context *ctx, Tensor *a, Tensor *b, OpType opType) {

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  Tensor *opA = materializeTensorOnContext(ctx, ops.a);
  Tensor *opB = materializeTensorOnContext(ctx, ops.b);

  PANIC_IF(!areTensorsSameShape(opA, opB), ERR_DIM_MISMATCH);

  Tensor *output = t_Zeros(ctx, opA->shape, opA->dtype);
  Result res = runCudaBinaryOp(ctx, opA->dtype, opType, opA->values, opB->values, output->values,
                               output->size);
  return output;
}

static Tensor *binaryOp(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);

  DeviceType deviceType = ctx->device == NULL ? CPU : ctx->device->type;

  switch (deviceType) {
    case CUDA: return binaryOpCuda(ctx, a, b, opType);
    case CPU:
    default: return binaryOpCpu(ctx, a, b, opType);
  }
}

Tensor *Add(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_ADD);
}

static inline ValuePair getValueOperandsForInplaceBinop(Tensor *a, Tensor *opB, tensor_size_t idx,
                                                        dim_t *currentCoord, dim_t *bCoords) {
  unravel_index(idx, &a->shape, currentCoord);

  // Compute storage index in a (accounts for per-dim boundary via getContigousIdxFromCoord).
  u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);

  for (u8 d = 0; d < a->shape.numOfDims; d++) {
    bCoords[d] = currentCoord[d] % opB->shape.dims[d];
  }

  Value aVal;
  VALUE_GET_FROM_ARR(a->values, aStorageIdx, &aVal, a->dtype);

  Value bVal;
  u64 bIdx = getContigousIdxFromCoord(opB, bCoords);
  VALUE_GET_FROM_ARR(opB->values, bIdx, &bVal, opB->dtype);

  return (ValuePair){.a = aVal, .b = bVal};
}

static inline void inPlaceBinopAdd(Tensor *a, Tensor *opB) {
  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];
  for (tensor_size_t x = 0; x < a->size; x++) {
    ValuePair pair = getValueOperandsForInplaceBinop(a, opB, x, currentCoord, bCoords);

    Value result;
    VALUE_BINOP(result, pair.a, pair.b, +);
    u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);
    VALUE_SET(a->values, aStorageIdx, result);
  }
}

static inline void inPlaceBinopMultiply(Tensor *a, Tensor *opB) {
  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];
  for (tensor_size_t x = 0; x < a->size; x++) {
    ValuePair pair = getValueOperandsForInplaceBinop(a, opB, x, currentCoord, bCoords);

    Value result;
    VALUE_BINOP(result, pair.a, pair.b, *);
    u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);
    VALUE_SET(a->values, aStorageIdx, result);
  }
}

static inline void inPlaceBinopSubtract(Tensor *a, Tensor *opB) {
  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];
  for (tensor_size_t x = 0; x < a->size; x++) {
    ValuePair pair = getValueOperandsForInplaceBinop(a, opB, x, currentCoord, bCoords);

    Value result;
    VALUE_BINOP(result, pair.a, pair.b, -);
    u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);
    VALUE_SET(a->values, aStorageIdx, result);
  }
}

static inline void inPlaceBinopDivide(Tensor *a, Tensor *opB) {
  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];
  for (tensor_size_t x = 0; x < a->size; x++) {
    ValuePair pair = getValueOperandsForInplaceBinop(a, opB, x, currentCoord, bCoords);

    Value result;
    VALUE_BINOP(result, pair.a, pair.b, /);
    u64 aStorageIdx = getContigousIdxFromCoord(a, currentCoord);
    VALUE_SET(a->values, aStorageIdx, result);
  }
}

static void inPlaceBinopCpu(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  Tensor *opA = a;
  Tensor *opB = b;
  Tensor *paddedB = NULL;
  PANIC_IF(NUM_DIMS(a) != NUM_DIMS(b) && NUM_DIMS(b) != 1, ERR_DIM_MISMATCH);

  if (b->shape.numOfDims > 1) {
    for (dim_t d = 1; d <= b->shape.numOfDims; d++) {
      dim_t aDim = a->shape.dims[a->shape.numOfDims - d];
      dim_t bDim = b->shape.dims[b->shape.numOfDims - d];

      PANIC_IF(aDim != bDim && bDim != 1, ERR_DIM_MISMATCH);
    }
    // TensorPair ops = padSmallerTensor(ctx, a, b);
    // opB = ops.b;
    // paddedB = ops.b;
  }

  if (!a->isContigous) {
    opA = copyToContiguous(ctx, a);
  }

  opB = materializeTensorOnContext(ctx, opB);

  binopFn fn = getBinopFn(binopTable[opType], opA->dtype);
  PANIC_IF(fn == NULL, ERR_NOT_A_BINOP);

  if (opA->isContigous && opB->isContigous && areTensorsSameShape(opA, opB)) {
    fn(opA->values, opB->values, opA->values, opA->size);
  } else {
    Dim outputShape = opA->size > opB->size ? opA->shape : opB->shape;
    broadcastBinop(outputShape, opA, opB, opA, opType);
  }
}

static void inPlaceBinopCuda(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  PANIC_IF(!isArithmeticOp(opType) || !a->isContigous || a->isView, ERR_NO_OP);
  PANIC_IF(a->context->device->type != CUDA, ERR_DEVICE_MISMATCH);
  PANIC_IF(b->context->device->type != CUDA, ERR_DEVICE_MISMATCH);
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(!areBroadcastable(a, b), ERR_DIM_MISMATCH);

  Tensor *opB = b;
  Tensor *paddedB = NULL;
  if (a->shape.numOfDims != b->shape.numOfDims) {
    TensorPair ops = padSmallerTensor(ctx, a, b);
    opB = ops.b;
    paddedB = ops.b;
  }

  opB = materializeTensorOnContext(ctx, opB);

  PANIC_IF(!areTensorsSameShape(a, opB), ERR_DTYPE_MISMATCH);

  Result res = runCudaBinaryOp(ctx, a->dtype, opType, a->values, opB->values, a->values, a->size);
  PANIC_IF(res != OK, res);
}


static void inPlaceBinop(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  PANIC_IF(a->dtype != b->dtype, ERR_DTYPE_MISMATCH);
  switch (ctx != NULL && ctx->device != NULL ? ctx->device->type : CPU) {
    case CUDA:
      PANIC_IF(!areTensorsSameShape(a, b) || a->shape.numOfDims != b->shape.numOfDims,
               ERR_DTYPE_MISMATCH);
      inPlaceBinopCuda(ctx, a, b, opType);
      return;
    case CPU:
    default: inPlaceBinopCpu(ctx, a, b, opType);
  }
}

void AddInPlace(Context *ctx, Tensor *a, Tensor *b) {
  inPlaceBinop(ctx, a, b, OP_ADD);
}

void SubtractInPlace(Context *ctx, Tensor *a, Tensor *b) {
  inPlaceBinop(ctx, a, b, OP_SUBTRACT);
}

void MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b) {
  inPlaceBinop(ctx, a, b, OP_MULTIPLY);
}

Tensor *Subtract(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_SUBTRACT);
}

Tensor *Multiply(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_MULTIPLY);
}

Tensor *Divide(Context *ctx, Tensor *numerator, Tensor *denominator) {
  // Implement division as: numerator / denominator = numerator * (denominator^-1)
  // This automatically gets correct gradients through the computation graph!

  Tensor *denom_inv = Pow(ctx, denominator, -1.0f);
  return Multiply(ctx, numerator, denom_inv);
}

Tensor *GreaterThan(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_GREATER);
}

Tensor *GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_GREATER_OR_EQUAL);
}

Tensor *Equal(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_EQUAL);
}

Tensor *LessThan(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_LESS);
}

Tensor *LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b) {
  return binaryOp(ctx, a, b, OP_LESS_OR_EQUAL);
}
