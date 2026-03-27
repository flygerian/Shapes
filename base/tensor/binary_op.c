#include "common.h"
#include "../memory.h"
#include "result/result.h"
#include "shapes.h"
#include "tensor_internal.h"
#include "value.h"
#include "unary.h"
#include <stdbool.h>
#include <stddef.h>

#define STRAIGHT_CMP_LOOP(TYPE, op)                                                                \
  do {                                                                                             \
    TYPE *pa = a->values;                                                                          \
    TYPE *pb = b->values;                                                                          \
    bool *po = dest->values;                                                                       \
    for (tensor_size_t i = 0; i < dest->size; i++) {                                               \
      po[i] = pa[i] op pb[i];                                                                      \
    }                                                                                              \
    return OK;                                                                                     \
  } while (0)

static bool isComparisonOp(OpType opType) {
  return opType == OP_GREATER || opType == OP_GREATER_OR_EQUAL || opType == OP_LESS ||
         opType == OP_LESS_OR_EQUAL;
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

#define SWITCH_ARITH_OP(OP_TYPE, ADD_EXPR, SUB_EXPR, MUL_EXPR)                                    \
  switch (OP_TYPE) {                                                                               \
    case OP_ADD: ADD_EXPR; break;                                                                  \
    case OP_SUBTRACT: SUB_EXPR; break;                                                             \
    case OP_MULTIPLY: MUL_EXPR; break;                                                             \
    default: return ERR_NOT_A_BINOP;                                                               \
  }

#define DEFINE_ARITH_HELPERS(TYPE, NAME)                                                           \
  static inline void add_##NAME(const TYPE *restrict a, const TYPE *restrict b,                    \
                                TYPE *restrict out, tensor_size_t n) {                             \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      out[i] = a[i] + b[i];                                                                        \
    }                                                                                              \
  }                                                                                                \
  static inline void subtract_##NAME(const TYPE *restrict a, const TYPE *restrict b,               \
                                     TYPE *restrict out, tensor_size_t n) {                        \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      out[i] = a[i] - b[i];                                                                        \
    }                                                                                              \
  }                                                                                                \
  static inline void multiply_##NAME(const TYPE *restrict a, const TYPE *restrict b,               \
                                     TYPE *restrict out, tensor_size_t n) {                        \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      out[i] = a[i] * b[i];                                                                        \
    }                                                                                              \
  }

DEFINE_ARITH_HELPERS(bool, bool)
DEFINE_ARITH_HELPERS(u8, u8)
DEFINE_ARITH_HELPERS(u16, u16)
DEFINE_ARITH_HELPERS(u32, u32)
DEFINE_ARITH_HELPERS(u64, u64)
DEFINE_ARITH_HELPERS(i8, i8)
DEFINE_ARITH_HELPERS(i16, i16)
DEFINE_ARITH_HELPERS(i32, i32)
DEFINE_ARITH_HELPERS(i64, i64)
DEFINE_ARITH_HELPERS(f32, f32)
DEFINE_ARITH_HELPERS(f64, f64)

#define DEFINE_INPLACE_ARITH_HELPERS(TYPE, NAME)                                                   \
  static inline void add_inplace_##NAME(TYPE *a, const TYPE *b, tensor_size_t n) {                 \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      a[i] += b[i];                                                                                \
    }                                                                                              \
  }                                                                                                \
  static inline void subtract_inplace_##NAME(TYPE *a, const TYPE *b, tensor_size_t n) {            \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      a[i] -= b[i];                                                                                \
    }                                                                                              \
  }                                                                                                \
  static inline void multiply_inplace_##NAME(TYPE *a, const TYPE *b, tensor_size_t n) {            \
    for (tensor_size_t i = 0; i < n; i++) {                                                        \
      a[i] *= b[i];                                                                                \
    }                                                                                              \
  }

DEFINE_INPLACE_ARITH_HELPERS(bool, bool)
DEFINE_INPLACE_ARITH_HELPERS(u8, u8)
DEFINE_INPLACE_ARITH_HELPERS(u16, u16)
DEFINE_INPLACE_ARITH_HELPERS(u32, u32)
DEFINE_INPLACE_ARITH_HELPERS(u64, u64)
DEFINE_INPLACE_ARITH_HELPERS(i8, i8)
DEFINE_INPLACE_ARITH_HELPERS(i16, i16)
DEFINE_INPLACE_ARITH_HELPERS(i32, i32)
DEFINE_INPLACE_ARITH_HELPERS(i64, i64)
DEFINE_INPLACE_ARITH_HELPERS(f32, f32)
DEFINE_INPLACE_ARITH_HELPERS(f64, f64)

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

static Result straightInPlaceBinop(Tensor *a, Tensor *b, OpType opType) {
  tensor_size_t n = a->size;

  switch (a->dtype) {
    case BOOL: {
      bool *pa = a->values;
      const bool *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_bool(pa, pb, n), subtract_inplace_bool(pa, pb, n),
                      multiply_inplace_bool(pa, pb, n));
      return OK;
    }
    case U8: {
      u8 *pa = a->values;
      const u8 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_u8(pa, pb, n), subtract_inplace_u8(pa, pb, n),
                      multiply_inplace_u8(pa, pb, n));
      return OK;
    }
    case U16: {
      u16 *pa = a->values;
      const u16 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_u16(pa, pb, n), subtract_inplace_u16(pa, pb, n),
                      multiply_inplace_u16(pa, pb, n));
      return OK;
    }
    case U32: {
      u32 *pa = a->values;
      const u32 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_u32(pa, pb, n), subtract_inplace_u32(pa, pb, n),
                      multiply_inplace_u32(pa, pb, n));
      return OK;
    }
    case U64: {
      u64 *pa = a->values;
      const u64 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_u64(pa, pb, n), subtract_inplace_u64(pa, pb, n),
                      multiply_inplace_u64(pa, pb, n));
      return OK;
    }
    case I8: {
      i8 *pa = a->values;
      const i8 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_i8(pa, pb, n), subtract_inplace_i8(pa, pb, n),
                      multiply_inplace_i8(pa, pb, n));
      return OK;
    }
    case I16: {
      i16 *pa = a->values;
      const i16 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_i16(pa, pb, n), subtract_inplace_i16(pa, pb, n),
                      multiply_inplace_i16(pa, pb, n));
      return OK;
    }
    case I32: {
      i32 *pa = a->values;
      const i32 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_i32(pa, pb, n), subtract_inplace_i32(pa, pb, n),
                      multiply_inplace_i32(pa, pb, n));
      return OK;
    }
    case I64: {
      i64 *pa = a->values;
      const i64 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_i64(pa, pb, n), subtract_inplace_i64(pa, pb, n),
                      multiply_inplace_i64(pa, pb, n));
      return OK;
    }
    case F16:
    case F32: {
      f32 *pa = a->values;
      const f32 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_f32(pa, pb, n), subtract_inplace_f32(pa, pb, n),
                      multiply_inplace_f32(pa, pb, n));
      return OK;
    }
    case F64: {
      f64 *pa = a->values;
      const f64 *pb = b->values;
      SWITCH_ARITH_OP(opType, add_inplace_f64(pa, pb, n), subtract_inplace_f64(pa, pb, n),
                      multiply_inplace_f64(pa, pb, n));
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
    default: return ERR_NOT_A_BINOP;
  }

  return ERR_NOT_A_BINOP;
}

static Result broadcastBinop(Dim outputShape, Tensor *opA, Tensor *opB, Tensor *output,
                             OpType opType) {
  dim_t currentCoord[output->shape.numOfDims];
  dim_t aCoords[output->shape.numOfDims];
  dim_t bCoords[output->shape.numOfDims];

  for (tensor_size_t x = 0; x < output->size; x++) {
    // unravel index into output shape
    // eg 7 -> (2, 2, 1)
    unravel_index(x, &outputShape, currentCoord);

    for (u8 d = 0; d < output->shape.numOfDims; d++) {
      aCoords[d] = currentCoord[d] % opA->shape.dims[d];
      bCoords[d] = currentCoord[d] % opB->shape.dims[d];
    }

    Value aVal;
    u64 idx = getContigousIdxFromCoord(opA, aCoords);
    VALUE_GET_FROM_ARR(opA->values, idx, &aVal, opA->dtype);

    Value bVal;
    idx = getContigousIdxFromCoord(opB, bCoords);
    VALUE_GET_FROM_ARR(opB->values, idx, &bVal, opB->dtype);

    Value result;
    switch (opType) {
      case OP_ADD: VALUE_BINOP(result, aVal, bVal, +); break;
      case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
      case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;
      case OP_GREATER: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, >, opA->dtype)); break;
      case OP_GREATER_OR_EQUAL: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, >=, opA->dtype)); break;
      case OP_LESS: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, <, opA->dtype)); break;
      case OP_LESS_OR_EQUAL: result = VALUE(BOOL, VALUE_CMP(aVal, bVal, <=, opA->dtype)); break;

      default: return ERR_NOT_A_BINOP;
    }

    VALUE_SET(output->values, x, result);
  }

  return OK;
}

static void cleanupPaddedPair(Context *ctx, Tensor *originalA, Tensor *originalB, TensorPair *ops) {
  if (ops->a != originalA) {
    FreeViewTensor(ctx, ops->a);
  }
  if (ops->b != originalB) {
    FreeViewTensor(ctx, ops->b);
  }
}

static Result copyContiguousTensorIntoTensor(Context *srcCtx, Tensor *src, Tensor *dest) {
  if (src == NULL || dest == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (src->size != dest->size || src->dtype != dest->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  Context *destCtx = dest->context != NULL ? dest->context : srcCtx;
  size_t valueBytes = src->size * getBytesForDtype(src->dtype);

  if (dest->isContigous && !dest->isView) {
    return copyBetweenContexts(srcCtx, destCtx, src->values, dest->values, valueBytes);
  }

  size_t elemBytes = getBytesForDtype(dest->dtype);
  dim_t currentCoord[dest->shape.numOfDims];

  for (tensor_size_t x = 0; x < dest->size; x++) {
    unravel_index(x, &dest->shape, currentCoord);
    u64 destStorageIdx = getContigousIdxFromCoord(dest, currentCoord);

    void *srcPtr = (char *)src->values + x * elemBytes;
    void *destPtr = (char *)dest->values + destStorageIdx * elemBytes;

    Result copyResult = copyBetweenContexts(srcCtx, destCtx, srcPtr, destPtr, elemBytes);
    if (copyResult != OK) {
      return copyResult;
    }
  }

  return OK;
}

static Result binaryOpCpu(Context *ctx, Tensor *a, Tensor *b, Tensor *destination, OpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  TensorArg aArg = {0};
  TensorArg bArg = {0};
  Result res = materializeTensorOnContext(ctx, ops.a, true, &aArg);
  if (res != OK) {
    cleanupPaddedPair(ctx, a, b, &ops);
    return res;
  }
  res = materializeTensorOnContext(ctx, ops.b, true, &bArg);
  if (res != OK) {
    releaseTensorArg(ctx, &aArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return res;
  }

  Tensor *opA = aArg.tensor;
  Tensor *opB = bArg.tensor;

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
  if (output == NULL) {
    releaseTensorArg(ctx, &aArg);
    releaseTensorArg(ctx, &bArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return ERR_OUT_OF_MEMORY;
  }

  if (areTensorsSameShape(opA, opB)) {
    res = straightBinop(opA, opB, output, opType);
  } else {
    res = broadcastBinop(outputShape, opA, opB, output, opType);
  }

  if (res != OK) {
    FreeTensor(ctx, output);
    releaseTensorArg(ctx, &aArg);
    releaseTensorArg(ctx, &bArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return res;
  }

  *destination = *output;
  freeAlloc(ctx->memory, output);

  releaseTensorArg(ctx, &aArg);
  releaseTensorArg(ctx, &bArg);
  cleanupPaddedPair(ctx, a, b, &ops);

  return OK;
}

static Result binaryOpViaCpuFallback(Context *ctx, Tensor *a, Tensor *b, Tensor *destination,
                                     OpType opType) {
  Context cpuCtx = {.memory = ctx->memory};
  Result res = binaryOpCpu(&cpuCtx, a, b, destination, opType);
  if (res != OK) {
    return res;
  }

  return moveTensor(&cpuCtx, ctx, destination);
}

static Result binaryOpCuda(Context *ctx, Tensor *a, Tensor *b, Tensor *destination,
                           OpType opType) {
  if (!isArithmeticOp(opType)) {
    return ERR_NO_OP;
  }

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  TensorArg aArg = {0};
  TensorArg bArg = {0};
  Result res = materializeTensorOnContext(ctx, ops.a, true, &aArg);
  if (res != OK) {
    cleanupPaddedPair(ctx, a, b, &ops);
    return res;
  }
  res = materializeTensorOnContext(ctx, ops.b, true, &bArg);
  if (res != OK) {
    releaseTensorArg(ctx, &aArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return res;
  }

  Tensor *opA = aArg.tensor;
  Tensor *opB = bArg.tensor;
  if (!areTensorsSameShape(opA, opB)) {
    releaseTensorArg(ctx, &aArg);
    releaseTensorArg(ctx, &bArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return binaryOpViaCpuFallback(ctx, a, b, destination, opType);
  }

  Tensor *output = t_Zeros(ctx, opA->shape, opA->dtype);
  if (output == NULL) {
    releaseTensorArg(ctx, &aArg);
    releaseTensorArg(ctx, &bArg);
    cleanupPaddedPair(ctx, a, b, &ops);
    return ERR_OUT_OF_MEMORY;
  }
  res = runCudaBinaryOp(ctx, opA->dtype, opType, opA->values, opB->values, output->values,
                        output->size);
  if (res == OK) {
    *destination = *output;
    freeAlloc(ctx->memory, output);
  } else {
    FreeTensor(ctx, output);
  }

  releaseTensorArg(ctx, &aArg);
  releaseTensorArg(ctx, &bArg);
  cleanupPaddedPair(ctx, a, b, &ops);

  return res;
}

static Result binaryOp(Context *ctx, Tensor *a, Tensor *b, Tensor *destination, OpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  switch (ctx != NULL && ctx->device != NULL ? ctx->device->type : CPU) {
    case CUDA: return binaryOpCuda(ctx, a, b, destination, opType);
    case CPU:
    default: return binaryOpCpu(ctx, a, b, destination, opType);
  }
}

Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_ADD);
}

static Result inPlaceBinopCpu(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *opB = b;
  Tensor *paddedB = NULL;
  if (a->shape.numOfDims != b->shape.numOfDims) {
    TensorPair ops = padSmallerTensor(ctx, a, b);
    opB = ops.b;
    paddedB = ops.b;
  }

  TensorArg bArg = {0};
  Result res = materializeTensorOnContext(ctx, opB, true, &bArg);
  if (res != OK) {
    if (paddedB != NULL) {
      FreeViewTensor(ctx, paddedB);
    }
    return res;
  }
  opB = bArg.tensor;

  if (a->isContigous && opB->isContigous && areTensorsSameShape(a, opB)) {
    res = straightInPlaceBinop(a, opB, opType);
    releaseTensorArg(ctx, &bArg);
    if (paddedB != NULL) {
      FreeViewTensor(ctx, paddedB);
    }
    return res;
  }

  dim_t currentCoord[a->shape.numOfDims];
  dim_t bCoords[a->shape.numOfDims];

  for (tensor_size_t x = 0; x < a->size; x++) {
    // Unravel using a's logical shape (ignoring boundary).
    unravel_index(x, &a->shape, currentCoord);

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

    Value result;
    switch (opType) {
      case OP_ADD: VALUE_BINOP(result, aVal, bVal, +); break;
      case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
      case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;
      default:
        releaseTensorArg(ctx, &bArg);
        if (paddedB != NULL) {
          FreeViewTensor(ctx, paddedB);
        }
        return ERR_NOT_A_BINOP;
    }

    VALUE_SET(a->values, aStorageIdx, result);
  }

  releaseTensorArg(ctx, &bArg);

  if (paddedB != NULL) {
    FreeViewTensor(ctx, paddedB);
  }

  return OK;
}

static Result inPlaceBinopCuda(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  if (!isArithmeticOp(opType) || !a->isContigous || a->isView || !isSameContext(a->context, ctx)) {
    return ERR_NO_OP;
  }

  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  Tensor *opB = b;
  Tensor *paddedB = NULL;
  if (a->shape.numOfDims != b->shape.numOfDims) {
    TensorPair ops = padSmallerTensor(ctx, a, b);
    opB = ops.b;
    paddedB = ops.b;
  }

  TensorArg bArg = {0};
  Result res = materializeTensorOnContext(ctx, opB, true, &bArg);
  if (res != OK) {
    if (paddedB != NULL) {
      FreeViewTensor(ctx, paddedB);
    }
    return res;
  }
  opB = bArg.tensor;

  if (!areTensorsSameShape(a, opB)) {
    releaseTensorArg(ctx, &bArg);
    if (paddedB != NULL) {
      FreeViewTensor(ctx, paddedB);
    }
    return ERR_NO_OP;
  }

  res = runCudaBinaryOp(ctx, a->dtype, opType, a->values, opB->values, a->values, a->size);

  releaseTensorArg(ctx, &bArg);
  if (paddedB != NULL) {
    FreeViewTensor(ctx, paddedB);
  }

  return res;
}

static Result inPlaceBinopViaCpuFallback(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  Context cpuCtx = {.memory = ctx->memory};
  TensorArg aArg = {0};
  Result res = materializeTensorOnContext(&cpuCtx, a, true, &aArg);
  if (res != OK) {
    return res;
  }

  res = inPlaceBinopCpu(&cpuCtx, aArg.tensor, b, opType);
  if (res == OK) {
    res = copyContiguousTensorIntoTensor(&cpuCtx, aArg.tensor, a);
  }

  releaseTensorArg(&cpuCtx, &aArg);
  return res;
}

static Result inPlaceBinop(Context *ctx, Tensor *a, Tensor *b, OpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  switch (ctx != NULL && ctx->device != NULL ? ctx->device->type : CPU) {
    case CUDA:
      if (!areTensorsSameShape(a, b) || a->shape.numOfDims != b->shape.numOfDims) {
        return inPlaceBinopViaCpuFallback(ctx, a, b, opType);
      }
      return inPlaceBinopCuda(ctx, a, b, opType);
    case CPU:
    default: return inPlaceBinopCpu(ctx, a, b, opType);
  }
}

Result AddInPlace(Context *ctx, Tensor *a, Tensor *b) {
  return inPlaceBinop(ctx, a, b, OP_ADD);
}

Result SubtractInPlace(Context *ctx, Tensor *a, Tensor *b) {
  return inPlaceBinop(ctx, a, b, OP_SUBTRACT);
}

Result MultiplyInPlace(Context *ctx, Tensor *a, Tensor *b) {
  return inPlaceBinop(ctx, a, b, OP_MULTIPLY);
}

Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_SUBTRACT);
}

Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_MULTIPLY);
}

Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination) {
  // Implement division as: numerator / denominator = numerator * (denominator^-1)
  // This automatically gets correct gradients through the computation graph!

  // Pow writes into dest by assigning a newly allocated tensor payload. Allocate only the
  // container struct here so we don't leak a preallocated payload on overwrite.
  Tensor *denom_inv = allocate(ctx->memory, sizeof(Tensor));
  Result allocRes = ensureAllocated(denom_inv);
  if (allocRes != OK) {
    return allocRes;
  }
  Result res = Pow(ctx, denominator, -1.0f, denom_inv);
  if (res != OK) {
    freeAlloc(ctx->memory, denom_inv);
    return res;
  }

  // Compute numerator * denominator^-1
  res = Multiply(ctx, numerator, denom_inv, destination);
  FreeTensor(ctx, denom_inv);
  return res;
}

Result GreaterThan(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_GREATER);
}

Result GreaterThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_GREATER_OR_EQUAL);
}

Result LessThan(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_LESS);
}

Result LessThanOrEqual(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_LESS_OR_EQUAL);
}
