#include "common.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor/value.h"
#include "tensor_internal.h"
#include <string.h>

// dtypeRank returns a numeric rank representing the width of a dtype.
// Higher rank means wider type. Types within the same sign family
// can only be cast from lower rank to higher rank (no truncation).
//
// Float family:  F16=0, F32=1, F64=2
// Signed family: I8=0, I16=1, I32=2, I64=3
// Unsigned family: U8=0, U16=1, U32=2, U64=3
// Bool family: BOOL=0
static int dtypeRank(Dtype d) {
  switch (d) {
    case F16: return 0;
    case F32: return 1;
    case F64: return 2;
    case I8: return 0;
    case I16: return 1;
    case I32: return 2;
    case I64: return 3;
    case U8: return 0;
    case U16: return 1;
    case U32: return 2;
    case U64: return 3;
    case BOOL: return 0;
    default: return -1;
  }
}

// dtypeFamily returns:
//   0 = float (F16, F32, F64)
//   1 = signed int (I8, I16, I32, I64)
//   2 = unsigned int (U8, U16, U32, U64)
//   3 = bool (BOOL)
//  -1 = unknown
static int dtypeFamily(Dtype d) {
  switch (d) {
    case F16:
    case F32:
    case F64: return 0;
    case I8:
    case I16:
    case I32:
    case I64: return 1;
    case U8:
    case U16:
    case U32:
    case U64: return 2;
    case BOOL: return 3;
    default: return -1;
  }
}

// castValue reads a Value in the source dtype and writes it into the
// destination dtype. This performs the actual numeric conversion.
static Value castValue(Value src, Dtype target) {
  // First widen the source to the largest type in its family,
  // then convert to the target type.
  double asDouble;
  switch (src.dtype) {
    case F16: asDouble = (double)src.as.f16; break;
    case F32: asDouble = (double)src.as.f32; break;
    case F64: asDouble = src.as.f64; break;
    case I8: asDouble = (double)src.as.i8; break;
    case I16: asDouble = (double)src.as.i16; break;
    case I32: asDouble = (double)src.as.i32; break;
    case I64: asDouble = (double)src.as.i64; break;
    case U8: asDouble = (double)src.as.u8; break;
    case U16: asDouble = (double)src.as.u16; break;
    case U32: asDouble = (double)src.as.u32; break;
    case U64: asDouble = (double)src.as.u64; break;
    case BOOL: asDouble = src.as.boolean ? 1.0 : 0.0; break;
    default: asDouble = 0.0; break;
  }

  Value dest;
  dest.dtype = target;
  switch (target) {
    case F16: dest.as.f16 = (f16)asDouble; break;
    case F32: dest.as.f32 = (f32)asDouble; break;
    case F64: dest.as.f64 = (f64)asDouble; break;
    case I8: dest.as.i8 = (i8)asDouble; break;
    case I16: dest.as.i16 = (i16)asDouble; break;
    case I32: dest.as.i32 = (i32)asDouble; break;
    case I64: dest.as.i64 = (i64)asDouble; break;
    case U8: dest.as.u8 = (u8)asDouble; break;
    case U16: dest.as.u16 = (u16)asDouble; break;
    case U32: dest.as.u32 = (u32)asDouble; break;
    case U64: dest.as.u64 = (u64)asDouble; break;
    case BOOL: dest.as.boolean = asDouble != 0.0; break;
    default: break;
  }
  return dest;
}

// isCastSafe checks whether casting from source dtype to target dtype is
// allowed. The rules are:
//
//   - Float <-> Signed int: always allowed in both directions (the user
//     accepts the semantic conversion, e.g. F32 -> I32 truncates decimals).
//   - Within the same family: only widening (target rank >= source rank).
//   - Unsigned: only within the unsigned family, widening only.
//   - Unsigned <-> Float or Signed: never allowed (sign mismatch).
static bool isCastSafe(Dtype source, Dtype target) {
  if (source == target) {
    return true;
  }
  if (target == BOOL) {
    return true;
  }
  if (source == BOOL) {
    return false;
  }

  int srcFamily = dtypeFamily(source);
  int tgtFamily = dtypeFamily(target);

  // Float <-> Signed int: always allowed
  if ((srcFamily == 0 && tgtFamily == 1) || (srcFamily == 1 && tgtFamily == 0)) {
    return true;
  }

  // Cross-family casts other than float<->signed are disallowed
  if (srcFamily != tgtFamily) {
    return false;
  }

  // Within the same family, target must be at least as wide
  return dtypeRank(target) >= dtypeRank(source);
}

Result Cast(Context *ctx, Tensor *source, Tensor *dest, Dtype target) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Dtype srcDtype = source->dtype;

  // Same dtype: just clone
  if (srcDtype == target) {
    return Clone(ctx, source, dest);
  }

  // Check cast safety (sign compatibility + truncation)
  int srcFamily = dtypeFamily(srcDtype);
  int tgtFamily = dtypeFamily(target);

  if (target == BOOL) {
    srcFamily = tgtFamily;
  }

  // Unsigned <-> anything else is a sign mismatch
  bool involvesUnsigned = (srcFamily == 2 || tgtFamily == 2);
  if (involvesUnsigned && srcFamily != tgtFamily) {
    return ERR_SIGN_MISMATCH_CAST;
  }

  // Check for truncation (within same family, narrowing is not allowed)
  if (!isCastSafe(srcDtype, target)) {
    return ERR_TRUNCATING_CAST;
  }

  // Ensure contiguous source
  Tensor *src = source;
  if (!source->isContigous) {
    src = copyToContiguous(ctx, source);
  }

  // Allocate destination shape
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * src->shape.numOfDims);
  memcpy(newDims, src->shape.dims, sizeof(dim_t) * src->shape.numOfDims);

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * src->shape.numOfDims);
  memcpy(newMultipliers, src->shape.multipliers, sizeof(multiplier_t) * src->shape.numOfDims);

  // Allocate destination values
  size_t valueBytes = getBytesForDtype(target) * src->size;
  void *newValues = allocate(ctx->memory, valueBytes);

  // Element-wise conversion
  for (tensor_size_t i = 0; i < src->size; i++) {
    Value v;
    VALUE_GET_FROM_ARR(src->values, i, &v, srcDtype);
    Value converted = castValue(v, target);
    VALUE_SET(newValues, i, converted);
  }

  *dest = (Tensor){
      .dtype = target,
      .values = newValues,
      .size = src->size,
      .isContigous = true,
      .isView = false,
      .boundary = NULL,
      .shape = {.dims = newDims, .numOfDims = src->shape.numOfDims, .multipliers = newMultipliers}};

  return OK;
}
