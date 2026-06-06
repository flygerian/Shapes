#include "stdbool.h"
#include "types.h"

#if defined(__GNUC__) || defined(__clang__)
#define SHAPES_PRAGMA_SIMD _Pragma("GCC ivdep")
#else
#define SHAPES_PRAGMA_SIMD
#endif

#define STRAIGHT_CMP_LOOP(TYPE, op)                                                                                                                                                                    \
  do {                                                                                                                                                                                                 \
    TYPE *pa = a->values;                                                                                                                                                                              \
    TYPE *pb = b->values;                                                                                                                                                                              \
    bool *po = dest->values;                                                                                                                                                                           \
    for (shapes_tensor_size_t i = 0; i < dest->size; i++) {                                                                                                                                                   \
      po[i] = pa[i] op pb[i];                                                                                                                                                                          \
    }                                                                                                                                                                                                  \
    return OK;                                                                                                                                                                                         \
  } while (0)

#define SWITCH_ARITH_OP(OP_TYPE, ADD_EXPR, SUB_EXPR, MUL_EXPR)                                                                                                                                         \
  switch (OP_TYPE) {                                                                                                                                                                                   \
    case OP_ADD: ADD_EXPR; break;                                                                                                                                                                      \
    case OP_SUBTRACT: SUB_EXPR; break;                                                                                                                                                                 \
    case OP_MULTIPLY: MUL_EXPR; break;                                                                                                                                                                 \
    default: return ERR_NOT_A_BINOP;                                                                                                                                                                   \
  }

#define DEFINE_ARITH_HELPERS(TYPE, NAME)                                                                                                                                                               \
  static inline void add_##NAME(const void *restrict a, const void *restrict b, void *restrict out, shapes_tensor_size_t n) {                                                                                 \
    SHAPES_PRAGMA_SIMD                                                                                                                                                                                 \
    for (shapes_tensor_size_t i = 0; i < n; i++) {                                                                                                                                                            \
      ((TYPE *)out)[i] = ((TYPE *)a)[i] + ((TYPE *)b)[i];                                                                                                                                              \
    }                                                                                                                                                                                                  \
  }                                                                                                                                                                                                    \
  static inline void subtract_##NAME(const void *restrict a, const void *restrict b, void *restrict out, shapes_tensor_size_t n) {                                                                            \
    SHAPES_PRAGMA_SIMD                                                                                                                                                                                 \
    for (shapes_tensor_size_t i = 0; i < n; i++) {                                                                                                                                                            \
      ((TYPE *)out)[i] = ((TYPE *)a)[i] - ((TYPE *)b)[i];                                                                                                                                              \
    }                                                                                                                                                                                                  \
  }                                                                                                                                                                                                    \
  static inline void multiply_##NAME(const void *restrict a, const void *restrict b, void *restrict out, shapes_tensor_size_t n) {                                                                            \
    SHAPES_PRAGMA_SIMD                                                                                                                                                                                 \
    for (shapes_tensor_size_t i = 0; i < n; i++) {                                                                                                                                                            \
      ((TYPE *)out)[i] = ((TYPE *)a)[i] * ((TYPE *)b)[i];                                                                                                                                              \
    }                                                                                                                                                                                                  \
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

typedef void (*binopFn)(const void *restrict a, const void *restrict b, void *restrict out, shapes_tensor_size_t n);

typedef struct binop {
  binopFn U8;
  binopFn U16;
  binopFn U32;
  binopFn U64;
  binopFn I8;
  binopFn I16;
  binopFn I32;
  binopFn I64;
  binopFn F32;
  binopFn F64;
} binop;

static inline binopFn getBinopFn(binop op, shapes_Dtype dt) {
  switch (dt) {
    case U8: return op.U8;
    case U16: return op.U16;
    case U32: return op.U32;
    case U64: return op.U64;
    case I8: return op.I8;
    case I16: return op.I16;
    case I32: return op.I32;
    case I64: return op.I64;
    case F32: return op.F32;
    case F64: return op.F64;
    default: return NULL;
  }
}

static binop binopTable[] = {
    [OP_ADD] = (binop){.U8 = add_u8, .U16 = add_u16, .U32 = add_u32, .U64 = add_u64, .I8 = add_i8, .I16 = add_i16, .I32 = add_i32, .I64 = add_i64, .F32 = add_f32, .F64 = add_f64},
    [OP_SUBTRACT] = (binop){.U8 = subtract_u8,
                            .U16 = subtract_u16,
                            .U32 = subtract_u32,
                            .U64 = subtract_u64,
                            .I8 = subtract_i8,
                            .I16 = subtract_i16,
                            .I32 = subtract_i32,
                            .I64 = subtract_i64,
                            .F32 = subtract_f32,
                            .F64 = subtract_f64},
    [OP_MULTIPLY] = (binop){.U8 = multiply_u8,
                            .U16 = multiply_u16,
                            .U32 = multiply_u32,
                            .U64 = multiply_u64,
                            .I8 = multiply_i8,
                            .I16 = multiply_i16,
                            .I32 = multiply_i32,
                            .I64 = multiply_i64,
                            .F32 = multiply_f32,
                            .F64 = multiply_f64},
};
