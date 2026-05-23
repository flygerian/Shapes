#ifndef shapes_value_h
#define shapes_value_h

#include "common.h"
#include "types.h"
#include <stddef.h>
#include <stdio.h>

#define VALUE_SET(arr, idx, v)                                                                     \
  do {                                                                                             \
    switch ((v).dtype) {                                                                           \
      case BOOL: ((bool *)(arr))[(idx)] = (v).as.boolean; break;                                   \
      case U8: ((u8 *)(arr))[(idx)] = (v).as.u8; break;                                            \
      case U16: ((u16 *)(arr))[(idx)] = (v).as.u16; break;                                         \
      case U32: ((u32 *)(arr))[(idx)] = (v).as.u32; break;                                         \
      case U64: ((u64 *)(arr))[(idx)] = (v).as.u64; break;                                         \
      case I8: ((i8 *)(arr))[(idx)] = (v).as.i8; break;                                            \
      case I16: ((i16 *)(arr))[(idx)] = (v).as.i16; break;                                         \
      case I32: ((i32 *)(arr))[(idx)] = (v).as.i32; break;                                         \
      case I64: ((i64 *)(arr))[(idx)] = (v).as.i64; break;                                         \
      case F16: ((float *)(arr))[(idx)] = (v).as.f16; break;                                       \
      case F32: ((float *)(arr))[(idx)] = (v).as.f32; break;                                       \
      case F64: ((double *)(arr))[(idx)] = (v).as.f64; break;                                      \
    }                                                                                              \
  } while (0)

#define VALUE_GET_FROM_ARR(arr, idx, v, dt)                                                        \
  do {                                                                                             \
    (v)->dtype = (dt);                                                                             \
    switch ((dt)) {                                                                                \
      case BOOL: (v)->as.boolean = ((bool *)(arr))[(idx)]; break;                                  \
      case U8: (v)->as.u8 = ((u8 *)(arr))[(idx)]; break;                                           \
      case U16: (v)->as.u16 = ((u16 *)(arr))[(idx)]; break;                                        \
      case U32: (v)->as.u32 = ((u32 *)(arr))[(idx)]; break;                                        \
      case U64: (v)->as.u64 = ((u64 *)(arr))[(idx)]; break;                                        \
      case I8: (v)->as.i8 = ((i8 *)(arr))[(idx)]; break;                                           \
      case I16: (v)->as.i16 = ((i16 *)(arr))[(idx)]; break;                                        \
      case I32: (v)->as.i32 = ((i32 *)(arr))[(idx)]; break;                                        \
      case I64: (v)->as.i64 = ((i64 *)(arr))[(idx)]; break;                                        \
      case F16: (v)->as.f16 = ((float *)(arr))[(idx)]; break;                                      \
      case F32: (v)->as.f32 = ((float *)(arr))[(idx)]; break;                                      \
      case F64: (v)->as.f64 = ((double *)(arr))[(idx)]; break;                                     \
    }                                                                                              \
  } while (0)

#define VALUE_BINOP(dest, a, b, op)                                                                \
  do {                                                                                             \
    switch ((a).dtype) {                                                                           \
      case BOOL: (dest).as.boolean = (a).as.boolean op(b).as.boolean; break;                       \
      case U8: (dest).as.u8 = (a).as.u8 op(b).as.u8; break;                                        \
      case U16: (dest).as.u16 = (a).as.u16 op(b).as.u16; break;                                    \
      case U32: (dest).as.u32 = (a).as.u32 op(b).as.u32; break;                                    \
      case U64: (dest).as.u64 = (a).as.u64 op(b).as.u64; break;                                    \
      case I8: (dest).as.i8 = (a).as.i8 op(b).as.i8; break;                                        \
      case I16: (dest).as.i16 = (a).as.i16 op(b).as.i16; break;                                    \
      case I32: (dest).as.i32 = (a).as.i32 op(b).as.i32; break;                                    \
      case I64: (dest).as.i64 = (a).as.i64 op(b).as.i64; break;                                    \
      case F16: (dest).as.f16 = (a).as.f16 op(b).as.f16; break;                                    \
      case F32: (dest).as.f32 = (a).as.f32 op(b).as.f32; break;                                    \
      case F64: (dest).as.f64 = (a).as.f64 op(b).as.f64; break;                                    \
    }                                                                                              \
    (dest).dtype = (a).dtype;                                                                      \
  } while (0)

#define VALUE_UNBOX(v, dest)                                                                       \
  do {                                                                                             \
    switch ((v).dtype) {                                                                           \
      case BOOL: *((bool *)(dest)) = (v).as.boolean; break;                                        \
      case U8: *((u8 *)(dest)) = (v).as.u8; break;                                                 \
      case U16: *((u16 *)(dest)) = (v).as.u16; break;                                              \
      case U32: *((u32 *)(dest)) = (v).as.u32; break;                                              \
      case U64: *((u64 *)(dest)) = (v).as.u64; break;                                              \
      case I8: *((i8 *)(dest)) = (v).as.i8; break;                                                 \
      case I16: *((i16 *)(dest)) = (v).as.i16; break;                                              \
      case I32: *((i32 *)(dest)) = (v).as.i32; break;                                              \
      case I64: *((i64 *)(dest)) = (v).as.i64; break;                                              \
      case F16: *((float *)(dest)) = (v).as.f16; break;                                            \
      case F32: *((float *)(dest)) = (v).as.f32; break;                                            \
      case F64: *((double *)(dest)) = (v).as.f64; break;                                           \
    }                                                                                              \
  } while (0)


#define VALUE(type, data)                                                                          \
  ((type) == BOOL  ? (Value){.dtype = (type), .as.boolean = (bool)(data)}                          \
   : (type) == U8  ? (Value){.dtype = (type), .as.u8 = (u8)(data)}                                 \
   : (type) == U16 ? (Value){.dtype = (type), .as.u16 = (u16)(data)}                               \
   : (type) == U32 ? (Value){.dtype = (type), .as.u32 = (u32)(data)}                               \
   : (type) == U64 ? (Value){.dtype = (type), .as.u64 = (u64)(data)}                               \
   : (type) == I8  ? (Value){.dtype = (type), .as.i8 = (i8)(data)}                                 \
   : (type) == I16 ? (Value){.dtype = (type), .as.i16 = (i16)(data)}                               \
   : (type) == I32 ? (Value){.dtype = (type), .as.i32 = (i32)(data)}                               \
   : (type) == I64 ? (Value){.dtype = (type), .as.i64 = (i64)(data)}                               \
   : (type) == F16 ? (Value){.dtype = (type), .as.f16 = (float)(data)}                             \
   : (type) == F32 ? (Value){.dtype = (type), .as.f32 = (float)(data)}                             \
                   : (Value){.dtype = (type), .as.f64 = (double)(data)})

#define VALUE_CMP(a, b, op, dtype)                                                                 \
  ((dtype) == BOOL  ? (a).as.boolean op(b).as.boolean                                              \
   : (dtype) == U8  ? (a).as.u8 op(b).as.u8                                                        \
   : (dtype) == U16 ? (a).as.u16 op(b).as.u16                                                      \
   : (dtype) == U32 ? (a).as.u32 op(b).as.u32                                                      \
   : (dtype) == U64 ? (a).as.u64 op(b).as.u64                                                      \
   : (dtype) == I8  ? (a).as.i8 op(b).as.i8                                                        \
   : (dtype) == I16 ? (a).as.i16 op(b).as.i16                                                      \
   : (dtype) == I32 ? (a).as.i32 op(b).as.i32                                                      \
   : (dtype) == I64 ? (a).as.i64 op(b).as.i64                                                      \
   : (dtype) == F16 ? (a).as.f16 op(b).as.f16                                                      \
   : (dtype) == F32 ? (a).as.f32 op(b).as.f32                                                      \
                    : (a).as.f64 op(b).as.f64)

#define PRINT_VALUE(v)                                                                             \
  do {                                                                                             \
    switch ((v).dtype) {                                                                           \
      case BOOL: printf("%s", (v).as.boolean ? "true" : "false"); break;                           \
      case U8: printf("%u", (unsigned int)(v).as.u8); break;                                       \
      case U16: printf("%u", (unsigned int)(v).as.u16); break;                                     \
      case U32: printf("%u", (unsigned int)(v).as.u32); break;                                     \
      case U64: printf("%llu", (unsigned long long)(v).as.u64); break;                             \
      case I8: printf("%d", (int)(v).as.i8); break;                                                \
      case I16: printf("%d", (int)(v).as.i16); break;                                              \
      case I32: printf("%d", (int)(v).as.i32); break;                                              \
      case I64: printf("%lld", (long long)(v).as.i64); break;                                      \
      case F16: printf("%f", (double)(v).as.f16); break;                                           \
      case F32: printf("%f", (double)(v).as.f32); break;                                           \
      case F64: printf("%f", (double)(v).as.f64); break;                                           \
    }                                                                                              \
  } while (0)

#define VALUE_TO_STRING(v, dest, size)                                                             \
  do {                                                                                             \
    switch ((v).dtype) {                                                                           \
      case BOOL: snprintf(dest, size, "%s", (v).as.boolean ? "true" : "false"); break;             \
      case U8: snprintf(dest, size, "%u", (unsigned int)(v).as.u8); break;                         \
      case U16: snprintf(dest, size, "%u", (unsigned int)(v).as.u16); break;                       \
      case U32: snprintf(dest, size, "%u", (unsigned int)(v).as.u32); break;                       \
      case U64: snprintf(dest, size, "%llu", (unsigned long long)(v).as.u64); break;               \
      case I8: snprintf(dest, size, "%d", (int)(v).as.i8); break;                                  \
      case I16: snprintf(dest, size, "%d", (int)(v).as.i16); break;                                \
      case I32: snprintf(dest, size, "%d", (int)(v).as.i32); break;                                \
      case I64: snprintf(dest, size, "%lld", (long long)(v).as.i64); break;                        \
      case F16: snprintf(dest, size, "%f", (double)(v).as.f16); break;                             \
      case F32: snprintf(dest, size, "%f", (double)(v).as.f32); break;                             \
      case F64: snprintf(dest, size, "%f", (double)(v).as.f64); break;                             \
    }                                                                                              \
  } while (0)

static dim_t idx_zero[] = {0};

#endif
