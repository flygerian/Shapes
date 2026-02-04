#ifndef shapes_value_h
#define shapes_value_h

#include <stddef.h>

#define VALUE_SET(arr, idx, v) do { \
  switch ((v).dtype) { \
    case U8:  ((u8*)(arr))[(idx)]     = (v).as.u8;  break; \
    case U16: ((u16*)(arr))[(idx)]    = (v).as.u16; break; \
    case U32: ((u32*)(arr))[(idx)]    = (v).as.u32; break; \
    case U64: ((u64*)(arr))[(idx)]    = (v).as.u64; break; \
    case F16: ((float*)(arr))[(idx)]  = (v).as.f16; break; \
    case F32: ((float*)(arr))[(idx)]  = (v).as.f32; break; \
    case F64: ((double*)(arr))[(idx)] = (v).as.f64; break; \
  } \
} while(0)

#define VALUE_GET_FROM_ARR(arr, idx, v, dt) do { \
  (v)->dtype = (dt); \
  switch ((dt)) { \
    case U8:  (v)->as.u8  = ((u8*)(arr))[(idx)];     break; \
    case U16: (v)->as.u16 = ((u16*)(arr))[(idx)];    break; \
    case U32: (v)->as.u32 = ((u32*)(arr))[(idx)];    break; \
    case U64: (v)->as.u64 = ((u64*)(arr))[(idx)];    break; \
    case F16: (v)->as.f16 = ((float*)(arr))[(idx)];  break; \
    case F32: (v)->as.f32 = ((float*)(arr))[(idx)];  break; \
    case F64: (v)->as.f64 = ((double*)(arr))[(idx)]; break; \
  } \
} while(0)

#define VALUE_BINOP(dest, a, b, op) do { \
  switch ((a).dtype) { \
    case U8:  (dest).as.u8  = (a).as.u8  op (b).as.u8;  break; \
    case U16: (dest).as.u16 = (a).as.u16 op (b).as.u16; break; \
    case U32: (dest).as.u32 = (a).as.u32 op (b).as.u32; break; \
    case U64: (dest).as.u64 = (a).as.u64 op (b).as.u64; break; \
    case F16: (dest).as.f16 = (a).as.f16 op (b).as.f16; break; \
    case F32: (dest).as.f32 = (a).as.f32 op (b).as.f32; break; \
    case F64: (dest).as.f64 = (a).as.f64 op (b).as.f64; break; \
  } \
  (dest).dtype = (a).dtype; \
} while(0)

#define VALUE_UNBOX(v, dest) do { \
  switch ((v).dtype) { \
    case U8:  *((u8*)(dest))     = (v).as.u8;  break; \
    case U16: *((u16*)(dest))    = (v).as.u16; break; \
    case U32: *((u32*)(dest))    = (v).as.u32; break; \
    case U64: *((u64*)(dest))    = (v).as.u64; break; \
    case F16: *((float*)(dest))  = (v).as.f16; break; \
    case F32: *((float*)(dest))  = (v).as.f32; break; \
    case F64: *((double*)(dest)) = (v).as.f64; break; \
  } \
} while(0)

#define VALUE(type, data) \
  ((type) == U8  ? (Value){.dtype = (type), .as.u8  = (u8) (data)} : \
   (type) == U16 ? (Value){.dtype = (type), .as.u16 = (u16) (data)} : \
   (type) == U32 ? (Value){.dtype = (type), .as.u32 = (u32) (data)} : \
   (type) == U64 ? (Value){.dtype = (type), .as.u64 = (u64) (data)} : \
   (type) == F16 ? (Value){.dtype = (type), .as.f16 = (float) (data)} : \
   (type) == F32 ? (Value){.dtype = (type), .as.f32 = (float)(data)} : \
                   (Value){.dtype = (type), .as.f64 = (double) (data)})

#endif
