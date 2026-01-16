#ifndef shapes_tensor_h
#define shapes_tensor_h

#include <stdint.h>
#include "../common.h"
#include "../result/result.h"

typedef enum  {
   F16, F32, F64, U8, U16, U32, U64 
}Dtype;

typedef struct {
  Dtype dtype;
  union {
    u8 u8;
    u16 u16;
    u32 u32;
    u64 u64;
    float f16;
    float f32;
    double f64;
  } as;
} Value;


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

#define VALUE_GET_FROM_ARR(arr, idx, v) do { \
  switch ((v)->dtype) { \
    case U8:  (v)->as.u8  = ((u8*)(arr))[(idx)];     break; \
    case U16: (v)->as.u16 = ((u16*)(arr))[(idx)];    break; \
    case U32: (v)->as.u32 = ((u32*)(arr))[(idx)];    break; \
    case U64: (v)->as.u64 = ((u64*)(arr))[(idx)];    break; \
    case F16: (v)->as.f16 = ((float*)(arr))[(idx)];  break; \
    case F32: (v)->as.f32 = ((float*)(arr))[(idx)];  break; \
    case F64: (v)->as.f64 = ((double*)(arr))[(idx)]; break; \
  } \
} while(0)

typedef struct {
  u32 *dims;
  u8 numOfDims;

  // Used to decide how much to move the idx on a values array
  u8 *multipliers;
} Dim;

typedef struct {
  Dtype dtype;
  void *values;
  Dim shape;
} Tensor;

Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination);
Result Subtract(Context *ctx, Tensor *a, Tensor *b,Tensor *destination);
Result Divide(Context *ctx, Tensor *numerator, Tensor *denominator, Tensor *destination);
Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination); 

Result GetAt(Tensor *t, Dim dim, Value *result);

Result AssignValue(Context *ctx, Tensor *t, Dim dim, Value value); 

// Tensor creation

Tensor T_Zeros(Context *ctx, Dim shape);

#endif
