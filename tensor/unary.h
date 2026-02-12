#ifndef shapes_unary_h
#define shapes_unary_h

#include "common.h"
#include "result/result.h"

#define COMPUTE_POW(val, power, dtype_enum, c_type, pow_fn)                                       \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (c_type)pow_fn((double)num, (double)(power));                              \
    return OK;                                                                                     \
  }

Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest);

#endif
