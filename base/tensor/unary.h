#ifndef shape_unary_h
#define hapes_unary_h

#include "common.h"
#include "result/result.h"

#define COMPUTE_POW(val, power, dtype_enum, c_type, pow_fn)                                        \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (c_type)pow_fn((double)num, (double)(power));                               \
    return OK;                                                                                     \
  }

#define COMPUTE_EXP(val, dtype_enum, c_type, exp_fn)                                               \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (c_type)exp_fn((double)num);                                                \
    return OK;                                                                                     \
  }

#define COMPUTE_ABS(val, dtype_enum, c_type, abs_fn)                                               \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (c_type)abs_fn(num);                                                \
    return OK;                                                                                     \
  }

#define COMPUTE_NEGATE(val, dtype_enum, c_type)                                                    \
  case dtype_enum: {                                                                               \
    (val)->as.c_type = -(val)->as.c_type;                                                          \
    return OK;                                                                                     \
  }

#define COMPUTE_LOG(val, dtype_enum, c_type, log_fn)                                               \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (c_type)log_fn((double)num);                                                \
    return OK;                                                                                     \
  }

#endif
