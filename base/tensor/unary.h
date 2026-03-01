#ifndef shapes_unary_h
#define shapes_unary_h

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

Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest);
Result Exp(Context *ctx, Tensor *t, Tensor *dest);
Result Negate(Context *ctx, Tensor *t, Tensor *dest);

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

Result Mean(Context *ctx, Tensor *t, Tensor *dest);
Result MeanDim(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result Log(Context *ctx, Tensor *t, Tensor *dest);
Result Max(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);
Result ArgMax(Context *ctx, Tensor *t, Tensor *dest, dim_t dim);

#endif
