#ifndef shapes_activation_h
#define shapes_activation_h

#include "common.h"
#include "result/result.h"


#define COMPUTE_TANH(val, dtype_enum, c_type, exp_fn)                                              \
  case dtype_enum: {                                                                               \
    c_type num = (val)->as.c_type;                                                                 \
    (val)->as.c_type = (exp_fn(num * 2) - 1) / (exp_fn(2 * num) + 1);                              \
    return OK;                                                                                     \
  }

Result Tanh(Context *ctx, Tensor *t, Tensor *dest);

#endif
