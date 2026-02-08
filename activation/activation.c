#include "common.h"
#include "grad/grad.h"
#include "result/result.h"
#include "tensor/tensor_internal.h"
#include "tensor/value.h"
#include <math.h>
#include <stddef.h>
#include "activation.h"

Result tanhValue(Value *v) {
  switch (v->dtype) {
    COMPUTE_TANH(v, F16, f16, expf);
    COMPUTE_TANH(v, F32, f32, expf);
    COMPUTE_TANH(v, F64, f64, exp);

    default: return ERR_TANH_VALUE_NOT_FLOAT;
  }
}

Result Tanh(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_TANH_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = tanhValue(&val);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;

  if (ctx->grad) {
    ConstructTanhBackwardpass(ctx, t, dest);
  }

  return OK;
}
