#include "common.h"
#include "grad/grad.h"
#include "result/result.h"
#include "tensor_internal.h"
#include "unary.h"
#include "value.h"
#include <math.h>
#include <stddef.h>

Result powValue(Value *v, f32 power) {
  switch (v->dtype) {
    COMPUTE_POW(v, power, F16, f16, pow);
    COMPUTE_POW(v, power, F32, f32, pow);
    COMPUTE_POW(v, power, F64, f64, pow);

    default: return ERR_POW_VALUE_NOT_FLOAT;
  }
}

Result Pow(Context *ctx, Tensor *t, f32 power, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != F16 && t->dtype != F32 && t->dtype != F64) {
    return ERR_POW_VALUE_NOT_FLOAT;
  }

  Tensor *output = t_Zeros(ctx, t->shape, t->dtype);

  for (size_t i = 0; i < t->size; i++) {
    Value val;
    VALUE_GET_FROM_ARR(t->values, i, &val, t->dtype);

    Result result = powValue(&val, power);
    if (result != OK) {
      return result;
    }

    VALUE_SET(output->values, i, val);
  }

  *dest = *output;

  if (ctx->grad) {
    ConstructPowBackwardpass(ctx, t, power, dest);
  }

  return OK;
}
