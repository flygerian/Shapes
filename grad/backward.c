#include "common.h"
#include "grad/grad.h"
#include "result/result.h"
#include "tensor/tensor.h"
#include "tensor/value.h"
#include <stdbool.h>
#include <stddef.h>
#include <time.h>

Result backward(Context *ctx, Tensor *t) {
  if (t->computation != NULL) {
    Result result = t->computation->backward(ctx, t->computation);
    if (result != OK) {
      return result;
    }

    for (size_t i = 0; i < t->computation->numInputs; i++) {
      backward(ctx, t->computation->inputs[i]);
    }
  }

  return OK;
}

Result Backward(Context *ctx, Tensor *t) {
  // Set the grad of the origin tensor to one
  SetValues(t->computation->grad, VALUE(t->dtype, 1.0));

  return backward(ctx, t);
}
