#include "shapes.h"
#include "types.h"

Tensor *nn_Softmax(Context *ctx, Tensor *logits) {
  u8 ndims = logits->shape.numOfDims;
  u8 classDim = ndims - 1;

  Tensor *maxLogits = Max(ctx, logits, classDim);
  Tensor *shifted = Subtract(ctx, logits, maxLogits);
  Tensor *exp = Exp(ctx, shifted);
  Tensor *probs = Divide(ctx, exp, Sum(ctx, exp, classDim));

  return probs;
}
