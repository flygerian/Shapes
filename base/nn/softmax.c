#include "shapes.h"
#include "types.h"

Tensor shapesnn_Softmax(Context *ctx, Tensor *logits) {
  u8 ndims = logits->shape.numOfDims;
  u8 classDim = ndims - 1;

  Tensor maxLogits = shapes_Max(ctx, logits, classDim);
  Tensor shifted = shapes_Subtract(ctx, logits, &maxLogits);
  Tensor exp = shapes_Exp(ctx, &shifted);
  Tensor sumExp = shapes_Sum(ctx, &exp, classDim);
  Tensor probs = shapes_Divide(ctx, &exp, &sumExp);

  return probs;
}
