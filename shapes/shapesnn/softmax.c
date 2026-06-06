#include "shapes.h"
#include "types.h"

shapes_Tensor shapesnn_Softmax(shapes_Context *ctx, shapes_Tensor *logits) {
  u8 ndims = logits->shape.numOfDims;
  u8 classDim = ndims - 1;

  shapes_Tensor maxLogits = shapes_Max(ctx, logits, classDim);
  shapes_Tensor shifted = shapes_Subtract(ctx, logits, &maxLogits);
  shapes_Tensor exp = shapes_Exp(ctx, &shifted);
  shapes_Tensor sumExp = shapes_Sum(ctx, &exp, classDim);
  shapes_Tensor probs = shapes_Divide(ctx, &exp, &sumExp);

  return probs;
}
