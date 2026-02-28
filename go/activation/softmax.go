package activation

import "github.com/flygerian/shapes"

func Softmax(ctx shapes.Context, logits shapes.Tensor) shapes.Tensor {
	ndims := uint32(len(logits.Shape()))
	classDim := ndims - 1

	maxLogits := logits.Max(ctx, classDim)
	shifted := logits.Minus(ctx, maxLogits)
	exp := shifted.Exp(ctx)
	probs := exp.Divide(ctx, exp.Sum(ctx, classDim))

	return probs
}
