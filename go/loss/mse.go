package loss

import (
	shapes "github.com/flygerian/shapes"
)

func Mse(ctx *shapes.Context, yGround *shapes.Tensor, yPred *shapes.Tensor) *shapes.Tensor {
	se := yPred.Minus(ctx, yGround).Pow(ctx, 2)
	result := se
	shape := shapes.ShapeOf(result)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] > 1 {
			result = result.Sum(ctx, uint32(i))
		}
	}
	return result
}
