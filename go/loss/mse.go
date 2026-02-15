package loss

import (
	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func Mse(ctx *shapes.Context, yGround *tensor.Tensor, yPred *tensor.Tensor) *tensor.Tensor {
	se := yPred.Minus(ctx, yGround).Pow(ctx, 2)
	result := se
	shape := tensor.ShapeOf(result)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] > 1 {
			result = result.Sum(ctx, uint32(i))
		}
	}
	return result
}
