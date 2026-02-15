package loss

import (
	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func Mse(ctx *shapes.Context, yGround *tensor.Tensor, yPred *tensor.Tensor) *tensor.Tensor {
	return yPred.Minus(ctx, yGround).Pow(ctx, 2)
}
