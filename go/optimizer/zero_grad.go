package optimizer

import (
	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func ZeroGrad(ctx *shapes.Context, cg *tensor.ComputationGraph) {
	for _, node := range cg.Nodes {
		node.Grad = tensor.Zeros(ctx, tensor.ShapeOf(node.Grad))
	}
}
