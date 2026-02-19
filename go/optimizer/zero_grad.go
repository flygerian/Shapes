package optimizer

import (
	shapes "github.com/flygerian/shapes"
)

func ZeroGrad(ctx *shapes.Context, cg *shapes.ComputationGraph) {
	for _, node := range cg.Nodes {
		node.Grad = shapes.Zeros(ctx, shapes.ShapeOf(node.Grad))
	}
}
