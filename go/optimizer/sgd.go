package optimizer

import (
	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
	"github.com/flygerian/shapes/tensor"
)

func SGD(ctx *shapes.Context, lr float32) func(*tensor.ComputationGraph) {
	lrTensor := tensor.Float(ctx, tensor.Shape{1}, lr)
	return func(cg *tensor.ComputationGraph) {
		parameters := extract.Parameters(cg)
		for _, p := range parameters {
			scaled := p.Grad().Times(ctx, lrTensor)
			neg := scaled.Negate(ctx)
			p.AddInPlace(ctx, neg)
			scaled.Free(ctx)
			neg.Free(ctx)
		}
	}
}
