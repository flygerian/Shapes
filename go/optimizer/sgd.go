package optimizer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

func SGD(ctx *shapes.Context, lr float32) func(*shapes.ComputationGraph) {
	lrTensor := shapes.Float(ctx, shapes.Shape{1}, lr)
	return func(cg *shapes.ComputationGraph) {
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
