package optimizer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

func SGD(ctx *shapes.Context, lr float32) func(*shapes.ComputationGraph) {
	lrTensor := shapes.Float(ctx, shapes.Shape{1}, lr)
	return func(cg *shapes.ComputationGraph) {
		fusedCtx := ctx.Fused()
		defer fusedCtx.Close()

		parameters := extract.Parameters(cg)
		for _, p := range parameters {
			scaled := p.Grad().Times(fusedCtx, lrTensor)
			neg := scaled.Negate(fusedCtx)
			p.AddInPlace(fusedCtx, neg)
		}
	}
}
