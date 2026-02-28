package optimizer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

func SGD(ctx shapes.Context, lr float32) func(shapes.ComputationGraph) {
	return func(cg shapes.ComputationGraph) {
		fusedCtx := ctx.NoGraph()
		defer fusedCtx.Finish()
		lrTensor := shapes.Float(fusedCtx, shapes.Shape{1}, lr)

		parameters := extract.Parameters(cg)
		for _, p := range parameters {
			fmt.Printf("adjusting tensor of shape %v\n", p.Shape())
			scaled := p.Grad().Times(fusedCtx, lrTensor)
			neg := scaled.Negate(fusedCtx)
			p.AddInPlace(fusedCtx, neg)
		}
	}
}
