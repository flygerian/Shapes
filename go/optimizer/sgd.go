package optimizer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

// SGD creates an SGD optimizer.
// It uses the optimized C implementation via shapes.Sgd.
func SGD(ctx shapes.Context, lr float32) func(shapes.ComputationGraph) {
	return func(cg shapes.ComputationGraph) {
		noGraphCtx := ctx.NoGraph()
		defer noGraphCtx.Finish()

		parameters := extract.Parameters(cg)
		if len(parameters) == 0 {
			return
		}

		grads := make([]shapes.Tensor, len(parameters))
		for i, p := range parameters {
			grad := p.Grad()
			if grad == nil {
				panic("shapes: parameter gradient is nil")
			}
			grads[i] = grad.(shapes.Tensor)
		}

		shapes.Sgd(noGraphCtx, parameters, grads, lr)
	}
}
