package optimizer

import (
	"math"

	"github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

/*
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t   # 1st moment: mean of gradients
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²  # 2nd moment: mean of squared gradients
*/

type adam struct {
	m map[uintptr]shapes.Tensor
	v map[uintptr]shapes.Tensor

	b1   float64
	b2   float64
	step int

	a       float64
	epsilon float64
}

func Adam(ctx shapes.Context) func(shapes.ComputationGraph) {
	state := adam{
		m:    make(map[uintptr]shapes.Tensor),
		v:    make(map[uintptr]shapes.Tensor),
		b1:   0.9,
		b2:   0.999,
		step: 0,

		a:       1e-3,
		epsilon: 1e-8,
	}

	return func(cg shapes.ComputationGraph) {
		noGraphCtx := ctx.NoGraph()
		defer noGraphCtx.Finish()

		state.step += 1

		parameters := extract.Parameters(cg)
		if len(parameters) == 0 {
			return
		}

		// Lazily initialize per-parameter state; this also supports
		// reusing the same optimizer across different parameter sets.
		for _, p := range parameters {
			idx := uintptr(p.UnsafeCTensor())
			if _, ok := state.m[idx]; !ok {
				state.m[idx] = shapes.Float(ctx, p.Shape(), 0)
			}
			if _, ok := state.v[idx]; !ok {
				state.v[idx] = shapes.Float(ctx, p.Shape(), 0)
			}
		}

		for _, p := range parameters {
			go func() {
				idx := uintptr(p.UnsafeCTensor())
				m := state.m[idx]
				v := state.v[idx]

				// Keep optimizer state tensors stable across epochs by updating in-place.
				// Rebinding m/v to tensors created in a short-lived subcontext can leave
				// dangling pointers after epoch sweep.
				m.SubtractInPlace(noGraphCtx, m.Times(noGraphCtx, 1-state.b1))
				m.AddInPlace(noGraphCtx, p.Grad().Times(noGraphCtx, 1-state.b1))

				v.SubtractInPlace(noGraphCtx, v.Times(noGraphCtx, 1-state.b2))
				v.AddInPlace(noGraphCtx, p.Grad().Pow(noGraphCtx, 2).Times(noGraphCtx, 1-state.b2))

				mHat := m.Divide(noGraphCtx, (1 - math.Pow(float64(state.b1), float64(state.step))))
				vHat := v.Divide(noGraphCtx, (1 - math.Pow(float64(state.b2), float64(state.step))))

				update := mHat.Times(noGraphCtx, state.a).Divide(noGraphCtx, vHat.Plus(noGraphCtx, state.epsilon).Pow(noGraphCtx, 0.5))

				p.SubtractInPlace(noGraphCtx, update)
			}()
		}

	}
}
