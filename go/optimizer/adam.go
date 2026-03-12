package optimizer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/extract"
)

type adamState struct {
	m       map[uintptr]shapes.Tensor
	v       map[uintptr]shapes.Tensor
	b1      float32
	b2      float32
	step    int
	a       float32
	epsilon float32
}

// Adam creates an Adam optimizer that uses the optimized C implementation.
// It accepts optional configuration functions to customize hyperparameters.
func Adam(ctx shapes.Context, opts ...AdamOption) func(shapes.ComputationGraph) {
	state := &adamState{
		m:       make(map[uintptr]shapes.Tensor),
		v:       make(map[uintptr]shapes.Tensor),
		b1:      0.9,
		b2:      0.999,
		step:    0,
		a:       1e-3,
		epsilon: 1e-8,
	}

	for _, opt := range opts {
		opt(state)
	}

	return func(cg shapes.ComputationGraph) {
		state.step++

		parameters := extract.Parameters(cg)
		if len(parameters) == 0 {
			return
		}

		// Lazily initialize per-parameter state in the main context
		// (not in a subcontext) so they survive context sweeps.
		for _, p := range parameters {
			idx := uintptr(p.UnsafeCTensor())
			if _, ok := state.m[idx]; !ok {
				state.m[idx] = shapes.Zeros(ctx, p.Shape())
			}
			if _, ok := state.v[idx]; !ok {
				state.v[idx] = shapes.Zeros(ctx, p.Shape())
			}
		}

		noGraphCtx := ctx.NoGraph()
		defer noGraphCtx.Finish()

		// Build AdamData array for C call
		adamData := make([]shapes.AdamData, len(parameters))

		for i, p := range parameters {
			idx := uintptr(p.UnsafeCTensor())
			grad := p.Grad()
			if grad == nil {
				panic("shapes: parameter gradient is nil")
			}

			adamData[i] = shapes.AdamData{
				Param:     p,
				ParamGrad: grad.(shapes.Tensor),
				M:         state.m[idx],
				V:         state.v[idx],
			}
		}

		shapes.Adam(noGraphCtx, adamData, state.b1, state.b2, state.step, state.a, state.epsilon)
	}
}

// AdamOption is a functional option for configuring the Adam optimizer.
type AdamOption func(*adamState)

// WithLearningRate sets the learning rate (alpha).
func WithLearningRate(lr float32) AdamOption {
	return func(s *adamState) {
		s.a = lr
	}
}

// WithBetas sets the beta1 and beta2 coefficients.
func WithBetas(beta1, beta2 float32) AdamOption {
	return func(s *adamState) {
		s.b1 = beta1
		s.b2 = beta2
	}
}

// WithEpsilon sets the epsilon value for numerical stability.
func WithEpsilon(eps float32) AdamOption {
	return func(s *adamState) {
		s.epsilon = eps
	}
}
