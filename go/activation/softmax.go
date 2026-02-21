package activation

import "github.com/flygerian/shapes"

func Softmax(t *shapes.WrappedTensor) *shapes.WrappedTensor {
	fusedCtx := t.Context().Fused()

	exp := t.Tensor().Minus(fusedCtx, t.Tensor().Max(fusedCtx)).Exp(fusedCtx)
	sm := exp.Divide(fusedCtx, exp.Sum(fusedCtx, 1))

	return fusedCtx.Wrap(sm)
}
