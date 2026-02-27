package activation

import (
	shapes "github.com/flygerian/shapes"
)

// Tanh applies the hyperbolic tangent activation element-wise, returning a new tensor.
// tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
func Tanh(ctx *shapes.Context, t *shapes.Tensor) *shapes.Tensor {
	fusedCtx := ctx.Fused()

	two := shapes.Float(fusedCtx, shapes.ShapeOf(t), 2.0)
	ones := shapes.Float(fusedCtx, shapes.ShapeOf(t), 1.0)

	exp2x := t.Times(fusedCtx, two).Exp(fusedCtx)
	out := exp2x.Minus(fusedCtx, ones).Divide(fusedCtx, exp2x.Plus(fusedCtx, ones))

	if ctx.BackwardEnabled {
		// Save `out` as metadata so backward can reuse tanh(x) without recomputing it.
		ctx.NewComputationGraphNode(out, shapes.OpTanh, tanhBackward, []*shapes.Tensor{t}, []*shapes.Tensor{}, []*shapes.Tensor{out})
	}
	return out
}

// tanhBackward computes the gradient for tanh.
// d(tanh(x))/dx = 1 - tanh(x)^2
// grad_input += grad_output * (1 - output^2)
func tanhBackward(ctx *shapes.Context, node shapes.ComputationGraphNode) {
	input := node.Inputs()[0]
	output := node.Metadata()[0]

	ones := shapes.Float(ctx, shapes.ShapeOf(output), 1.0)
	outputSquared := output.Times(ctx, output)
	localGrad := ones.Minus(ctx, outputSquared)
	gradInput := node.Grad().(*shapes.Tensor).Times(ctx, localGrad)

	input.Grad().Accumulate(ctx, gradInput)
}
