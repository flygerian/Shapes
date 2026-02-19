package activation

import (
	shapes "github.com/flygerian/shapes"
)

// Tanh applies the hyperbolic tangent activation element-wise, returning a new tensor.
// tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
func Tanh(ctx *shapes.Context, t *shapes.Tensor) *shapes.Tensor {
	noGrad := ctx.NoGrad()

	two := shapes.Float(noGrad, shapes.ShapeOf(t), 2.0)
	ones := shapes.Float(noGrad, shapes.ShapeOf(t), 1.0)

	exp2x := t.Times(noGrad, two).Exp(noGrad)
	out := exp2x.Minus(noGrad, ones).Divide(noGrad, exp2x.Plus(noGrad, ones))

	if ctx.BackwardEnabled {
		shapes.AttachComputationGraphNode(ctx, out, shapes.OpTanh, tanhBackward, t)
	}
	return out
}

// tanhBackward computes the gradient for tanh.
// d(tanh(x))/dx = 1 - tanh(x)^2
// grad_input += grad_output * (1 - output^2)
func tanhBackward(ctx *shapes.Context, node *shapes.ComputationGraphNode) {
	input := node.Inputs[0]
	output := node.Output

	ones := shapes.Float(ctx, shapes.ShapeOf(output), 1.0)
	outputSquared := output.Times(ctx, output)
	localGrad := ones.Minus(ctx, outputSquared)
	gradInput := node.Grad.Times(ctx, localGrad)

	input.Computation.Grad = input.Grad().Plus(ctx, gradInput)
}
