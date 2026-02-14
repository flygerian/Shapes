package activation

import (
	"github.com/flygerian/shapes/tensor"
)

// Tanh applies the hyperbolic tangent activation element-wise, returning a new tensor.
// tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
func Tanh(t *tensor.Tensor) *tensor.Tensor {
	ctx := t.Context()
	noGrad := ctx.NoGrad()

	two := tensor.Float(noGrad, tensor.ShapeOf(t), 2.0)
	ones := tensor.Float(noGrad, tensor.ShapeOf(t), 1.0)

	exp2x := t.Times(two).Exp()
	out := exp2x.Minus(ones).Divide(exp2x.Plus(ones))

	if ctx.GradEnabled() {
		tensor.AttachComputationGraphNode(out, tanhBackward, t)
	}
	return out
}

// tanhBackward computes the gradient for tanh.
// d(tanh(x))/dx = 1 - tanh(x)^2
// grad_input += grad_output * (1 - output^2)
func tanhBackward(node *tensor.ComputationGraphNode) {
	input := node.Inputs[0]
	output := node.Output
	ctx := input.Context().NoGrad()

	ones := tensor.Float(ctx, tensor.ShapeOf(output), 1.0)
	outputSquared := output.Times(output)
	localGrad := ones.Minus(outputSquared)
	gradInput := node.Grad.Times(localGrad)

	input.Computation.Grad = input.Computation.Grad.Plus(gradInput)
}
