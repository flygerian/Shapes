package shapes

/*
#include "cwrappers.h"
*/
import "C"

// Relu applies rectified linear activation element-wise.
func Relu(ctx Context, t Tensor) Tensor {
	return t.Relu(ctx)
}

// Tanh applies the hyperbolic tangent activation element-wise.
func Tanh(ctx Context, t Tensor) Tensor {
	return t.Tanh(ctx)
}

// ReluBackward computes the gradient for relu.
// d(relu(x))/dx = 1 when x > 0, otherwise 0.
func ReluBackward(ctx Context, node ComputationGraphNode) {
	input := node.Inputs()[0]
	output := node.(Tensor)
	gradOut := node.Grad().(Tensor)
	result := C.wrap_ReluBackwardAccumulate(
		(*C.Context)(ctx.UnsafePtr()),
		output.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		input.Grad().(*tensor).cTensor,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// tanhBackward computes the gradient for tanh.
// d(tanh(x))/dx = 1 - tanh(x)^2
// grad_input += grad_output * (1 - output^2)
func tanhBackward(ctx Context, node ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	input := node.Inputs()[0]
	output := node.(Tensor)

	ones := Float(ctx, output.Shape(), 1.0)
	outputSquared := output.Times(ctx, output)
	localGrad := ones.Minus(ctx, outputSquared)
	gradInput := node.Grad().Times(ctx, localGrad)

	input.Grad().Accumulate(ctx, gradInput)
}
