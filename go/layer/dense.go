package layer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func Dense(inputSize int, outputSize int) func(*shapes.Context, *tensor.Tensor) *tensor.Tensor {
	return func(ctx *shapes.Context, x *tensor.Tensor) *tensor.Tensor {
		noGrad := ctx.NoGrad()

		inputShape := tensor.ShapeOf(x)
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(noGrad, 0)
		}

		tensorLastDimSize := inputShape[len(inputShape)-1]
		w := tensor.Float(noGrad, tensor.Shape{uint32(outputSize), tensorLastDimSize}, 0.001)
		b := tensor.Float(noGrad, tensor.Shape{uint32(outputSize), 1}, 0.001)

		// wx + b
		o := w.Mul(noGrad, input.Transpose(noGrad)).Plus(noGrad, b)

		// If 1D input, squeeze back to 1D output.
		if is1D {
			o = o.Squeeze(noGrad)
		}

		if ctx.GradEnabled() {
			tensor.AttachComputationGraphNode(o, tensor.OpDense, constructDenseBackwardPass, w, x, b)
		}

		return o
	}
}

func constructDenseBackwardPass(ctx *shapes.Context, node *tensor.ComputationGraphNode) {
	w := node.Inputs[0]
	x := node.Inputs[1]
	b := node.Inputs[2]

	// Forward: o = w @ xᵀ + b
	// ∂L/∂w = grad @ x  (outer product when 1-D)
	dW := node.Grad.Mul(ctx, x.Transpose(ctx))
	gradW := tensor.ReduceBroadcast(ctx, w, dW)
	w.Computation.Grad = w.Computation.Grad.Plus(ctx, gradW)

	// ∂L/∂x = (wᵀ @ grad)ᵀ  (transpose back to match x's shape)
	dX := w.Transpose(ctx).Mul(ctx, node.Grad).Transpose(ctx)
	gradX := tensor.ReduceBroadcast(ctx, x, dX)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)

	// ∂L/∂b = grad
	gradB := tensor.ReduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}
