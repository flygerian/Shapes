package layer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func Dense(inputSize int, outputSize int) func(*shapes.Context, *tensor.Tensor) *tensor.Tensor {
	return func(ctx *shapes.Context, x *tensor.Tensor) *tensor.Tensor {
		fusedCtx := ctx.Fused()

		inputShape := tensor.ShapeOf(x)
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(fusedCtx, 0)
		}

		tensorLastDimSize := inputShape[len(inputShape)-1]
		w := tensor.Float(fusedCtx, tensor.Shape{uint32(outputSize), tensorLastDimSize}, 0.001)
		b := tensor.Float(fusedCtx, tensor.Shape{uint32(outputSize), 1}, 0.001)

		// wx + b
		o := w.Mul(fusedCtx, input.Transpose(fusedCtx)).Plus(fusedCtx, b)

		// If 1D input, squeeze back to 1D output.
		if is1D {
			o = o.Squeeze(fusedCtx)
		}

		tensor.AttachComputationGraphNode(o, tensor.OpDense, constructDenseBackwardPass, w, x, b)

		return o
	}
}

func constructDenseBackwardPass(ctx *shapes.Context, node *tensor.ComputationGraphNode) {
	w := node.Inputs[0]
	x := node.Inputs[1]
	b := node.Inputs[2]

	// Forward: o = w @ xᵀ + b
	// ∂L/∂w = grad @ x  (outer product when 1-D)

	dW := node.Grad.SafeUnSqueeze(ctx, 1).Mul(ctx, x.SafeUnSqueeze(ctx, 0))
	gradW := tensor.ReduceBroadcast(ctx, w, dW)
	w.Computation.Grad = w.Computation.Grad.Plus(ctx, gradW)

	// ∂L/∂x = (wᵀ @ grad)ᵀ  (transpose back to match x's shape)
	dX := w.SafeUnSqueeze(ctx).Transpose(ctx).Mul(
		ctx,
		node.Grad.SafeUnSqueeze(ctx, 1),
	).Transpose(ctx)

	gradX := tensor.ReduceBroadcast(ctx, x, dX)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)

	// ∂L/∂b = grad
	gradB := tensor.ReduceBroadcast(ctx, b, node.Grad.SafeUnSqueeze(ctx, 1))
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}
