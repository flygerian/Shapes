package layer

import (
	shapes "github.com/flygerian/shapes"
)

func Dense(inputSize int, outputSize int) func(*shapes.WrappedTensor) *shapes.WrappedTensor {
	var w, b *shapes.WrappedTensor

	return func(x *shapes.WrappedTensor) *shapes.WrappedTensor {
		ctx := x.Context()
		fusedCtx := ctx.Fused()

		inputShape := x.Shape()
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(0)
		}

		// Initialize weights once on first call.
		// Wrap with the original ctx so context validation passes on subsequent ops.
		if w == nil {
			tensorLastDimSize := inputShape[len(inputShape)-1]
			w = ctx.Wrap(shapes.FloatRandom(fusedCtx, shapes.Shape{uint32(outputSize), tensorLastDimSize}))
			b = ctx.Wrap(shapes.FloatRandom(fusedCtx, shapes.Shape{uint32(outputSize)}))
		}

		// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
		o := input.Mul(w.Transpose()).Plus(b)

		// If 1D input, squeeze back to 1D output.
		if is1D {
			o = o.Squeeze()
		}

		ctx.NewComputationGraphNode(o.Tensor(), shapes.OpDense, constructDenseBackwardPass, w.Tensor(), x.Tensor(), b.Tensor())

		return o
	}
}

func constructDenseBackwardPass(ctx *shapes.Context, node *shapes.ComputationGraphNode) {
	w := node.Inputs[0]
	x := node.Inputs[1]
	b := node.Inputs[2]

	// Forward: o = x @ wᵀ + b
	grad := node.Grad.SafeUnSqueeze(ctx, 0)

	// ∂L/∂w = gradᵀ @ x  ([out, batch] @ [batch, in] = [out, in])
	dW := grad.Transpose(ctx).Mul(ctx, x.SafeUnSqueeze(ctx, 0))
	gradW := shapes.ReduceBroadcast(ctx, w, dW)
	w.Computation.Grad = w.Grad().Plus(ctx, gradW)

	// ∂L/∂x = grad @ w  ([batch, out] @ [out, in] = [batch, in])
	dX := grad.Mul(ctx, w)
	gradX := shapes.ReduceBroadcast(ctx, x, dX)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)

	// ∂L/∂b = grad (reduced to match b's shape)
	gradB := shapes.ReduceBroadcast(ctx, b, grad)
	b.Computation.Grad = b.Grad().Plus(ctx, gradB)

	node.Parameters = []*shapes.Tensor{w, b}
}
