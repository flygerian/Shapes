package layer

import (
	shapes "github.com/flygerian/shapes"
	"github.com/flygerian/shapes/tensor"
)

func Dense(inputSize int, outputSize int) func(*shapes.Context, *tensor.Tensor) *tensor.Tensor {
	var w, b *tensor.Tensor

	return func(ctx *shapes.Context, x *tensor.Tensor) *tensor.Tensor {
		fusedCtx := ctx.Fused()

		inputShape := tensor.ShapeOf(x)
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(fusedCtx, 0)
		}

		// Initialize weights once on first call.
		if w == nil {
			tensorLastDimSize := inputShape[len(inputShape)-1]
			w = tensor.FloatRandom(fusedCtx, tensor.Shape{uint32(outputSize), tensorLastDimSize})
			b = tensor.FloatRandom(fusedCtx, tensor.Shape{uint32(outputSize)})
		}

		// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
		o := input.Mul(fusedCtx, w.Transpose(fusedCtx)).Plus(fusedCtx, b)

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

	// Forward: o = x @ wᵀ + b
	grad := node.Grad.SafeUnSqueeze(ctx, 0)

	// ∂L/∂w = gradᵀ @ x  ([out, batch] @ [batch, in] = [out, in])
	dW := grad.Transpose(ctx).Mul(ctx, x.SafeUnSqueeze(ctx, 0))
	gradW := tensor.ReduceBroadcast(ctx, w, dW)
	w.Computation.Grad = w.Computation.Grad.Plus(ctx, gradW)

	// ∂L/∂x = grad @ w  ([batch, out] @ [out, in] = [batch, in])
	dX := grad.Mul(ctx, w)
	gradX := tensor.ReduceBroadcast(ctx, x, dX)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)

	// ∂L/∂b = grad (reduced to match b's shape)
	gradB := tensor.ReduceBroadcast(ctx, b, grad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)

	node.Parameters = []*tensor.Tensor{w, b}
}
