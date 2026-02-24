package layer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
)

func Dense(outerCtx *shapes.Context, inputSize int, outputSize int) func(*shapes.WrappedTensor) *shapes.WrappedTensor {
	var w, b, o *shapes.WrappedTensor

	// Initialize the hidden state
	w = outerCtx.FloatRandom(shapes.Shape{uint32(outputSize), uint32(inputSize)})
	b = outerCtx.FloatRandom(shapes.Shape{uint32(outputSize)})

	return func(x *shapes.WrappedTensor) *shapes.WrappedTensor {

		// They need to live outsize of the fused scope

		// Initialize a fused context
		fusedCtx := outerCtx.Fused()

		inputShape := x.Shape()
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(0)
		}

		tensorLastDimSize := inputShape[len(inputShape)-1]
		if tensorLastDimSize != uint32(inputSize) {
			err := fmt.Errorf("shapes: Dense layer expecting dim size %d on dim[1] of input got dim size %d", inputSize, tensorLastDimSize)
			panic(err)
		}

		// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
		output := input.Mul(w.Transpose()).Plus(b)

		// If 1D input, squeeze back to 1D output.
		if is1D {
			output = output.Squeeze()
		}

		// If o is not initialized output declared, create o in the outer ctx
		if o == nil {
			o = outerCtx.Zeros(output.Shape())
		}

		// Clone the result into o
		outerCtx.Copy(output, o)

		// Register node via fusedCtx so forward temporaries are swept automatically
		fusedCtx.NewComputationGraphNode(o.Tensor(), shapes.OpDense, constructDenseBackwardPass, w.Tensor(), x.Tensor(), b.Tensor())

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
