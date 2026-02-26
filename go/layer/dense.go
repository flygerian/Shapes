package layer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
)

func Dense(outerCtx shapes.Context, inputSize int, outputSize int) func(*shapes.Tensor) *shapes.Tensor {
	var w, b, o *shapes.Tensor

	// Initialize the hidden state
	w = shapes.FloatRandom(outerCtx, shapes.Shape{uint32(outputSize), uint32(inputSize)})
	b = shapes.FloatRandom(outerCtx, shapes.Shape{uint32(outputSize)})

	return func(x *shapes.Tensor) *shapes.Tensor {

		// Initialize a fused context
		fusedCtx := outerCtx.Fused(shapes.WithInputs(x), shapes.WithHiddenState(w, b))

		inputShape := x.Shape()
		is1D := len(inputShape) == 1

		// If 1D, promote to [1, n] so MatMul works.
		input := x
		if is1D {
			input = x.UnSqueeze(fusedCtx, 0)
		}

		tensorLastDimSize := inputShape[len(inputShape)-1]
		if tensorLastDimSize != uint32(inputSize) {
			err := fmt.Errorf("shapes: Dense layer expecting dim size %d on dim[1] of input got dim size %d", inputSize, tensorLastDimSize)
			panic(err)
		}

		// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
		output := input.Mul(fusedCtx, w.Transpose(fusedCtx)).Plus(fusedCtx, b)

		// If 1D input, squeeze back to 1D output.
		if is1D {
			output = output.Squeeze(fusedCtx)
		}

		// Result area

		// If o is not initialized output declared, create o in the outer ctx
		if o == nil {
			o = shapes.Zeros(fusedCtx, output.Shape())
		}

		fusedCtx.Finish(shapes.WithResult(o))
		return o
	}
}

func constructDenseBackwardPass(ctx shapes.Context, node shapes.ComputationGraphNode) {
	noGraphCtx := ctx.NoGraph()
	defer noGraphCtx.Finish()

	w := node.Inputs()[0]
	x := node.Inputs()[1]
	b := node.Inputs()[2]

	// Forward: o = x @ wᵀ + b
	grad := node.Grad().SafeUnSqueeze(noGraphCtx, 0)

	// ∂L/∂w = gradᵀ @ x  ([out, batch] @ [batch, in] = [out, in])
	dW := grad.Transpose(noGraphCtx).Mul(noGraphCtx, x.SafeUnSqueeze(noGraphCtx, 0))
	gradW := shapes.ReduceBroadcast(noGraphCtx, w, dW)
	w.Grad().Accumulate(noGraphCtx, gradW)

	// ∂L/∂x = grad @ w  ([batch, out] @ [out, in] = [batch, in])
	dX := grad.Mul(noGraphCtx, w)
	gradX := shapes.ReduceBroadcast(noGraphCtx, x, dX)
	x.Grad().Accumulate(noGraphCtx, gradX)

	// ∂L/∂b = grad (reduced to match b's shape)
	gradB := shapes.ReduceBroadcast(noGraphCtx, b, grad)
	b.Grad().Accumulate(noGraphCtx, gradB)
}
