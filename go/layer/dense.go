package layer

import (
	"fmt"

	shapes "github.com/flygerian/shapes"
)

type dense struct {
	w, b, o    shapes.Tensor
	inputSize  int
	outputSize int
	persistant bool
}

func (d *dense) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {

	// Initialize a fused context
	fusedCtx := ctx.Fused(
		shapes.WithInputs(x),
		shapes.WithHiddenState(d.w, d.b),
		shapes.WithOpType(shapes.OpDense),
	)

	inputShape := x.Shape()
	is1D := len(inputShape) == 1

	// If 1D, promote to [1, n] so MatMul works.
	input := x
	if is1D {
		input = x.UnSqueeze(fusedCtx, 0)
	}

	tensorLastDimSize := inputShape[len(inputShape)-1]
	if tensorLastDimSize != uint32(d.inputSize) {
		err := fmt.Errorf("shapes: Dense layer expecting dim size %d on dim[1] of input got dim size %d", d.inputSize, tensorLastDimSize)
		panic(err)
	}

	// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
	output := input.Mul(fusedCtx, d.w.Transpose(fusedCtx)).Plus(fusedCtx, d.b)

	// If 1D input, squeeze back to 1D output.
	if is1D {
		output = output.Squeeze(fusedCtx)
	}

	// Result area

	// If o is not initialized output declared, create o in the outer ctx
	if d.o == nil {
		d.o = shapes.Zeros(fusedCtx, output.Shape())
	}

	fusedCtx.Finish(
		shapes.WithResult(d.o),
		shapes.WithBackward(constructDenseBackwardPass),
	)
	return d.o
}

func Dense(outerCtx shapes.Context, inputSize int, outputSize int) Layer {

	// Initialize the hidden state
	w := shapes.FloatRandom(outerCtx, shapes.Shape{uint32(outputSize), uint32(inputSize)})
	b := shapes.FloatRandom(outerCtx, shapes.Shape{uint32(outputSize)})

	return &dense{w: w, b: b, inputSize: inputSize, outputSize: outputSize}
}

func constructDenseBackwardPass(ctx shapes.Context, node shapes.ComputationGraphNode) {
	noGraphCtx := ctx.NoGraph()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]

	hiddenState := node.HiddenState()
	w := hiddenState[0]
	b := hiddenState[1]

	// Forward: o = x @ wᵀ + b
	grad := node.Grad().SafeUnSqueeze(noGraphCtx, 0)

	// ∂L/∂w = gradᵀ @ x  ([out, batch] @ [batch, in] = [out, in])
	dW := grad.Transpose(noGraphCtx).Mul(noGraphCtx, x.SafeUnSqueeze(noGraphCtx, 0))
	gradW := shapes.ReduceBroadcast(noGraphCtx, w, dW.(shapes.GradTensor))
	w.Grad().Accumulate(noGraphCtx, gradW)

	// ∂L/∂x = grad @ w  ([batch, out] @ [out, in] = [batch, in])
	dX := grad.Mul(noGraphCtx, w)
	gradX := shapes.ReduceBroadcast(noGraphCtx, x, dX.(shapes.GradTensor))
	x.Grad().Accumulate(noGraphCtx, gradX)

	// ∂L/∂b = grad (reduced to match b's shape)
	gradB := shapes.ReduceBroadcast(noGraphCtx, b, grad.(shapes.GradTensor))
	b.Grad().Accumulate(noGraphCtx, gradB)
}
