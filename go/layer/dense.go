package layer

import (
	"fmt"
	"math"

	shapes "github.com/flygerian/shapes"
)

type dense struct {
	w, b, o    shapes.Tensor
	inputSize  int
	outputSize int

	isBiasApplied bool
}

func Dense(outerCtx shapes.Context, inputSize int, outputSize int, options ...denseLayerOption) Layer {

	// Initialize the hidden state
	initialization := (5 / 3) / (math.Pow(float64(inputSize), 0.5)) // Kaiming initalization ish
	w := shapes.FloatRandom(outerCtx, shapes.Shape{uint(outputSize), uint(inputSize)}).Times(outerCtx, initialization)
	b := shapes.FloatRandom(outerCtx, shapes.Shape{uint(outputSize)})

	return &dense{w: w, b: b, inputSize: inputSize, outputSize: outputSize}
}

func (d *dense) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {

	// Initialize a fused context
	fusedCtx := ctx.Forward(
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
	if tensorLastDimSize != uint(d.inputSize) {
		err := fmt.Errorf("shapes: Dense layer expecting dim size %d on dim[1] of input got dim size %d", d.inputSize, tensorLastDimSize)
		panic(err)
	}

	// x @ wᵀ + b (batch-friendly: [batch, in] @ [in, out] = [batch, out])
	d.o = input.Mul(fusedCtx, d.w.Transpose(fusedCtx))

	if d.isBiasApplied {
		d.o = d.o.Plus(fusedCtx, d.b)
	}

	// If 1D input, squeeze back to 1D output.
	if is1D {
		d.o = d.o.Squeeze(fusedCtx)
	}

	// Result area

	// If o is not initialized output declared, create o in the outer ctx
	if d.o == nil {
		d.o = shapes.Zeros(fusedCtx, d.o.Shape())
	}

	fusedCtx.Finish(
		shapes.WithResult(d.o),
		shapes.WithBackward(constructDenseBackwardPass),
	)
	return d.o
}

func constructDenseBackwardPass(ctx shapes.Context, node shapes.ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
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

type denseLayerOption func(layer dense)

func WithBias(isBiasApplied bool) denseLayerOption {
	return func(layer dense) {
		layer.isBiasApplied = isBiasApplied
	}
}
