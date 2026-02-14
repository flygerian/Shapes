package layer

import "github.com/flygerian/shapes/tensor"

func Dense(inputSize int, outputSize int) func(*tensor.Tensor) {
	return func(x *tensor.Tensor) {
		inputShape := tensor.ShapeOf(x)
		tensorLastDimSize := inputShape[len(inputShape)-1]
		w := tensor.Float(x.Context(), tensor.Shape{uint32(outputSize), tensorLastDimSize}, 0.001)
		b := tensor.Float(x.Context(), tensor.Shape{uint32(outputSize)}, 0.001)

		// wx + b
		var xT *tensor.Tensor

		if len(inputShape) > 1 {
			xT = x.Transpose(0, uint32(len(inputShape)-1))
		} else {
			xT = x
		}

		o := w.Mul(xT).Plus(b)

		if x.Context().GradEnabled() {
			tensor.AttachComputationGraphNode(o, tensor.OpDense, constructDenseBackwardPass, w, x, b)
		}
	}
}

func constructDenseBackwardPass(node *tensor.ComputationGraphNode) {
	ctx := node.Grad.Context().NoGrad()

	w := node.Inputs[0]
	x := node.Inputs[1]
	b := node.Inputs[2]

	dW := node.Grad.Mul(x)
	gradW := tensor.ReduceBroadcast(node.Grad.Context(), w, dW)
	w.Computation.Grad.Plus(gradW)

	var wT *tensor.Tensor

	wShape := tensor.ShapeOf(w)
	if len(wShape) > 1 {
		wT = w.Transpose(0, uint32(len(wShape)-1))
	} else {
		wT = w
	}

	dX := wT.Mul(node.Grad)
	gradX := tensor.ReduceBroadcast(node.Grad.Context(), x, dX)
	x.Computation.Grad.Plus(gradX)

	gradB := tensor.ReduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad.Plus(gradB)
}
