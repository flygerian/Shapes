package shapes

// Relu applies rectified linear activation element-wise.
func Relu(ctx Context, t Tensor) Tensor {
	return t.Relu(ctx)
}

// Tanh applies the hyperbolic tangent activation element-wise.
func Tanh(ctx Context, t Tensor) Tensor {
	return t.Tanh(ctx)
}

func floatMaskForTensor(ctx Context, t Tensor) Tensor {
	var maskData []float32
	switch values := t.Values().(type) {
	case []float32:
		maskData = make([]float32, len(values))
		for i, v := range values {
			if v > 0 {
				maskData[i] = 1.0
			}
		}
	case []float64:
		maskData = make([]float32, len(values))
		for i, v := range values {
			if v > 0 {
				maskData[i] = 1.0
			}
		}
	default:
		panic("shapes: relu requires float tensor")
	}

	mask := FromFloat32(ctx, t.Shape(), maskData)
	return castMaskToTensorDtype(ctx, mask, t.Dtype())
}

func castMaskToTensorDtype(ctx Context, mask Tensor, dtype Dtype) Tensor {
	switch dtype {
	case DtypeF16:
		return mask.F16(ctx)
	case DtypeF32:
		return mask.F32(ctx)
	case DtypeF64:
		return mask.F64(ctx)
	default:
		panic("shapes: relu requires float tensor")
	}
}

// reluBackward computes the gradient for relu.
// d(relu(x))/dx = 1 when x > 0, otherwise 0.
func reluBackward(ctx Context, node ComputationGraphNode) {
	backwardCtx := ctx.Backward()
	defer backwardCtx.Finish()

	input := node.Inputs()[0]
	output := node.(Tensor)

	typedMask := floatMaskForTensor(backwardCtx, output)
	gradInput := node.Grad().Times(backwardCtx, typedMask)

	input.Grad().Accumulate(backwardCtx, gradInput)
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
