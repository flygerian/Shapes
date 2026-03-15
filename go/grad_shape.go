package shapes

// reshapeBackward reshapes the gradient back to the input's original shape.
func reshapeBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	origShape := x.Shape()
	intShape := make([]int, len(origShape))
	for i, s := range origShape {
		intShape[i] = int(s)
	}
	gradX := node.Grad().Reshape(noGraphCtx, intShape...)
	x.Grad().Accumulate(noGraphCtx, gradX)
}

// transposeBackward transposes the gradient back using the same dims.
func transposeBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	dims := node.Metadata().([2]uint)
	gradX := node.Grad().Transpose(noGraphCtx, dims[0], dims[1])
	x.Grad().Accumulate(noGraphCtx, gradX)
}

// permuteBackward applies the inverse permutation to map output grad back to input layout.
func permuteBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	dims := node.Metadata().([]uint)
	inverse := make([]uint, len(dims))
	for outAxis, inAxis := range dims {
		inverse[inAxis] = uint(outAxis)
	}

	gradX := node.Grad().Permute(noGraphCtx, inverse...)
	x.Grad().Accumulate(noGraphCtx, gradX)
}

// squeezeBackward reshapes the gradient back to the input's original shape.
func squeezeBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	origShape := x.Shape()
	intShape := make([]int, len(origShape))
	for i, s := range origShape {
		intShape[i] = int(s)
	}
	gradX := node.Grad().Reshape(noGraphCtx, intShape...)
	x.Grad().Accumulate(ctx, gradX)
}

// squeezeDimBackward unsqueezes the gradient at the dim that was squeezed.
func squeezeDimBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	dim := node.Metadata().(uint)
	gradX := node.Grad().UnSqueeze(ctx, dim)
	x.Grad().Accumulate(ctx, gradX)
}

// oneHotBackward is a no-op: indices are discrete integers and have no gradient.
// The node is registered so the computation graph can traverse through OneHot.
func oneHotBackward(_ Context, _ ComputationGraphNode) {}

// unSqueezeBackward squeezes the gradient at the dim that was unsqueezed.
func unSqueezeBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	dim := node.Metadata().(uint)
	gradX := node.Grad().SqueezeDim(ctx, dim)
	x.Grad().Accumulate(ctx, gradX)
}

// concatBackward slices the gradient and distributes it to all input tensors.
func concatBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	inputs := node.Inputs()
	dim := node.Metadata().(uint)
	grad := node.Grad()

	// Calculate cumulative offsets for slicing
	offset := uint(0)
	for _, input := range inputs {
		inputShape := input.Shape()
		dimSize := inputShape[dim]

		// Build range for slicing along the concat dimension
		ranges := make([]Range, len(inputShape))
		for d := range ranges {
			ranges[d] = Range{0, inputShape[d]}
		}
		ranges[dim] = Range{offset, offset + dimSize}

		// Slice the gradient for this input
		gradSlice := grad.Slice(noGraphCtx, ranges...)
		input.Grad().Accumulate(noGraphCtx, gradSlice)

		offset += dimSize
	}
}
