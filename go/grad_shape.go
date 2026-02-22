package shapes

// reshapeBackward reshapes the gradient back to the input's original shape.
func reshapeBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	origShape := shapeOf(x)
	intShape := make([]int, len(origShape))
	for i, s := range origShape {
		intShape[i] = int(s)
	}
	gradX := node.Grad.Reshape(ctx, intShape...)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// transposeBackward transposes the gradient back using the same dims.
func transposeBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dims := node.Metadata.([2]uint32)
	gradX := node.Grad.Transpose(ctx, dims[0], dims[1])
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// squeezeBackward reshapes the gradient back to the input's original shape.
func squeezeBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	origShape := shapeOf(x)
	intShape := make([]int, len(origShape))
	for i, s := range origShape {
		intShape[i] = int(s)
	}
	gradX := node.Grad.Reshape(ctx, intShape...)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// squeezeDimBackward unsqueezes the gradient at the dim that was squeezed.
func squeezeDimBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dim := node.Metadata.(uint32)
	gradX := node.Grad.UnSqueeze(ctx, dim)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// oneHotBackward is a no-op: indices are discrete integers and have no gradient.
// The node is registered so the computation graph can traverse through OneHot.
func oneHotBackward(_ *Context, _ *ComputationGraphNode) {}

// unSqueezeBackward squeezes the gradient at the dim that was unsqueezed.
func unSqueezeBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dim := node.Metadata.(uint32)
	gradX := node.Grad.SqueezeDim(ctx, dim)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}
