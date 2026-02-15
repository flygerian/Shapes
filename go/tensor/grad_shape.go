package tensor

import shapes "github.com/flygerian/shapes"

// reshapeBackward reshapes the gradient back to the input's original shape.
func reshapeBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	origShape := shapeOf(x)
	gradX := node.Grad.Reshape(ctx, origShape)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// transposeBackward transposes the gradient back using the same dims.
func transposeBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dims := node.Metadata.([2]uint32)
	gradX := node.Grad.Transpose(ctx, dims[0], dims[1])
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// squeezeBackward reshapes the gradient back to the input's original shape.
func squeezeBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	origShape := shapeOf(x)
	gradX := node.Grad.Reshape(ctx, origShape)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// squeezeDimBackward unsqueezes the gradient at the dim that was squeezed.
func squeezeDimBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dim := node.Metadata.(uint32)
	gradX := node.Grad.UnSqueeze(ctx, dim)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// unSqueezeBackward squeezes the gradient at the dim that was unsqueezed.
func unSqueezeBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	dim := node.Metadata.(uint32)
	gradX := node.Grad.SqueezeDim(ctx, dim)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}
