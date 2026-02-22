package loss

import (
	shapes "github.com/flygerian/shapes"
)

// Mse computes the mean squared error loss between ground truth and predictions.
// yGround and yPred should be WrappedTensors with the same shape.
// Returns: sum((yPred - yGround)^2) reduced to a scalar.
func Mse(yGround *shapes.WrappedTensor, yPred *shapes.WrappedTensor) *shapes.WrappedTensor {
	ctx := yPred.Context()
	fusedCtx := ctx.Fused()

	// Compute: (yPred - yGround)^2
	diff := yPred.Tensor().Minus(fusedCtx, yGround.Tensor())
	se := diff.Pow(fusedCtx, 2)

	// Sum over all dimensions to reduce to scalar
	result := se
	shape := shapes.ShapeOf(result)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] > 1 {
			result = result.Sum(fusedCtx, uint32(i))
		}
	}

	fusedCtx.NewComputationGraphNode(result, shapes.OpMse, mseBackward, yGround.Tensor(), yPred.Tensor())
	return ctx.Wrap(result)
}

// mseBackward computes gradients for MSE loss.
// loss = sum((yPred - yGround)^2)
// ∂L/∂yPred = 2*(yPred - yGround)
// ∂L/∂yGround = -2*(yPred - yGround)
func mseBackward(ctx *shapes.Context, node *shapes.ComputationGraphNode) {
	yGround := node.Inputs[0]
	yPred := node.Inputs[1]

	// ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
	diff := yPred.Minus(ctx, yGround)
	two := shapes.Float(ctx, shapes.ShapeOf(diff), 2.0)
	localGrad := two.Times(ctx, diff)
	gradYPred := node.Grad.Times(ctx, localGrad)
	reducedYPred := shapes.ReduceBroadcast(ctx, yPred, gradYPred)
	yPred.Computation.Grad = yPred.Grad().Plus(ctx, reducedYPred)

	// ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
	negLocalGrad := localGrad.Negate(ctx)
	gradYGround := node.Grad.Times(ctx, negLocalGrad)
	reducedYGround := shapes.ReduceBroadcast(ctx, yGround, gradYGround)
	yGround.Computation.Grad = yGround.Grad().Plus(ctx, reducedYGround)
}
