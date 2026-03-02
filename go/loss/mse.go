package loss

import (
	shapes "github.com/flygerian/shapes"
)

// Mse computes the mean squared error loss between ground truth and predictions.
// yGround and yPred should be WrappedTensors with the same shape.
// Returns: sum((yPred - yGround)^2) reduced to a scalar.
func Mse(c shapes.Context, yGround shapes.Tensor, yPred shapes.Tensor) func(shapes.Tensor) shapes.Tensor {
	return func(t shapes.Tensor) shapes.Tensor {
		_ = t

		fusedCtx := c.Forward(shapes.WithInputs(yGround, yPred))

		// Compute: (yPred - yGround)^2
		diff := yPred.Minus(fusedCtx, yGround)
		se := diff.Pow(fusedCtx, 2)

		// Sum over all dimensions to reduce to scalar
		result := se
		shape := result.Shape()
		for i := len(shape) - 1; i >= 0; i-- {
			if shape[i] > 1 {
				result = result.Sum(fusedCtx, uint32(i))
			}
		}

		fusedCtx.Finish(
			shapes.WithResult(result),
			shapes.WithOpType(shapes.OpMse),
			shapes.WithBackward(mseBackward),
		)

		return result
	}
}

// mseBackward computes gradients for MSE loss.
// loss = sum((yPred - yGround)^2)
// ∂L/∂yPred = 2*(yPred - yGround)
// ∂L/∂yGround = -2*(yPred - yGround)
func mseBackward(c shapes.Context, node shapes.ComputationGraphNode) {
	ctx := c.Backward()
	defer ctx.Finish()

	yGround := node.Inputs()[0]
	yPred := node.Inputs()[1]

	// ∂L/∂yPred = upstream_grad * 2*(yPred - yGround)
	diff := yPred.Minus(ctx, yGround)
	two := shapes.Float(ctx, diff.Shape(), 2.0)
	localGrad := two.Times(ctx, diff)
	gradYPred := node.Grad().Times(ctx, localGrad)
	reducedYPred := shapes.ReduceBroadcast(ctx, yPred, gradYPred.(shapes.GradTensor))
	yPred.Grad().Accumulate(ctx, reducedYPred)

	// ∂L/∂yGround = upstream_grad * -2*(yPred - yGround)
	negLocalGrad := localGrad.Negate(ctx)
	gradYGround := node.Grad().Times(ctx, negLocalGrad)
	reducedYGround := shapes.ReduceBroadcast(ctx, yGround, gradYGround.(shapes.GradTensor))
	yGround.Grad().Accumulate(ctx, reducedYGround)
}
