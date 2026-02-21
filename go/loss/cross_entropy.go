package loss

import (
	shapes "github.com/flygerian/shapes"
)

// CrossEntropy computes the fused softmax cross-entropy loss.
// yGround should be one-hot encoded ground truth labels (same shape as logits).
// logits should be the raw (pre-softmax) model outputs.
// Softmax is applied internally for numerical stability.
// Returns: -mean_batch(sum_classes(yGround * log(softmax(logits)))).
func CrossEntropy(yGround *shapes.WrappedTensor, logits *shapes.WrappedTensor) *shapes.WrappedTensor {
	ctx := logits.Context()
	fusedCtx := ctx.Fused()

	// Determine the class dimension (last dimension).
	ndims := uint32(len(shapes.ShapeOf(logits.Tensor())))
	classDim := ndims - 1

	// Apply softmax along the class dimension for numerical stability:
	//   probs = exp(logits - max(logits)) / sum(exp(logits - max(logits)))
	maxLogits := logits.Tensor().Max(fusedCtx, classDim)
	shifted := logits.Tensor().Minus(fusedCtx, maxLogits)
	exp := shifted.Exp(fusedCtx)
	probs := exp.Divide(fusedCtx, exp.Sum(fusedCtx, classDim))

	// loss = -mean(sum_classes(yGround * log(probs)))
	// Sum over classes first, then mean over batch (or over all if 1D).
	logProbs := probs.Log(fusedCtx)
	perSampleLoss := yGround.Tensor().Times(fusedCtx, logProbs).Sum(fusedCtx, classDim)
	result := perSampleLoss.Mean(fusedCtx).Negate(fusedCtx)

	ctx.NewComputationGraphNode(result, shapes.OpCrossEntropy, crossEntropyBackward, yGround.Tensor(), logits.Tensor())
	// Store probs for use in the backward pass.
	result.Computation.Metadata = probs

	return ctx.Wrap(result)
}

// crossEntropyBackward computes gradients for the fused softmax cross-entropy.
//
// The combined softmax + cross-entropy gradient simplifies to:
//
//	∂loss/∂logits = (1/batch_size) * (probs - yGround)
//
// where probs = softmax(logits) and batch_size is the number of samples
// (the dimension being averaged over by Mean).
func crossEntropyBackward(ctx *shapes.Context, node *shapes.ComputationGraphNode) {
	yGround := node.Inputs[0]
	logits := node.Inputs[1]
	probs := node.Metadata.(*shapes.Tensor)

	// batch_size is the leading dimension (or 1 for a single 1D sample).
	logitsShape := shapes.ShapeOf(logits)
	batchSize := float32(logitsShape[0])
	if len(logitsShape) == 1 {
		batchSize = 1.0
	}

	n := shapes.Float(ctx, shapes.ShapeOf(probs), batchSize)

	// local gradient: (probs - yGround) / batch_size
	localGrad := probs.Minus(ctx, yGround).Divide(ctx, n)
	gradLogits := node.Grad.Times(ctx, localGrad)
	reducedLogits := shapes.ReduceBroadcast(ctx, logits, gradLogits)
	logits.Computation.Grad = logits.Grad().Plus(ctx, reducedLogits)

	shapes.MarkIntermediate(ctx, n)
	shapes.MarkIntermediate(ctx, localGrad)
	shapes.MarkIntermediate(ctx, gradLogits)
	shapes.MarkIfIntermediate(ctx, reducedLogits, gradLogits)
}
