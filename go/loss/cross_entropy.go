package loss

import (
	shapes "github.com/flygerian/shapes"
)

// CrossEntropy computes the fused softmax cross-entropy loss.
// yGround should be one-hot encoded ground truth labels (same shape as logits).
// logits should be the raw (pre-softmax) model outputs.
// Softmax is applied internally for numerical stability.
// Returns: -mean_batch(sum_classes(yGround * log(softmax(logits)))).
func CrossEntropy() func(shapes.Context, shapes.Tensor, shapes.Tensor) shapes.Tensor {
	return func(shapesCtx shapes.Context, yGround shapes.Tensor, logits shapes.Tensor) shapes.Tensor {

		fusedCtx := shapesCtx.Forward(shapes.WithInputs(yGround, logits))

		// Determine the class dimension (last dimension).
		ndims := uint(len(logits.Shape()))
		classDim := ndims - 1

		// Apply softmax along the class dimension for numerical stability:
		//   probs = exp(logits - max(logits)) / sum(exp(logits - max(logits)))
		maxLogits := logits.Max(fusedCtx, classDim)
		shifted := logits.Minus(fusedCtx, maxLogits)
		exp := shifted.Exp(fusedCtx)
		probs := exp.Divide(fusedCtx, exp.Sum(fusedCtx, classDim))

		// loss = -mean(sum_classes(yGround * log(probs)))
		// Sum over classes first, then mean over batch (or over all if 1D).
		logProbs := probs.Log(fusedCtx)
		perSampleLoss := yGround.Times(fusedCtx, logProbs).Sum(fusedCtx, classDim)
		result := perSampleLoss.Mean(fusedCtx).Negate(fusedCtx)

		// Use Saved API to keep probs alive for backward, then register via fusedCtx for sweeping

		fusedCtx.Finish(
			shapes.WithResult(result),
			shapes.WithBackward(crossEntropyBackward),
			shapes.WithMetadata(probs),
		)

		return result
	}
}

// crossEntropyBackward computes gradients for the fused softmax cross-entropy.
//
// The combined softmax + cross-entropy gradient simplifies to:
//
//	∂loss/∂logits = (1/batch_size) * (probs - yGround)
//
// where probs = softmax(logits) and batch_size is the number of samples
// (the dimension being averaged over by Mean).
func crossEntropyBackward(c shapes.Context, node shapes.ComputationGraphNode) {
	ctx := c.Backward()
	defer ctx.Finish()

	yGround := node.Inputs()[0]
	logits := node.Inputs()[1]
	probs := node.Metadata().(shapes.Tensor)

	// batch_size is the leading dimension (or 1 for a single 1D sample).
	logitsShape := logits.Shape()
	batchSize := float32(logitsShape[0])
	if len(logitsShape) == 1 {
		batchSize = 1.0
	}

	n := shapes.Float(ctx, probs.Shape(), batchSize)

	// local gradient: (probs - yGround) / batch_size
	localGrad := probs.Minus(ctx, yGround).Divide(ctx, n)
	gradLogits := node.Grad().Times(ctx, localGrad)
	reducedLogits := shapes.ReduceBroadcast(ctx, logits, gradLogits.(shapes.GradTensor))
	logits.Grad().Accumulate(ctx, reducedLogits)
}
