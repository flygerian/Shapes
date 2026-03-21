package loss_fns

import (
	shapes "github.com/flygerian/shapes"
)

// CrossEntropy computes the fused softmax cross-entropy loss.
// yGround should be one-hot encoded ground truth labels (same shape as logits).
// logits should be the raw (pre-softmax) model outputs.
// Softmax is applied internally for numerical stability.
// Returns: -mean_batch(sum_classes(yGround * log(softmax(logits)))).
func CrossEntropy(ctx shapes.Context) func(shapes.Tensor, shapes.Tensor) shapes.Tensor {
	return func(yGround shapes.Tensor, logits shapes.Tensor) shapes.Tensor {

		fusedCtx := ctx.Forward(shapes.WithInputs(yGround, logits))
		result, probs := shapes.CrossEntropyForward(fusedCtx, yGround, logits)
		// C singleValueTensor returns a 0-d scalar; make it [1] so grad tracking
		// in the current Go context model remains valid.
		result = result.UnSqueeze(fusedCtx, 0)

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
	gradLogits := shapes.CrossEntropyBackward(ctx, yGround, probs, node.Grad().(shapes.Tensor))
	reducedLogits := shapes.ReduceBroadcast(ctx, logits, gradLogits.(shapes.GradTensor))
	logits.Grad().Accumulate(ctx, reducedLogits)
}
