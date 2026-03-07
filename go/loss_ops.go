package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

// CrossEntropyForward computes fused softmax cross-entropy in C and returns:
// - loss: scalar mean loss tensor
// - probs: softmax probabilities (same shape as logits)
func CrossEntropyForward(ctx Context, yGround Tensor, logits Tensor) (Tensor, Tensor) {
	var loss *C.Tensor
	var probs *C.Tensor

	result := C.wrap_CrossEntropyForward(
		(*C.Context)(ctx.UnsafePtr()),
		yGround.(*tensor).cTensor,
		logits.(*tensor).cTensor,
		&loss,
		&probs,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: loss}), track(ctx, &tensor{cTensor: probs})
}

// CrossEntropyBackward computes dL/dlogits in C for fused softmax cross-entropy.
func CrossEntropyBackward(ctx Context, yGround Tensor, probs Tensor, gradOut Tensor) Tensor {
	var dLogits *C.Tensor

	result := C.wrap_CrossEntropyBackward(
		(*C.Context)(ctx.UnsafePtr()),
		yGround.(*tensor).cTensor,
		probs.(*tensor).cTensor,
		gradOut.(*tensor).cTensor,
		&dLogits,
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dLogits})
}
