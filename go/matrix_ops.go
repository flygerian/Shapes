package shapes

/*
#include "cwrappers.h"
*/
import "C"

type hasMatrixOps interface {
	Mul(ctx Context, other Tensor) Tensor
	Dot(ctx Context, other Tensor) Tensor
}

// Mul performs matrix multiplication of t and other, returning a new tensor.
func (t *tensor) Mul(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_MatMul((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// Dot computes the dot product of two 1-D tensors, returning a scalar tensor.
func (t *tensor) Dot(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_Dot((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}
