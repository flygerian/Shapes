package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = MatMul(ctx, a, b, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Dot(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Dot(ctx, a, b, dest);
	*out = dest;
	return r;
}
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
