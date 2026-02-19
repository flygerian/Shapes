package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

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
import shapes "github.com/flygerian/shapes"

// Mul performs matrix multiplication of t and other, returning a new tensor.
func (t *Tensor) Mul(ctx *shapes.Context, other *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_MatMul((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Dot computes the dot product of two 1-D tensors, returning a scalar tensor.
func (t *Tensor) Dot(ctx *shapes.Context, other *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Dot((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}
