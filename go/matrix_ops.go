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

// Mul performs matrix multiplication of t and other, returning a new tensor.
func (t *Tensor) Mul(ctx *Context, other *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_MatMul((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Dot computes the dot product of two 1-D tensors, returning a scalar tensor.
func (t *Tensor) Dot(ctx *Context, other *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Dot((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Mul performs matrix multiplication of wt and other, returning a new WrappedTensor.
func (wt *WrappedTensor) Mul(other *WrappedTensor) *WrappedTensor {
	wt.validateSameContext(other)
	return wt.context.Wrap(wt.tensor.Mul(wt.context, other.tensor))
}

// Dot computes the dot product of wt and other, returning a new WrappedTensor.
func (wt *WrappedTensor) Dot(other *WrappedTensor) *WrappedTensor {
	wt.validateSameContext(other)
	return wt.context.Wrap(wt.tensor.Dot(wt.context, other.tensor))
}
