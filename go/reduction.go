package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_Sum(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Sum(ctx, t, dest, dim);
	*out = dest;
	return r;
}
*/
import "C"

type hadReductionOps interface {
	Sum(ctx Context, dim uint32) Tensor
}

// Sum reduces the tensor along the given dimension by summing, returning a new tensor.
func (t *tensor) Sum(ctx Context, dim uint32) Tensor {
	var dest *C.Tensor
	result := C.wrap_Sum((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(dim), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSum, sumBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = dim
	}
	return out
}
