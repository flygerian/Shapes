package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_Pow(Context *ctx, Tensor *t, f32 power, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Pow(ctx, t, power, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Exp(Context *ctx, Tensor *t, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Exp(ctx, t, dest);
	*out = dest;
	return r;
}
*/
import "C"

// Pow raises every element to the given power, returning a new tensor.
func (t *Tensor) Pow(power float32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Pow((*C.Context)(t.ctx.UnsafePtr()), t.cTensor, C.f32(power), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(t.ctx, &Tensor{cTensor: dest})
}

// Exp computes e^x for every element, returning a new tensor.
func (t *Tensor) Exp() *Tensor {
	var dest *C.Tensor
	result := C.wrap_Exp((*C.Context)(t.ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(t.ctx, &Tensor{cTensor: dest})
}
