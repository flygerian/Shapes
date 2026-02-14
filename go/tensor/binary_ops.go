package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_Add(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Add(ctx, a, b, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Subtract(ctx, a, b, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Multiply(ctx, a, b, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Divide(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Divide(ctx, a, b, dest);
	*out = dest;
	return r;
}
*/
import "C"

// Plus performs element-wise addition of t and other, returning a new tensor.
func (t *Tensor) Plus(other *Tensor) *Tensor {
	ctx := requireSameCtx(t, other)
	var dest *C.Tensor
	result := C.wrap_Add((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.GradEnabled() {
		attachNode(out, addBackward, t, other)
	}
	return out
}

// Minus performs element-wise subtraction of other from t, returning a new tensor.
func (t *Tensor) Minus(other *Tensor) *Tensor {
	ctx := requireSameCtx(t, other)
	var dest *C.Tensor
	result := C.wrap_Subtract((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.GradEnabled() {
		attachNode(out, subtractBackward, t, other)
	}
	return out
}

// Times performs element-wise multiplication of t and other, returning a new tensor.
func (t *Tensor) Times(other *Tensor) *Tensor {
	ctx := requireSameCtx(t, other)
	var dest *C.Tensor
	result := C.wrap_Multiply((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.GradEnabled() {
		attachNode(out, multiplyBackward, t, other)
	}
	return out
}

// Divide performs element-wise division of t by other, returning a new tensor.
func (t *Tensor) Divide(other *Tensor) *Tensor {
	ctx := requireSameCtx(t, other)
	var dest *C.Tensor
	result := C.wrap_Divide((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.GradEnabled() {
		attachNode(out, divideBackward, t, other)
	}
	return out
}
