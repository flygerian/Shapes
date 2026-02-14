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
import shapes "github.com/flygerian/shapes"

// Add performs element-wise addition of t and other, returning a new tensor.
func (t *Tensor) Add(ctx *shapes.Context, other *Tensor) (*Tensor, error) {
	var dest *C.Tensor
	result := C.wrap_Add((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		return nil, shapes.ResultError(uint32(result))
	}
	return track(ctx, &Tensor{cTensor: dest}), nil
}

// Sub performs element-wise subtraction of other from t, returning a new tensor.
func (t *Tensor) Sub(ctx *shapes.Context, other *Tensor) (*Tensor, error) {
	var dest *C.Tensor
	result := C.wrap_Subtract((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		return nil, shapes.ResultError(uint32(result))
	}
	return track(ctx, &Tensor{cTensor: dest}), nil
}

// Mul performs element-wise multiplication of t and other, returning a new tensor.
func (t *Tensor) Mul(ctx *shapes.Context, other *Tensor) (*Tensor, error) {
	var dest *C.Tensor
	result := C.wrap_Multiply((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		return nil, shapes.ResultError(uint32(result))
	}
	return track(ctx, &Tensor{cTensor: dest}), nil
}

// Div performs element-wise division of t by other, returning a new tensor.
func (t *Tensor) Div(ctx *shapes.Context, other *Tensor) (*Tensor, error) {
	var dest *C.Tensor
	result := C.wrap_Divide((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
	if result != C.OK {
		return nil, shapes.ResultError(uint32(result))
	}
	return track(ctx, &Tensor{cTensor: dest}), nil
}
