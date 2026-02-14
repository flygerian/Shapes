package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Tensor *wrap_T_Zeros(Context *ctx, Dim *shape) {
	return T_Zeros(ctx, *shape);
}
static inline Tensor *wrap_T_Int(Context *ctx, Dim *shape, i8 value) {
	return T_Int(ctx, *shape, value);
}
static inline Tensor *wrap_T_Float(Context *ctx, Dim *shape, f32 value) {
	return T_Float(ctx, *shape, value);
}
static inline Result wrap_Clone(Context *ctx, Tensor *src, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Clone(ctx, src, dest);
	*out = dest;
	return r;
}
*/
import "C"
import shapes "github.com/flygerian/shapes"

// Zeros creates a tensor filled with zeros.
func Zeros(ctx *shapes.Context, shape Shape) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Zeros((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape))
	return track(ctx, &Tensor{cTensor: cTensor})
}

// Int creates a tensor filled with the given int8 value.
func Int(ctx *shapes.Context, shape Shape, value int8) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Int((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.i8(value))
	return track(ctx, &Tensor{cTensor: cTensor})
}

// Float creates a tensor filled with the given float32 value.
func Float(ctx *shapes.Context, shape Shape, value float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Float((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.f32(value))
	return track(ctx, &Tensor{cTensor: cTensor})
}

// Clone creates a deep copy of the tensor.
func Clone(src *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Clone((*C.Context)(src.ctx.UnsafePtr()), src.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(src.ctx, &Tensor{cTensor: dest})
}
