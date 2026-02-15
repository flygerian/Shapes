package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
#include <string.h>

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
import (
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

// Zeros creates a tensor filled with zeros.
func Zeros(ctx *shapes.Context, shape Shape) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Zeros((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(t)
	}
	return t
}

// Int creates a tensor filled with the given int8 value.
func Int(ctx *shapes.Context, shape Shape, value int8) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Int((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.i8(value))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(t)
	}
	return t
}

// Float creates a tensor filled with the given float32 value.
func Float(ctx *shapes.Context, shape Shape, value float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}
	cTensor := C.wrap_T_Float((*C.Context)(ctx.UnsafePtr()), dim(ctx, shape), C.f32(value))
	t := track(ctx, &Tensor{cTensor: cTensor})
	if ctx.GradEnabled {
		leafNode(t)
	}
	return t
}

// FromFloat32 creates a tensor from a Go []float32 slice with the given shape.
// Panics if the number of elements in data does not match the shape.
func FromFloat32(ctx *shapes.Context, shape Shape, data []float32) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	t := Zeros(ctx, shape)

	expected := int(t.cTensor.size)
	if len(data) != expected {
		panic("shapes: data length does not match shape")
	}

	C.memcpy(t.cTensor.values, unsafe.Pointer(&data[0]), C.size_t(expected)*C.sizeof_float)

	return t
}

// Clone creates a deep copy of the tensor.
func Clone(ctx *shapes.Context, src *Tensor) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Clone((*C.Context)(ctx.UnsafePtr()), src.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}
