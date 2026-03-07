package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

type hasUnaryOps interface {
	Pow(ctx Context, power float32) Tensor
	Exp(ctx Context) Tensor
	Tanh(ctx Context) Tensor
	Negate(ctx Context) Tensor
	Abs(ctx Context) Tensor
	Log(ctx Context) Tensor
	Mean(ctx Context, dims ...uint) Tensor
	Max(ctx Context, dims ...uint) Tensor
	ArgMax(ctx Context, dims ...uint) Tensor
}

// Pow raises every element to the given power, returning a new tensor.
func (t *tensor) Pow(ctx Context, power float32) Tensor {
	var dest *C.Tensor
	result := C.wrap_Pow((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.f32(power), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpPow, powBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = power
	}
	return out
}

// Exp computes e^x for every element, returning a new tensor.
func (t *tensor) Exp(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Exp((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// Tanh computes tanh(x) for every element, returning a new tensor.
func (t *tensor) Tanh(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Tanh((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// Negate negates every element (-t), returning a new tensor.
func (t *tensor) Negate(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Negate((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpNegate, negateBackward, []Tensor{t}, []Tensor{}, nil)
	}
	return out
}

// Log computes the natural logarithm of every element, returning a new tensor.
func (t *tensor) Log(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Log((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// Abs computes absolute value element-wise.
func (t *tensor) Abs(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Abs((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}
