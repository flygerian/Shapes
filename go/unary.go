package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

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

static inline Result wrap_Negate(Context *ctx, Tensor *t, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Negate(ctx, t, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Mean(Context *ctx, Tensor *t, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Mean(ctx, t, dest);
	*out = dest;
	return r;
}

static inline Result wrap_Log(Context *ctx, Tensor *t, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Log(ctx, t, dest);
	*out = dest;
	return r;
}
*/
import "C"

// Pow raises every element to the given power, returning a new tensor.
func (t *Tensor) Pow(ctx *Context, power float32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Pow((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.f32(power), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		attachNode(ctx, out, OpPow, powBackward, t)
		out.Computation.Metadata = power
	}
	return out
}

// Exp computes e^x for every element, returning a new tensor.
func (t *Tensor) Exp(ctx *Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Exp((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Negate negates every element (-t), returning a new tensor.
func (t *Tensor) Negate(ctx *Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Negate((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		attachNode(ctx, out, OpNegate, negateBackward, t)
	}
	return out
}

// Pow raises every element to the given power, returning a new WrappedTensor.
func (wt *WrappedTensor) Pow(power float32) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Pow(wt.context, power))
}

// Exp computes e^x for every element, returning a new WrappedTensor.
func (wt *WrappedTensor) Exp() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Exp(wt.context))
}

// Negate negates every element (-wt), returning a new WrappedTensor.
func (wt *WrappedTensor) Negate() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Negate(wt.context))
}

// Mean computes the mean of all elements, returning a scalar tensor.
func (t *Tensor) Mean(ctx *Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Mean((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Log computes the natural logarithm of every element, returning a new tensor.
func (t *Tensor) Log(ctx *Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Log((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Mean computes the mean of all elements, returning a scalar WrappedTensor.
func (wt *WrappedTensor) Mean() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Mean(wt.context))
}

// Log computes the natural logarithm of every element, returning a new WrappedTensor.
func (wt *WrappedTensor) Log() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Log(wt.context))
}
