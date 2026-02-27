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

static inline Result wrap_Max(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Max(ctx, t, dest, dim);
	*out = dest;
	return r;
}

static inline Result wrap_MeanDim(Context *ctx, Tensor *t, dim_t dim, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = MeanDim(ctx, t, dest, dim);
	*out = dest;
	return r;
}
*/
import "C"

type hasUnaryOps interface {
	Pow(ctx Context, power float32) Tensor
	Exp(ctx Context) Tensor
	Negate(ctx Context) Tensor
	Log(ctx Context) Tensor
	Mean(ctx Context, dims ...uint32) Tensor
	Max(ctx Context, dims ...uint32) Tensor
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

// Mean computes the mean of all elements (no dims) or along a single dimension.
func (t *tensor) Mean(ctx Context, dims ...uint32) Tensor {
	if len(dims) == 0 {
		var dest *C.Tensor
		result := C.wrap_Mean((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
		if result != C.OK {
			panic("shapes: " + resultString(uint32(result)))
		}
		return track(ctx, &tensor{cTensor: dest})
	}
	if len(dims) == 1 {
		var dest *C.Tensor
		result := C.wrap_MeanDim((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(dims[0]), &dest)
		if result != C.OK {
			panic("shapes: " + resultString(uint32(result)))
		}
		return track(ctx, &tensor{cTensor: dest})
	}
	panic("shapes: mean expects 0 or 1 dim args")
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

// Max reduces the tensor along a single dimension. If no dims provided, reduces all dims.
func (t *tensor) Max(ctx Context, dims ...uint32) Tensor {
	if len(dims) == 0 {
		out := t
		for i := uint32(0); i < uint32(len(shapeOf(t))); i++ {
			out = out.Max(ctx, i).(*tensor)
		}
		return out.Squeeze(ctx)
	}
	if len(dims) == 1 {
		var dest *C.Tensor
		result := C.wrap_Max((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(dims[0]), &dest)
		if result != C.OK {
			panic("shapes: " + resultString(uint32(result)))
		}
		return track(ctx, &tensor{cTensor: dest})
	}
	panic("shapes: max expects 0 or 1 dim args")
}
