package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

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

static inline Result wrap_AddInPlace(Context *ctx, Tensor *a, Tensor *b) {
	return AddInPlace(ctx, a, b);
}
*/
import "C"

type hasNonMutatingBinaryOps interface {
	Plus(ctx Context, other Tensor) Tensor
	Minus(ctx Context, other Tensor) Tensor
	Times(ctx Context, other Tensor) Tensor
	Divide(ctx Context, other Tensor) Tensor
}

type hasMutatingBinaryOps interface {
	AddInPlace(ctx Context, other Tensor)
}

// Plus performs element-wise addition of t and other, returning a new tensor.
func (t *tensor) Plus(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_Add((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpAdd, addBackward, []Tensor{t, other}, []Tensor{}, nil)
	}
	return out
}

// Minus performs element-wise subtraction of other from t, returning a new tensor.
func (t *tensor) Minus(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_Subtract((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSubtract, subtractBackward, []Tensor{t, other}, []Tensor{}, nil)
	}
	return out
}

// Times performs element-wise multiplication of t and other, returning a new tensor.
func (t *tensor) Times(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_Multiply((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpMultiply, multiplyBackward, []Tensor{t, other}, []Tensor{}, nil)
	}
	return out
}

// Divide performs element-wise division of t by other, returning a new tensor.
func (t *tensor) Divide(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_Divide((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpDivide, divideBackward, []Tensor{t, other}, []Tensor{}, nil)
	}
	return out
}

// AddInPlace performs element-wise addition of other into t, modifying t in place.
func (t *tensor) AddInPlace(ctx Context, other Tensor) {
	result := C.wrap_AddInPlace((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// AddInPlace performs element-wise addition of other into wt, modifying wt in place.
func (wt *WrappedTensor) AddInPlace(other *WrappedTensor) {
	wt.tensor.AddInPlace(wt.context, other.tensor)
}
