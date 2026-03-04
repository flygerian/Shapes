package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

type hasNonMutatingBinaryOps interface {
	Plus(ctx Context, other Tensor) Tensor
	Minus(ctx Context, other Tensor) Tensor
	Times(ctx Context, other Tensor) Tensor
	Divide(ctx Context, other Tensor) Tensor
	GreaterThan(ctx Context, other Tensor) Tensor
	GreaterThanOrEqual(ctx Context, other Tensor) Tensor
	LessThan(ctx Context, other Tensor) Tensor
	LessThanOrEqual(ctx Context, other Tensor) Tensor
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

// GreaterThan performs element-wise greater-than comparison.
func (t *tensor) GreaterThan(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_GreaterThan((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// GreaterThanOrEqual performs element-wise greater-than-or-equal comparison.
func (t *tensor) GreaterThanOrEqual(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_GreaterThanOrEqual((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// LessThan performs element-wise less-than comparison.
func (t *tensor) LessThan(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_LessThan((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// LessThanOrEqual performs element-wise less-than-or-equal comparison.
func (t *tensor) LessThanOrEqual(ctx Context, other Tensor) Tensor {
	var dest *C.Tensor
	result := C.wrap_LessThanOrEqual((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
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
