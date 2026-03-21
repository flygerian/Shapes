package shapes

/*
#include "cwrappers.h"
*/
import "C"

type hasNonMutatingBinaryOps interface {
	Plus(ctx Context, other interface{}) Tensor
	Minus(ctx Context, other interface{}) Tensor
	Times(ctx Context, other interface{}) Tensor
	Divide(ctx Context, other interface{}) Tensor
	GreaterThan(ctx Context, other interface{}) Tensor
	GreaterThanOrEqual(ctx Context, other interface{}) Tensor
	LessThan(ctx Context, other interface{}) Tensor
	LessThanOrEqual(ctx Context, other interface{}) Tensor
}

type hasMutatingBinaryOps interface {
	AddInPlace(ctx Context, other interface{})
	SubtractInPlace(ctx Context, other interface{})
	MultiplyInPlace(ctx Context, other interface{})
}

func scalarTensorForDtype(ctx Context, dtype Dtype, value interface{}) Tensor {
	shape := Shape{1}

	if dtype == DtypeBool {
		switch v := value.(type) {
		case bool:
			if v {
				return Int8(ctx, shape, 1).Bool(ctx)
			}
			return Int8(ctx, shape, 0).Bool(ctx)
		default:
			panic("shapes: bool tensors only support bool scalars in binary ops")
		}
	}

	if dtype == DtypeU8 || dtype == DtypeU16 || dtype == DtypeU32 || dtype == DtypeU64 {
		panic("shapes: scalar numeric operands are unsupported for unsigned tensor dtypes")
	}

	scalar, ok := scalarToFloat32(value)
	if !ok {
		panic("shapes: unsupported scalar type for binary op")
	}

	t := Float(ctx, shape, scalar)
	if dtype == DtypeF32 {
		return t
	}
	return castToDtype(ctx, t, dtype)
}

func scalarToFloat32(value interface{}) (float32, bool) {
	switch v := value.(type) {
	case float32:
		return v, true
	case float64:
		return float32(v), true
	case int:
		return float32(v), true
	case int8:
		return float32(v), true
	case int16:
		return float32(v), true
	case int32:
		return float32(v), true
	case int64:
		return float32(v), true
	case uint:
		return float32(v), true
	case uint8:
		return float32(v), true
	case uint16:
		return float32(v), true
	case uint32:
		return float32(v), true
	case uint64:
		return float32(v), true
	default:
		return 0, false
	}
}

func castToDtype(ctx Context, t Tensor, dtype Dtype) Tensor {
	switch dtype {
	case DtypeF16:
		return t.F16(ctx)
	case DtypeF32:
		return t.F32(ctx)
	case DtypeF64:
		return t.F64(ctx)
	case DtypeU8:
		return t.U8(ctx)
	case DtypeU16:
		return t.U16(ctx)
	case DtypeU32:
		return t.U32(ctx)
	case DtypeU64:
		return t.U64(ctx)
	case DtypeI8:
		return t.I8(ctx)
	case DtypeI16:
		return t.I16(ctx)
	case DtypeI32:
		return t.I32(ctx)
	case DtypeI64:
		return t.I64(ctx)
	case DtypeBool:
		return t.Bool(ctx)
	default:
		panic("shapes: unsupported dtype")
	}
}

func normalizeBinaryOperand(ctx Context, lhs *tensor, other interface{}) Tensor {
	switch operand := other.(type) {
	case Tensor:
		return operand
	case *WrappedTensor:
		return operand.tensor
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, bool:
		return scalarTensorForDtype(ctx, lhs.Dtype(), operand)
	default:
		panic("shapes: binary operand must be a Tensor, WrappedTensor, or scalar number")
	}
}

// Plus performs element-wise addition of t and other, returning a new tensor.
func (t *tensor) Plus(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_Add((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpAdd, addBackward, []Tensor{t, otherTensor}, []Tensor{}, nil)
	}
	return out
}

// Minus performs element-wise subtraction of other from t, returning a new tensor.
func (t *tensor) Minus(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_Subtract((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSubtract, subtractBackward, []Tensor{t, otherTensor}, []Tensor{}, nil)
	}
	return out
}

// Times performs element-wise multiplication of t and other, returning a new tensor.
func (t *tensor) Times(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_Multiply((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpMultiply, multiplyBackward, []Tensor{t, otherTensor}, []Tensor{}, nil)
	}
	return out
}

// Divide performs element-wise division of t by other, returning a new tensor.
func (t *tensor) Divide(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_Divide((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpDivide, divideBackward, []Tensor{t, otherTensor}, []Tensor{}, nil)
	}
	return out
}

// GreaterThan performs element-wise greater-than comparison.
func (t *tensor) GreaterThan(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_GreaterThan((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// GreaterThanOrEqual performs element-wise greater-than-or-equal comparison.
func (t *tensor) GreaterThanOrEqual(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_GreaterThanOrEqual((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// LessThan performs element-wise less-than comparison.
func (t *tensor) LessThan(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_LessThan((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// LessThanOrEqual performs element-wise less-than-or-equal comparison.
func (t *tensor) LessThanOrEqual(ctx Context, other interface{}) Tensor {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	var dest *C.Tensor
	result := C.wrap_LessThanOrEqual((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// AddInPlace performs element-wise addition of other into t, modifying t in place.
func (t *tensor) AddInPlace(ctx Context, other interface{}) {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	result := C.wrap_AddInPlace((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// SubtractInPlace performs element-wise subtraction of other from t, modifying t in place.
func (t *tensor) SubtractInPlace(ctx Context, other interface{}) {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	result := C.wrap_SubtractInPlace((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

// MultiplyInPlace performs element-wise multiplication of t by other, modifying t in place.
func (t *tensor) MultiplyInPlace(ctx Context, other interface{}) {
	otherTensor := normalizeBinaryOperand(ctx, t, other)
	result := C.wrap_MultiplyInPlace((*C.Context)(ctx.UnsafePtr()), t.cTensor, otherTensor.(*tensor).cTensor)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
}

func wrappedBinaryResult(wt *WrappedTensor, out Tensor) *WrappedTensor {
	return &WrappedTensor{
		tensor:  out,
		context: wt.context,
	}
}

func unwrapWrappedOperand(other interface{}) interface{} {
	if wrapped, ok := other.(*WrappedTensor); ok {
		return wrapped.tensor
	}
	return other
}

// Plus performs element-wise addition and returns a wrapped tensor.
func (wt *WrappedTensor) Plus(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.Plus(wt.context, unwrapWrappedOperand(other)))
}

// Minus performs element-wise subtraction and returns a wrapped tensor.
func (wt *WrappedTensor) Minus(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.Minus(wt.context, unwrapWrappedOperand(other)))
}

// Times performs element-wise multiplication and returns a wrapped tensor.
func (wt *WrappedTensor) Times(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.Times(wt.context, unwrapWrappedOperand(other)))
}

// Divide performs element-wise division and returns a wrapped tensor.
func (wt *WrappedTensor) Divide(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.Divide(wt.context, unwrapWrappedOperand(other)))
}

// GreaterThan performs element-wise greater-than comparison and returns a wrapped tensor.
func (wt *WrappedTensor) GreaterThan(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.GreaterThan(wt.context, unwrapWrappedOperand(other)))
}

// GreaterThanOrEqual performs element-wise greater-than-or-equal comparison and returns a wrapped tensor.
func (wt *WrappedTensor) GreaterThanOrEqual(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.GreaterThanOrEqual(wt.context, unwrapWrappedOperand(other)))
}

// LessThan performs element-wise less-than comparison and returns a wrapped tensor.
func (wt *WrappedTensor) LessThan(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.LessThan(wt.context, unwrapWrappedOperand(other)))
}

// LessThanOrEqual performs element-wise less-than-or-equal comparison and returns a wrapped tensor.
func (wt *WrappedTensor) LessThanOrEqual(other interface{}) *WrappedTensor {
	return wrappedBinaryResult(wt, wt.tensor.LessThanOrEqual(wt.context, unwrapWrappedOperand(other)))
}

// AddInPlace performs element-wise addition of other into wt, modifying wt in place.
func (wt *WrappedTensor) AddInPlace(other interface{}) {
	wt.tensor.AddInPlace(wt.context, unwrapWrappedOperand(other))
}

// SubtractInPlace performs element-wise subtraction of other from wt, modifying wt in place.
func (wt *WrappedTensor) SubtractInPlace(other interface{}) {
	wt.tensor.SubtractInPlace(wt.context, unwrapWrappedOperand(other))
}

// MultiplyInPlace performs element-wise multiplication of wt by other, modifying wt in place.
func (wt *WrappedTensor) MultiplyInPlace(other interface{}) {
	wt.tensor.MultiplyInPlace(wt.context, unwrapWrappedOperand(other))
}
