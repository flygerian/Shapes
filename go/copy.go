package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
#include <string.h>
*/
import "C"
import "unsafe"

type hasCopyOps interface {
	toF16Slice() []float32
	toF32Slice() []float32
	toF64Slice() []float64
	toU8Slice() []uint8
	toU16Slice() []uint16
	toU32Slice() []uint32
	toU64Slice() []uint64
	toI8Slice() []int8
	toI16Slice() []int16
	toI32Slice() []int32
	toI64Slice() []int64
	toBoolSlice() []bool
}

func tensorForCopy(t *tensor) *C.Tensor {
	if t == nil || t.cTensor == nil {
		panic("shapes: nil tensor")
	}
	if !bool(t.cTensor.isContigous) {
		panic("shapes: Values requires contiguous tensor")
	}
	return t.cTensor
}

func (t *tensor) toF16Slice() []float32 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_float))
	return out
}

func (t *tensor) toF32Slice() []float32 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]float32, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_float))
	return out
}

func (t *tensor) toF64Slice() []float64 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_double))
	return out
}

func (t *tensor) toU8Slice() []uint8 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]uint8, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_uint8_t))
	return out
}

func (t *tensor) toU16Slice() []uint16 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]uint16, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_uint16_t))
	return out
}

func (t *tensor) toU32Slice() []uint32 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]uint32, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_uint32_t))
	return out
}

func (t *tensor) toU64Slice() []uint64 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]uint64, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_uint64_t))
	return out
}

func (t *tensor) toI8Slice() []int8 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]int8, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_int8_t))
	return out
}

func (t *tensor) toI16Slice() []int16 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]int16, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_int16_t))
	return out
}

func (t *tensor) toI32Slice() []int32 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]int32, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_int32_t))
	return out
}

func (t *tensor) toI64Slice() []int64 {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]int64, n)
	if n == 0 {
		return out
	}
	C.memcpy(unsafe.Pointer(&out[0]), ct.values, C.size_t(n)*C.size_t(C.sizeof_int64_t))
	return out
}

func (t *tensor) toBoolSlice() []bool {
	ct := tensorForCopy(t)
	n := int(ct.size)
	out := make([]bool, n)
	if n == 0 {
		return out
	}

	raw := make([]C.bool, n)
	C.memcpy(unsafe.Pointer(&raw[0]), ct.values,
		C.size_t(n)*C.size_t(unsafe.Sizeof(C.bool(false))))

	for i := range n {
		out[i] = raw[i] != C.bool(false)
	}

	return out
}

func (t *tensor) Values() interface{} {
	if t == nil || t.cTensor == nil {
		panic("shapes: nil tensor")
	}

	switch t.Dtype() {
	case DtypeBool:
		return t.toBoolSlice()
	case DtypeF16:
		return t.toF16Slice()
	case DtypeF32:
		return t.toF32Slice()
	case DtypeF64:
		return t.toF64Slice()
	case DtypeU8:
		return t.toU8Slice()
	case DtypeU16:
		return t.toU16Slice()
	case DtypeU32:
		return t.toU32Slice()
	case DtypeU64:
		return t.toU64Slice()
	case DtypeI8:
		return t.toI8Slice()
	case DtypeI16:
		return t.toI16Slice()
	case DtypeI32:
		return t.toI32Slice()
	case DtypeI64:
		return t.toI64Slice()
	default:
		panic("shapes: unknown dtype")
	}
}
