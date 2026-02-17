package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_GetAt(Tensor *t, dim_t *coords, u8 ndims, Value *out) {
	Dim d = {.dims = coords, .numOfDims = ndims, .multipliers = NULL};
	return GetAt(t, d, out);
}

static inline double value_as_double(Value v) {
	switch (v.dtype) {
		case F16: return (double)v.as.f16;
		case F32: return (double)v.as.f32;
		case F64: return (double)v.as.f64;
		case U8:  return (double)v.as.u8;
		case U16: return (double)v.as.u16;
		case U32: return (double)v.as.u32;
		case U64: return (double)v.as.u64;
		case I8:  return (double)v.as.i8;
		case I16: return (double)v.as.i16;
		case I32: return (double)v.as.i32;
		case I64: return (double)v.as.i64;
		default:  return 0.0;
	}
}

static inline u64 value_as_u64(Value v) {
	switch (v.dtype) {
		case U8:  return (u64)v.as.u8;
		case U16: return (u64)v.as.u16;
		case U32: return (u64)v.as.u32;
		case U64: return (u64)v.as.u64;
		case I8:  return (u64)v.as.i8;
		case I16: return (u64)v.as.i16;
		case I32: return (u64)v.as.i32;
		case I64: return (u64)v.as.i64;
		default:  return 0;
	}
}

static inline i64 value_as_i64(Value v) {
	switch (v.dtype) {
		case I8:  return (i64)v.as.i8;
		case I16: return (i64)v.as.i16;
		case I32: return (i64)v.as.i32;
		case I64: return (i64)v.as.i64;
		case U8:  return (i64)v.as.u8;
		case U16: return (i64)v.as.u16;
		case U32: return (i64)v.as.u32;
		case U64: return (i64)v.as.u64;
		default:  return 0;
	}
}
*/
import "C"
import "unsafe"

// GetI8 reads an int8 value at the given coordinates.
// Panics if coordinates are invalid.
func (t *Tensor) GetI8(coords ...uint32) int8 {
	var v C.Value
	result := C.wrap_GetAt(t.cTensor, (*C.dim_t)(unsafe.Pointer(&coords[0])), C.u8(len(coords)), &v)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return int8(v.as[0])
}

// GetF32 reads a float32 value at the given coordinates.
// Panics if coordinates are invalid.
func (t *Tensor) GetF32(coords ...uint32) float32 {
	var v C.Value
	result := C.wrap_GetAt(t.cTensor, (*C.dim_t)(unsafe.Pointer(&coords[0])), C.u8(len(coords)), &v)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return float32(*(*C.f32)(unsafe.Pointer(&v.as[0])))
}

// Get reads a value at the given coordinates and returns it as the appropriate Go type.
// Panics if coordinates are invalid.
func (t *Tensor) Get(coords ...uint32) interface{} {
	var v C.Value
	result := C.wrap_GetAt(t.cTensor, (*C.dim_t)(unsafe.Pointer(&coords[0])), C.u8(len(coords)), &v)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	switch t.Dtype() {
	case DtypeF16, DtypeF32, DtypeF64:
		return float64(C.value_as_double(v))
	case DtypeU8:
		return uint8(C.value_as_u64(v))
	case DtypeU16:
		return uint16(C.value_as_u64(v))
	case DtypeU32:
		return uint32(C.value_as_u64(v))
	case DtypeU64:
		return uint64(C.value_as_u64(v))
	case DtypeI8:
		return int8(C.value_as_i64(v))
	case DtypeI16:
		return int16(C.value_as_i64(v))
	case DtypeI32:
		return int32(C.value_as_i64(v))
	case DtypeI64:
		return int64(C.value_as_i64(v))
	default:
		panic("shapes: unknown dtype")
	}
}
