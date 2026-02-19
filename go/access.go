package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

static inline Result wrap_GetTensorAt(Context *ctx, Tensor *source, dim_t index, Tensor *dest) {
	return GetTensorAt(ctx, source, index, dest);
}

static inline Result wrap_IndexWithTensor(Context *ctx, Tensor *source, Tensor *indices, Tensor *dest) {
	return IndexWithTensor(ctx, source, indices, dest);
}

static inline Result wrap_GetScalar(Tensor *t, Value *result) {
	return GetScalar(t, result);
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
import "fmt"

// Get returns a sub-tensor at the given coordinates.
// If the tensor is multidimensional, returns a view of the sub-tensor.
// If coordinates specify all dimensions, returns a 0-dimensional (scalar) tensor.
// Panics if coordinates are invalid.
//
// Get can also accept a tensor as an argument to perform advanced indexing,
// similar to PyTorch's x[indices] where indices is a tensor of integers.
func (t *Tensor) Get(ctx *Context, indices ...interface{}) *Tensor {
	if len(indices) == 0 {
		panic("shapes: Get requires at least one argument")
	}

	// Check if first argument is a tensor for advanced indexing
	if len(indices) == 1 {
		if idxTensor, ok := indices[0].(*Tensor); ok {
			return t.getWithTensor(ctx, idxTensor)
		}
	}

	// Otherwise, treat all arguments as coordinates
	coords := make([]uint32, len(indices))
	for i, v := range indices {
		switch val := v.(type) {
		case uint32:
			coords[i] = val
		case int:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint32(val)
		case uint:
			coords[i] = uint32(val)
		case uint64:
			coords[i] = uint32(val)
		case int32:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint32(val)
		case int64:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint32(val)
		default:
			panic(fmt.Sprintf("shapes: Get expects uint32 coordinates or a *Tensor, got %T", v))
		}
	}

	return t.getWithCoords(ctx, coords)
}

// getWithCoords returns a sub-tensor at the given coordinates.
func (t *Tensor) getWithCoords(ctx *Context, coords []uint32) *Tensor {
	current := t
	for _, idx := range coords {
		if current.cTensor.shape.numOfDims == 0 {
			panic("shapes: cannot index a 0-dimensional tensor")
		}

		var result C.Tensor
		cCtx := (*C.Context)(ctx.UnsafePtr())
		res := C.wrap_GetTensorAt(cCtx, current.cTensor, C.dim_t(idx), &result)
		if res != C.OK {
			panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
		}
		current = &Tensor{
			cTensor: &result,
		}
	}
	return current
}

// getWithTensor performs advanced indexing using a tensor of indices.
// The indices tensor must contain integer values.
func (t *Tensor) getWithTensor(ctx *Context, indices *Tensor) *Tensor {
	var result C.Tensor
	cCtx := (*C.Context)(ctx.UnsafePtr())
	res := C.wrap_IndexWithTensor(cCtx, t.cTensor, indices.cTensor, &result)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
	}

	return track(ctx, &Tensor{
		cTensor: &result,
	})
}

// Item extracts the scalar value from a 0-dimensional tensor.
// Returns the value as the appropriate Go type.
// Panics if the tensor is not 0-dimensional.
func (t *Tensor) Item() interface{} {
	if t.cTensor.shape.numOfDims != 0 {
		panic("shapes: Item() can only be called on 0-dimensional tensors")
	}

	var v C.Value
	res := C.wrap_GetScalar(t.cTensor, &v)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
	}

	switch t.Dtype() {
	case DtypeF16, DtypeF32, DtypeF64:
		return float32(C.value_as_double(v))
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
