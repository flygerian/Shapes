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
*/
import "C"
import (
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

// GetI8 reads an int8 value at the given coordinates.
func (t *Tensor) GetI8(coords ...uint32) (int8, error) {
	var v C.Value
	result := C.wrap_GetAt(t.cTensor, (*C.dim_t)(unsafe.Pointer(&coords[0])), C.u8(len(coords)), &v)
	if result != C.OK {
		return 0, shapes.ResultError(uint32(result))
	}
	return int8(v.as[0]), nil
}

// GetF32 reads a float32 value at the given coordinates.
func (t *Tensor) GetF32(coords ...uint32) (float32, error) {
	var v C.Value
	result := C.wrap_GetAt(t.cTensor, (*C.dim_t)(unsafe.Pointer(&coords[0])), C.u8(len(coords)), &v)
	if result != C.OK {
		return 0, shapes.ResultError(uint32(result))
	}
	return float32(*(*C.f32)(unsafe.Pointer(&v.as[0]))), nil
}
