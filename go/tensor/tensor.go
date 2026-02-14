package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
*/
import "C"
import (
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

// Tensor wraps a C Tensor pointer
type Tensor struct {
	cTensor *C.Tensor
}

// Float creates a tensor filled with the given float value.
// shape is a slice of dimensions, e.g. []uint32{3, 4} for a 3x4 tensor.
func Float(ctx *shapes.Context, shape []uint32, value float32) *Tensor {
	// Allocate C array for dimensions
	numDims := len(shape)
	// Cast shapes.C.Memory to local C.Memory via unsafe.Pointer
	dimsPtr := (*C.dim_t)(C.allocate((*C.Memory)(unsafe.Pointer(ctx.Memory())), C.size_t(numDims)*C.size_t(unsafe.Sizeof(C.dim_t(0)))))

	// Copy shape to C array
	dimsSlice := unsafe.Slice(dimsPtr, numDims)
	for i, d := range shape {
		dimsSlice[i] = C.dim_t(d)
	}

	// Create Dim struct
	dim := C.Dim{
		dims:        dimsPtr,
		numOfDims:   C.u8(numDims),
		multipliers: nil,
	}

	// Create tensor - cast shapes.C.Context to local C.Context via unsafe.Pointer
	cTensor := C.T_Float((*C.Context)(unsafe.Pointer(ctx.Ptr())), dim, C.f32(value))

	return &Tensor{cTensor: cTensor}
}

// Ptr returns the C Tensor pointer for passing to C functions
func (t *Tensor) Ptr() *C.Tensor {
	return t.cTensor
}
