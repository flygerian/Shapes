package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
#include <string.h>

static inline Dim *makeDim(Memory *mem, dim_t *dims, u8 numDims) {
	Dim *d = allocate(mem, sizeof(Dim));
	d->dims = allocate(mem, sizeof(dim_t) * numDims);
	memcpy(d->dims, dims, sizeof(dim_t) * numDims);
	d->numOfDims = numDims;
	d->multipliers = NULL;
	return d;
}
*/
import "C"
import "unsafe"

type Shape = []uint32
type Range = []uint32

// Tensor wraps a C Tensor pointer.
type Tensor struct {
	cTensor     *C.Tensor
	Computation *ComputationGraphNode
	Label       string
}

// dim builds a C Dim on the arena from a Go shape slice in a single CGo call.
func dim(ctx *Context, shape Shape) *C.Dim {
	return C.makeDim(
		(*C.Memory)(ctx.UnsafeMemory()),
		(*C.dim_t)(unsafe.Pointer(&shape[0])),
		C.u8(len(shape)),
	)
}

// track registers a tensor's C pointer with the context for lifetime management.
func track(ctx *Context, t *Tensor) *Tensor {
	ctx.Track((*unsafe.Pointer)(unsafe.Pointer(&t.cTensor)))
	return t
}

// resultString converts a C Result code to a human-readable string.
func resultString(r uint32) string {
	return ResultString(r)
}

// ptrOffset returns an unsafe.Pointer offset by i elements of *C.dim_t size.
func ptrOffset(base *C.dim_t, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(base)) + uintptr(i)*unsafe.Sizeof(*base))
}

// UnsafeCPtr returns the C Tensor as an unsafe.Pointer for cross-package CGo casts.
func (t *Tensor) UnsafeCPtr() unsafe.Pointer {
	return unsafe.Pointer(t.cTensor)
}

// Track wraps a C tensor pointer into a Go Tensor and registers it with the context.
// Used by external packages (e.g., activation) to create Tensor values from C pointers.
func Track(ctx *Context, cPtr unsafe.Pointer) *Tensor {
	return track(ctx, &Tensor{cTensor: (*C.Tensor)(cPtr)})
}

// ShapeOf returns the shape of the tensor as a Go slice.
func ShapeOf(t *Tensor) Shape {
	return shapeOf(t)
}

// Dtype represents the data type of tensor elements.
type Dtype int

// Dtype constants matching the C Dtype enum.
const (
	DtypeF16 Dtype = iota
	DtypeF32
	DtypeF64
	DtypeU8
	DtypeU16
	DtypeU32
	DtypeU64
	DtypeI8
	DtypeI16
	DtypeI32
	DtypeI64
)

func (d Dtype) String() string {
	switch d {
	case DtypeF16:
		return "f16"
	case DtypeF32:
		return "f32"
	case DtypeF64:
		return "f64"
	case DtypeU8:
		return "u8"
	case DtypeU16:
		return "u16"
	case DtypeU32:
		return "u32"
	case DtypeU64:
		return "u64"
	case DtypeI8:
		return "i8"
	case DtypeI16:
		return "i16"
	case DtypeI32:
		return "i32"
	case DtypeI64:
		return "i64"
	default:
		return "unknown"
	}
}

// Dtype returns the tensor's data type.
func (t *Tensor) Dtype() Dtype {
	return Dtype(t.cTensor.dtype)
}

// Shape returns the shape of the tensor as a Go slice.
func (t *Tensor) Shape() Shape {
	return shapeOf(t)
}

// Shape returns the shape of the WrappedTensor.
func (wt *WrappedTensor) Shape() Shape {
	return wt.tensor.Shape()
}

// Dtype returns the data type of the WrappedTensor.
func (wt *WrappedTensor) Dtype() Dtype {
	return wt.tensor.Dtype()
}
