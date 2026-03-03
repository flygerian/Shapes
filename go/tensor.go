package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

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

static inline void zeroTensorValues(Tensor *t) {
	if (t != NULL && t->values != NULL) {
		size_t bytes = t->size * getBytesForDtype(t->dtype);
		memset(t->values, 0, bytes);
	}
}
*/
import "C"
import "unsafe"

type Shape = []uint
type Range = []uint

type Tensor interface {
	hasMutatingBinaryOps
	hasNonMutatingBinaryOps
	hadReductionOps
	hasShapeOps
	hasUnaryOps
	hasCastOps
	hasMatrixOps
	hasAccessOps
	hasBackward

	ComputationGraphNode

	Grad() GradTensor
	Accumulate(ctx Context, operandB Tensor)
	Computation() Computation
	RequiresGrad() bool

	I64(ctx Context) Tensor

	SetLabel(label string)
	Label() string

	Dtype() Dtype
}

// Tensor wraps a C Tensor pointer.
type tensor struct {
	cTensor     *C.Tensor
	computation *Computation
	label       string
}

// dim builds a C Dim on the arena from a Go shape slice in a single CGo call.
func dim(ctx Context, shape Shape) *C.Dim {
	cDims := make([]C.dim_t, len(shape))
	for i, d := range shape {
		cDims[i] = C.dim_t(d)
	}

	return C.makeDim(
		(*C.Memory)(ctx.UnsafeMemory()),
		(*C.dim_t)(unsafe.Pointer(&cDims[0])),
		C.u8(len(shape)),
	)
}

// track registers a tensor with the context for lifetime management.
func track(ctx Context, t *tensor) *tensor {
	if ctx.GradEnabled() && t.computation == nil {
		leafNode(ctx, t)
	}

	ctx.Track(t)
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
func (t *tensor) UnsafeCPtr() unsafe.Pointer {
	return unsafe.Pointer(t.cTensor)
}

// Track wraps a C tensor pointer into a Go Tensor and registers it with the context.
// Used by external packages (e.g., activation) to create Tensor values from C pointers.
func Track(ctx Context, cPtr unsafe.Pointer) *tensor {
	return track(ctx, &tensor{cTensor: (*C.Tensor)(cPtr)})
}

// ShapeOf returns the shape of the tensor as a Go slice.
func ShapeOf(t *tensor) Shape {
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
func (t *tensor) Dtype() Dtype {
	return Dtype(t.cTensor.dtype)
}

// Shape returns the shape of the tensor as a Go slice.
func (t *tensor) Shape() Shape {
	return shapeOf(t)
}

func (t *tensor) Computation() Computation {
	return *t.computation
}

func (t *tensor) SetValuesToZero() {
	C.zeroTensorValues((*C.Tensor)(t.Grad().(*tensor).cTensor))
}

// UnsafeCTensor returns the underlying C tensor pointer for use in CGo calls from subpackages.
// This is unsafe and should only be used when necessary.
func (t *tensor) UnsafeCTensor() unsafe.Pointer {
	return unsafe.Pointer(t.cTensor)
}

func (t *tensor) SetLabel(label string) {
	t.label = label
}

func (t *tensor) Label() string {
	return t.label
}

func (t *tensor) Op() OpType {
	return t.computation.op
}

// Shape returns the shape of the WrappedTensor.
func (wt *WrappedTensor) Shape() Shape {
	return wt.tensor.Shape()
}
