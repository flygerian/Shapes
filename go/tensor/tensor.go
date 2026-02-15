package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

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
import (
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

type Shape = []uint32
type Range = []uint32

// Tensor wraps a C Tensor pointer and holds a reference to the context it was created with.
type Tensor struct {
	cTensor     *C.Tensor
	ctx         *shapes.Context
	Computation *ComputationGraphNode
	Label       string
}

// dim builds a C Dim on the arena from a Go shape slice in a single CGo call.
func dim(ctx *shapes.Context, shape Shape) *C.Dim {
	return C.makeDim(
		(*C.Memory)(ctx.UnsafeMemory()),
		(*C.dim_t)(unsafe.Pointer(&shape[0])),
		C.u8(len(shape)),
	)
}

// track registers a tensor's C pointer with the context for lifetime management.
func track(ctx *shapes.Context, t *Tensor) *Tensor {
	t.ctx = ctx
	ctx.Track((*unsafe.Pointer)(unsafe.Pointer(&t.cTensor)))
	return t
}

// requireSameCtx checks that two tensors belong to the same context. Panics if not.
// Compares underlying C context pointers so that NoGrad-derived contexts pass.
func requireSameCtx(a, b *Tensor) *shapes.Context {
	if a.ctx.UnsafePtr() != b.ctx.UnsafePtr() {
		panic("shapes: tensors belong to different contexts")
	}
	return a.ctx
}

// resultString converts a C Result code to a human-readable string.
func resultString(r uint32) string {
	return shapes.ResultString(r)
}

// ptrOffset returns an unsafe.Pointer offset by i elements of *C.dim_t size.
func ptrOffset(base *C.dim_t, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(unsafe.Pointer(base)) + uintptr(i)*unsafe.Sizeof(*base))
}

// UnsafeCPtr returns the C Tensor as an unsafe.Pointer for cross-package CGo casts.
func (t *Tensor) UnsafeCPtr() unsafe.Pointer {
	return unsafe.Pointer(t.cTensor)
}

// Context returns the shapes.Context this tensor belongs to.
func (t *Tensor) Context() *shapes.Context {
	return t.ctx
}

// Track wraps a C tensor pointer into a Go Tensor and registers it with the context.
// Used by external packages (e.g., activation) to create Tensor values from C pointers.
func Track(ctx *shapes.Context, cPtr unsafe.Pointer) *Tensor {
	return track(ctx, &Tensor{cTensor: (*C.Tensor)(cPtr)})
}

// ShapeOf returns the shape of the tensor as a Go slice.
func ShapeOf(t *Tensor) Shape {
	return shapeOf(t)
}

// Dtype constants matching the C Dtype enum.
const (
	DtypeF16 = iota
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

// Dtype returns the tensor's data type as an integer matching the C Dtype enum.
func (t *Tensor) Dtype() int {
	return int(t.cTensor.dtype)
}

// AttachComputationGraphNode is the exported version of attachNode for use by external packages.
func AttachComputationGraphNode(result *Tensor, op OpType, backward BackwardFn, inputs ...*Tensor) {
	attachNode(result, op, backward, inputs...)
}
