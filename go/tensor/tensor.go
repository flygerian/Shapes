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
	cTensor *C.Tensor
	ctx     *shapes.Context
}

// Ptr returns the C Tensor pointer for passing to C functions.
func (t *Tensor) Ptr() *C.Tensor {
	return t.cTensor
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
func requireSameCtx(a, b *Tensor) *shapes.Context {
	if a.ctx != b.ctx {
		panic("shapes: tensors belong to different contexts")
	}
	return a.ctx
}

// resultString converts a C Result code to a human-readable string.
func resultString(r uint32) string {
	return shapes.ResultString(r)
}
