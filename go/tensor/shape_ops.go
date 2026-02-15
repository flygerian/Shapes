package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"

// Fixed-arity wrappers that forward to the real variadic Slice.
static inline Result wrap_Slice1(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice2(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice3(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice4(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice5(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice6(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice7(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
	*out = dest;
	return res;
}
static inline Result wrap_Slice8(Context *ctx, Tensor *src, Tensor **out, Range *r) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result res = Slice(ctx, src, dest, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
	*out = dest;
	return res;
}

static inline Result wrap_Reshape(Context *ctx, Tensor *src, Tensor **out, Dim *newShape) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Reshape(ctx, src, dest, *newShape);
	*out = dest;
	return r;
}

static inline Result wrap_Transpose(Context *ctx, Tensor *src, Tensor **out, dim_t d0, dim_t d1) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Transpose(ctx, src, dest, d0, d1);
	*out = dest;
	return r;
}

static inline Result wrap_Squeeze(Context *ctx, Tensor *src, Tensor **out) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = Squeeze(ctx, src, dest);
	*out = dest;
	return r;
}

static inline Result wrap_SqueezeDim(Context *ctx, Tensor *src, Tensor **out, dim_t d) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = SqueezeDim(ctx, src, dest, d);
	*out = dest;
	return r;
}

static inline Result wrap_UnSqueeze(Context *ctx, Tensor *src, Tensor **out, dim_t d) {
	Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
	Result r = UnSqueeze(ctx, src, dest, d);
	*out = dest;
	return r;
}
*/
import "C"
import (
	"fmt"
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

// TODO shape ops should be exact clones of the tensor
// Slice creates a view into the tensor. Each range is a Range{start, end}
// specifying a half-open interval for that dimension.
func (t *Tensor) Slice(ctx *shapes.Context, ranges ...Range) *Tensor {
	ndims := len(ranges)
	if ndims == 0 || ndims > 8 {
		panic(fmt.Sprintf("shapes: slice supports 1-8 dimensions, got %d", ndims))
	}

	cRanges := make([]C.Range, ndims)
	for i, r := range ranges {
		if len(r) != 2 {
			panic(fmt.Sprintf("shapes: slice range %d must have 2 elements [start, end], got %d", i, len(r)))
		}
		cRanges[i] = C.Range{start: C.u64(r[0]), end: C.u64(r[1])}
	}

	cCtx := (*C.Context)(ctx.UnsafePtr())
	rp := (*C.Range)(unsafe.Pointer(&cRanges[0]))

	var dest *C.Tensor
	var result C.Result
	switch ndims {
	case 1:
		result = C.wrap_Slice1(cCtx, t.cTensor, &dest, rp)
	case 2:
		result = C.wrap_Slice2(cCtx, t.cTensor, &dest, rp)
	case 3:
		result = C.wrap_Slice3(cCtx, t.cTensor, &dest, rp)
	case 4:
		result = C.wrap_Slice4(cCtx, t.cTensor, &dest, rp)
	case 5:
		result = C.wrap_Slice5(cCtx, t.cTensor, &dest, rp)
	case 6:
		result = C.wrap_Slice6(cCtx, t.cTensor, &dest, rp)
	case 7:
		result = C.wrap_Slice7(cCtx, t.cTensor, &dest, rp)
	case 8:
		result = C.wrap_Slice8(cCtx, t.cTensor, &dest, rp)
	}

	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Reshape returns a tensor with the same data but a different shape.
func (t *Tensor) Reshape(ctx *shapes.Context, shape Shape) *Tensor {
	var dest *C.Tensor
	d := dim(ctx, shape)
	result := C.wrap_Reshape((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, d)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Transpose swaps two dimensions, returning a view.
// With no extra args it swaps the last two dimensions (the common default).
// With two args it swaps those specific dimensions.
func (t *Tensor) Transpose(ctx *shapes.Context, dims ...uint32) *Tensor {
	var d0, d1 uint32
	switch len(dims) {
	case 0:
		ndims := uint32(t.cTensor.shape.numOfDims)
		if ndims < 2 {
			return t
		}
		d0 = ndims - 2
		d1 = ndims - 1
	case 2:
		d0 = dims[0]
		d1 = dims[1]
	default:
		panic("shapes: transpose expects 0 or 2 dimension args")
	}

	var dest *C.Tensor
	result := C.wrap_Transpose(
		(*C.Context)(ctx.UnsafePtr()),
		t.cTensor,
		&dest,
		C.dim_t(d0),
		C.dim_t(d1),
	)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// Squeeze removes all dimensions of size 1, returning a view.
func (t *Tensor) Squeeze(ctx *shapes.Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Squeeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// SqueezeDim removes a single dimension at the given position (must be size 1), returning a view.
func (t *Tensor) SqueezeDim(ctx *shapes.Context, dim uint32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_SqueezeDim((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// UnSqueeze inserts a dimension of size 1 at the given position, returning a view.
func (t *Tensor) UnSqueeze(ctx *shapes.Context, dim uint32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_UnSqueeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &Tensor{cTensor: dest})
}

// SafeUnsqeeze nserts a new dimention at dom 0 only if the tensor is 1D
func (t *Tensor) SafeUnSqueeze(ctx *shapes.Context, dims ...uint32) *Tensor {
	if len(shapeOf(t)) > 1 {
		return t
	}

	if len(dims) > 1 {
		panic("shapes: only 1 dim is allowed when doing a safe unsqueeze")
	}

	if len(dims) == 1 {
		return t.UnSqueeze(ctx, dims[0])
	}

	return t.UnSqueeze(ctx, uint32(0))
}
