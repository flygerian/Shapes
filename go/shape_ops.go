package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

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
	freeAlloc(ctx->memory, newShape);
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
)

type HasShapeOps interface {
	Slice(ctx *Context, ranges ...Range) *Tensor
	Reshape(ctx *Context, dims ...int) *Tensor
	Transpose(ctx *Context, dims ...uint32) *Tensor
	Squeeze(ctx *Context) *Tensor
	SqueezeDim(ctx *Context, dim uint32) *Tensor
	UnSqueeze(ctx *Context, dim uint32) *Tensor
	SafeUnSqueeze(ctx *Context, dims ...uint32) *Tensor
}

// TODO shape ops should be exact clones of the tensor
// Slice creates a view into the tensor. Each range is a Range{start, end}
// specifying a half-open interval for that dimension.
func (t *Tensor) Slice(ctx *Context, ranges ...Range) *Tensor {
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
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		// Deep-copy ranges so metadata is stable after the caller's slice goes out of scope.
		copiedRanges := make([]Range, len(ranges))
		copy(copiedRanges, ranges)
		ctx.newNode(out, OpSlice, sliceBackward, nil, t)
		out.Computation.Metadata = copiedRanges
	}
	return out
}

// Reshape returns a tensor with the same data but a different shape.
// Use -1 for one dimension to infer it automatically based on total element count.
func (t *Tensor) Reshape(ctx *Context, dims ...int) *Tensor {
	// Calculate total elements in the tensor
	totalElements := uint32(1)
	currentShape := shapeOf(t)
	for _, s := range currentShape {
		totalElements *= s
	}

	// Find -1 position and calculate product of known dimensions
	inferredIndex := -1
	knownProduct := uint32(1)
	for i, d := range dims {
		if d == -1 {
			if inferredIndex != -1 {
				panic("shapes: reshape can only have one -1 dimension")
			}
			inferredIndex = i
		} else if d < 0 {
			panic("shapes: reshape dimensions must be positive or -1")
		} else {
			knownProduct *= uint32(d)
		}
	}

	// Convert to uint32 shape, inferring -1 if present
	shape := make(Shape, len(dims))
	for i, d := range dims {
		if d == -1 {
			if knownProduct == 0 {
				panic("shapes: cannot infer dimension when other dimensions are zero")
			}
			if totalElements%knownProduct != 0 {
				panic("shapes: reshape size mismatch - cannot infer dimension")
			}
			shape[i] = totalElements / knownProduct
		} else {
			shape[i] = uint32(d)
		}
	}

	var dest *C.Tensor
	d := dim(ctx, shape)
	result := C.wrap_Reshape((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, d)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		ctx.newNode(out, OpReshape, reshapeBackward, nil, t)
	}
	return out
}

// Transpose swaps two dimensions, returning a view.
// With no extra args it swaps the last two dimensions (the common default).
// With two args it swaps those specific dimensions.
func (t *Tensor) Transpose(ctx *Context, dims ...uint32) *Tensor {
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
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		ctx.newNode(out, OpTranspose, transposeBackward, nil, t)
		out.Computation.Metadata = [2]uint32{d0, d1}
	}
	return out
}

// Squeeze removes all dimensions of size 1, returning a view.
func (t *Tensor) Squeeze(ctx *Context) *Tensor {
	var dest *C.Tensor
	result := C.wrap_Squeeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		ctx.newNode(out, OpSqueeze, squeezeBackward, nil, t)
	}
	return out
}

// SqueezeDim removes a single dimension at the given position (must be size 1), returning a view.
func (t *Tensor) SqueezeDim(ctx *Context, dim uint32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_SqueezeDim((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		ctx.newNode(out, OpSqueezeDim, squeezeDimBackward, nil, t)
		out.Computation.Metadata = dim
	}
	return out
}

// UnSqueeze inserts a dimension of size 1 at the given position, returning a view.
func (t *Tensor) UnSqueeze(ctx *Context, dim uint32) *Tensor {
	var dest *C.Tensor
	result := C.wrap_UnSqueeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &Tensor{cTensor: dest})
	if ctx.BackwardEnabled {
		ctx.newNode(out, OpUnSqueeze, unSqueezeBackward, nil, t)
		out.Computation.Metadata = dim
	}
	return out
}

// SafeUnsqeeze nserts a new dimention at dom 0 only if the tensor is 1D
func (t *Tensor) SafeUnSqueeze(ctx *Context, dims ...uint32) *Tensor {
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

// Slice creates a view into the tensor. Each range is a Range{start, end}
// specifying a half-open interval for that dimension.
func (wt *WrappedTensor) Slice(ranges ...Range) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Slice(wt.context, ranges...))
}

// Reshape returns a tensor with the same data but a different shape.
// Use -1 for one dimension to infer it automatically based on total element count.
func (wt *WrappedTensor) Reshape(dims ...int) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Reshape(wt.context, dims...))
}

// Transpose swaps two dimensions, returning a view.
// With no extra args it swaps the last two dimensions (the common default).
// With two args it swaps those specific dimensions.
func (wt *WrappedTensor) Transpose(dims ...uint32) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Transpose(wt.context, dims...))
}

// Squeeze removes all dimensions of size 1, returning a view.
func (wt *WrappedTensor) Squeeze() *WrappedTensor {
	return wt.context.Wrap(wt.tensor.Squeeze(wt.context))
}

// SqueezeDim removes a single dimension at the given position (must be size 1), returning a view.
func (wt *WrappedTensor) SqueezeDim(dim uint32) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.SqueezeDim(wt.context, dim))
}

// UnSqueeze inserts a dimension of size 1 at the given position, returning a view.
func (wt *WrappedTensor) UnSqueeze(dim uint32) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.UnSqueeze(wt.context, dim))
}

// SafeUnSqueeze inserts a new dimension at dim 0 only if the tensor is 1D.
func (wt *WrappedTensor) SafeUnSqueeze(dims ...uint32) *WrappedTensor {
	return wt.context.Wrap(wt.tensor.SafeUnSqueeze(wt.context, dims...))
}
