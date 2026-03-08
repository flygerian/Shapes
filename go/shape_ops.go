package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type hasShapeOps interface {
	Slice(ctx Context, ranges ...Range) Tensor
	Reshape(ctx Context, dims ...int) Tensor
	Transpose(ctx Context, dims ...uint) Tensor
	Permute(ctx Context, dims ...uint) Tensor
	Squeeze(ctx Context) Tensor
	SqueezeDim(ctx Context, dim uint) Tensor
	UnSqueeze(ctx Context, dim uint) Tensor
	SafeUnSqueeze(ctx Context, dims ...uint) Tensor
	Shape() Shape
}

// TODO shape ops should be exact clones of the tensor
// Slice creates a view into the tensor. Each range is a Range{start, end}
// specifying a half-open interval for that dimension.
func (t *tensor) Slice(ctx Context, ranges ...Range) Tensor {
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
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		// Deep-copy ranges (including each inner [start,end] slice) so metadata
		// stays stable after the caller's variadic arguments go out of scope.
		copiedRanges := make([]Range, len(ranges))
		for i, r := range ranges {
			copiedRanges[i] = make(Range, len(r))
			copy(copiedRanges[i], r)
		}
		toComputationGraphNode(out, OpSlice, sliceBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = copiedRanges
	}
	return out
}

// Reshape returns a tensor with the same data but a different shape.
// Use -1 for one dimension to infer it automatically based on total element count.
func (t *tensor) Reshape(ctx Context, dims ...int) Tensor {
	// Calculate total elements in the tensor
	totalElements := uint(1)
	currentShape := t.Shape()
	for _, s := range currentShape {
		totalElements *= s
	}

	// Find -1 position and calculate product of known dimensions
	inferredIndex := -1
	knownProduct := uint(1)
	for i, d := range dims {
		if d == -1 {
			if inferredIndex != -1 {
				panic("shapes: reshape can only have one -1 dimension")
			}
			inferredIndex = i
		} else if d < 0 {
			panic("shapes: reshape dimensions must be positive or -1")
		} else {
			knownProduct *= uint(d)
		}
	}

	// Convert to shape (uint), inferring -1 if present
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
			shape[i] = uint(d)
		}
	}

	var dest *C.Tensor
	d := dim(ctx, shape)
	result := C.wrap_Reshape((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, d)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpReshape, reshapeBackward, []Tensor{t}, []Tensor{}, nil)
	}
	return out
}

// Transpose swaps two dimensions, returning a view.
// With no extra args it swaps the last two dimensions (the common default).
// With two args it swaps those specific dimensions.
func (t *tensor) Transpose(ctx Context, dims ...uint) Tensor {
	var d0, d1 uint
	switch len(dims) {
	case 0:
		ndims := uint(t.cTensor.shape.numOfDims)
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
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpTranspose, transposeBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = [2]uint{d0, d1}
	}
	return out
}

// Permute reorders axes according to dims, where dims[i] selects the source axis for output axis i.
// Example: NCHW -> NHWC uses Permute(ctx, 0, 2, 3, 1).
func (t *tensor) Permute(ctx Context, dims ...uint) Tensor {
	ndims := len(t.Shape())
	if len(dims) != ndims {
		panic(fmt.Sprintf("shapes: permute expects %d dims, got %d", ndims, len(dims)))
	}

	seen := make([]bool, ndims)
	for _, d := range dims {
		if int(d) >= ndims {
			panic(fmt.Sprintf("shapes: permute dim %d out of bounds for rank %d", d, ndims))
		}
		if seen[d] {
			panic(fmt.Sprintf("shapes: permute dimensions must be unique, got duplicate %d", d))
		}
		seen[d] = true
	}

	var dest *C.Tensor
	order := dim(ctx, dims)
	result := C.wrap_Permute((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, order)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		copiedDims := append([]uint(nil), dims...)
		toComputationGraphNode(out, OpPermute, permuteBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = copiedDims
	}

	return out
}

// Squeeze removes all dimensions of size 1, returning a view.
func (t *tensor) Squeeze(ctx Context) Tensor {
	var dest *C.Tensor
	result := C.wrap_Squeeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSqueeze, squeezeBackward, []Tensor{t}, []Tensor{}, nil)
	}
	return out
}

// SqueezeDim removes a single dimension at the given position (must be size 1), returning a view.
func (t *tensor) SqueezeDim(ctx Context, dim uint) Tensor {
	var dest *C.Tensor
	result := C.wrap_SqueezeDim((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSqueezeDim, squeezeDimBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = dim
	}
	return out
}

// UnSqueeze inserts a dimension of size 1 at the given position, returning a view.
func (t *tensor) UnSqueeze(ctx Context, dim uint) Tensor {
	var dest *C.Tensor
	result := C.wrap_UnSqueeze((*C.Context)(ctx.UnsafePtr()), t.cTensor, &dest, C.dim_t(dim))
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpUnSqueeze, unSqueezeBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = dim
	}
	return out
}

// SafeUnsqeeze nserts a new dimention at dom 0 only if the tensor is 1D
func (t *tensor) SafeUnSqueeze(ctx Context, dims ...uint) Tensor {
	if len(t.Shape()) > 1 {
		return t
	}

	if len(dims) > 1 {
		panic("shapes: only 1 dim is allowed when doing a safe unsqueeze")
	}

	if len(dims) == 1 {
		return t.UnSqueeze(ctx, dims[0])
	}

	return t.UnSqueeze(ctx, uint(0))
}
