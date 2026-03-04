package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "cwrappers.h"
*/
import "C"

type hadReductionOps interface {
	Sum(ctx Context, dim uint) Tensor
}

// Sum reduces the tensor along the given dimension by summing, returning a new tensor.
func (t *tensor) Sum(ctx Context, dim uint) Tensor {
	var dest *C.Tensor
	result := C.wrap_Sum((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(dim), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpSum, sumBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = dim
	}
	return out
}

// Mean computes the mean along a single dimension.
// If no dim is provided, it defaults to dimension 0.
func (t *tensor) Mean(ctx Context, dims ...uint) Tensor {
	workingDim := uint(0)

	if len(dims) == 1 {
		workingDim = dims[0]
	} else if len(dims) > 1 {
		panic("shapes: mean expects at most 1 dim arg")
	}

	var dest *C.Tensor
	result := C.wrap_Mean((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(workingDim), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// Max reduces the tensor along a single dimension. If no dims provided, reduces all dims.
func (t *tensor) Max(ctx Context, dims ...uint) Tensor {
	workingDim := uint(0)

	if len(dims) == 1 {
		workingDim = dims[0]
	} else if len(dims) > 1 {
		panic("shapes: max expects 0 or 1 dim args")
	}

	var dest *C.Tensor
	result := C.wrap_Max((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(workingDim), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}
	return track(ctx, &tensor{cTensor: dest})
}

// ArgMax reduces the tensor along a single dimension and returns index positions as I64.
// If no dims are provided, it reduces all dims and returns a scalar index.
func (t *tensor) ArgMax(ctx Context, dims ...uint) Tensor {
	workingDim := uint(0)

	if len(dims) == 1 {
		workingDim = dims[0]
	} else if len(dims) > 1 {
		panic("shapes: argmax expects 0 or 1 dim args")
	}

	var dest *C.Tensor
	result := C.wrap_ArgMax((*C.Context)(ctx.UnsafePtr()), t.cTensor, C.dim_t(workingDim), &dest)
	if result != C.OK {
		panic("shapes: " + resultString(uint32(result)))
	}

	return track(ctx, &tensor{cTensor: dest})
}
