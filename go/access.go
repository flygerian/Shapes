package shapes

/*
#include "cwrappers.h"
*/
import "C"
import "fmt"

// Get returns a sub-tensor at the given coordinates.
// If the tensor is multidimensional, returns a view of the sub-tensor.
// If coordinates specify all dimensions, returns a 0-dimensional (scalar) tensor.
// Panics if coordinates are invalid.
//
// Get can also accept a tensor as an argument to perform advanced indexing,
// similar to PyTorch's x[indices] where indices is a tensor of integers.
// When two tensors are provided, performs 2D advanced indexing.
type hasAccessOps interface {
	Get(ctx Context, indices ...interface{}) Tensor
	Item() interface{}
}

func (t *tensor) Get(ctx Context, indices ...interface{}) Tensor {
	if len(indices) == 0 {
		panic("shapes: Get requires at least one argument")
	}

	// Check for two tensors (2D advanced indexing)
	if len(indices) == 2 {
		rowTensor, rowOk := indices[0].(*tensor)
		colTensor, colOk := indices[1].(*tensor)
		if rowOk && colOk {
			return t.getWithTensor2d(ctx, rowTensor, colTensor)
		}
	}

	// Check if first argument is a tensor for 1D advanced indexing
	if len(indices) == 1 {
		if idxTensor, ok := indices[0].(*tensor); ok {
			return t.getWithTensor(ctx, idxTensor)
		}
	}

	// Otherwise, treat all arguments as coordinates
	coords := make([]uint, len(indices))
	for i, v := range indices {
		switch val := v.(type) {
		case uint32:
			coords[i] = uint(val)
		case int:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint(val)
		case uint:
			coords[i] = val
		case uint64:
			coords[i] = uint(val)
		case int32:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint(val)
		case int64:
			if val < 0 {
				panic("shapes: negative indices not supported")
			}
			coords[i] = uint(val)
		default:
			panic(fmt.Sprintf("shapes: Get expects uint coordinates or a *Tensor, got %T", v))
		}
	}

	return t.getWithCoords(ctx, coords)
}

// getWithCoords returns a sub-tensor at the given coordinates.
func (t *tensor) getWithCoords(ctx Context, coords []uint) Tensor {
	current := t
	for _, idx := range coords {
		if current.cTensor.shape.numOfDims == 0 {
			panic("shapes: cannot index a 0-dimensional tensor")
		}

		prevInput := current
		var dest *C.Tensor
		cCtx := (*C.Context)(ctx.UnsafePtr())
		res := C.wrap_GetTensorAt(cCtx, current.cTensor, C.dim_t(idx), &dest)
		if res != C.OK {
			panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
		}
		current = track(ctx, &tensor{cTensor: dest})
		if ctx.BackwardEnabled() {
			toComputationGraphNode(current, OpGetTensorAt, getTensorAtBackward, []Tensor{prevInput}, []Tensor{}, nil)
			current.computation.meta = idx
		}
	}
	return current
}

// getWithTensor performs advanced indexing using a tensor of indices.
// The indices tensor must contain integer values.
func (t *tensor) getWithTensor(ctx Context, indices *tensor) Tensor {
	var dest *C.Tensor
	cCtx := (*C.Context)(ctx.UnsafePtr())
	res := C.wrap_IndexWithTensor(cCtx, t.cTensor, indices.cTensor, &dest)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
	}

	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpIndexWithTensor, indexWithTensorBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = indices
	}
	return out
}

// getWithTensor2d performs 2D advanced indexing using two tensors of indices.
// The row and column index tensors must contain integer values and have the same shape.
func (t *tensor) getWithTensor2d(ctx Context, rowIndices, colIndices *tensor) Tensor {
	var dest *C.Tensor
	cCtx := (*C.Context)(ctx.UnsafePtr())
	res := C.wrap_IndexWithTensor2d(cCtx, t.cTensor, rowIndices.cTensor, colIndices.cTensor, &dest)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
	}

	out := track(ctx, &tensor{cTensor: dest})
	if ctx.BackwardEnabled() {
		toComputationGraphNode(out, OpIndexWithTensor2d, indexWithTensor2dBackward, []Tensor{t}, []Tensor{}, nil)
		out.computation.meta = [2]Tensor{rowIndices, colIndices}
	}
	return out
}

// Item extracts the scalar value from a 0-dimensional tensor.
// Returns the value as the appropriate Go type.
// Panics if the tensor is not 0-dimensional.
func (t *tensor) Item() interface{} {
	if t.cTensor.shape.numOfDims != 0 {
		panic("shapes: Item() can only be called on 0-dimensional tensors")
	}

	var v C.Value
	res := C.wrap_GetScalar(t.cTensor, &v)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: %s", resultString(uint32(res))))
	}

	switch t.Dtype() {
	case DtypeBool:
		return bool(C.value_as_bool(v))
	case DtypeF16, DtypeF32, DtypeF64:
		return float32(C.value_as_double(v))
	case DtypeU8:
		return uint8(C.value_as_u64(v))
	case DtypeU16:
		return uint16(C.value_as_u64(v))
	case DtypeU32:
		return uint32(C.value_as_u64(v))
	case DtypeU64:
		return uint64(C.value_as_u64(v))
	case DtypeI8:
		return int8(C.value_as_i64(v))
	case DtypeI16:
		return int16(C.value_as_i64(v))
	case DtypeI32:
		return int32(C.value_as_i64(v))
	case DtypeI64:
		return int64(C.value_as_i64(v))
	default:
		panic("shapes: unknown dtype")
	}
}
