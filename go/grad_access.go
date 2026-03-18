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

// getTensorAtBackward propagates gradient through a GetTensorAt (row selection) op.
// Forward: out = x[idx]
// Backward: x.grad[idx] += out.grad  (via view + AddInPlace)
func getTensorAtBackward(ctx Context, node ComputationGraphNode) {

	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	idx := node.Metadata().(uint)

	// Get a view into x.Grad at position idx.
	dxView := x.Grad().Get(noGraphCtx, idx)
	dxView.AddInPlace(noGraphCtx, node.Grad().(Tensor))
}

// sliceBackward propagates gradient through a Slice op.
// Forward: out = x[ranges...]
// Backward: x.grad[ranges...] += out.grad  (via view + AddInPlace)
func sliceBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	ranges := node.Metadata().([]Range)
	cRanges := make([]C.Range, len(ranges))
	for i, r := range ranges {
		cRanges[i] = C.Range{start: C.size_t(r[0]), end: C.size_t(r[1])}
	}

	res := C.wrap_SliceAccumulate(
		(*C.Context)(noGraphCtx.UnsafePtr()),
		x.Grad().(*tensor).cTensor,
		(*C.Range)(unsafe.Pointer(&cRanges[0])),
		node.Grad().(*tensor).cTensor,
	)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: SliceAccumulate failed: %s", resultString(uint32(res))))
	}
}

// indexWithTensorBackward propagates gradient through IndexWithTensor (1D advanced gather).
// Forward: out = x[indices]  (gathers rows)
// Backward: x.grad[indices[i], :] += out.grad[i, :]  (scatter-add)
func indexWithTensorBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	indices := node.Metadata().(*tensor)

	cCtx := (*C.Context)(noGraphCtx.UnsafePtr())
	res := C.wrap_IndexAccumulate1d(cCtx, x.Grad().(*tensor).cTensor, indices.cTensor, node.Grad().(*tensor).cTensor)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: IndexAccumulate1d failed: %s", resultString(uint32(res))))
	}

	noGraphCtx.Mark(indices)
}

// indexWithTensor2dBackward propagates gradient through IndexWithTensor2d (2D advanced gather).
// Forward: out = x[rowIndices, colIndices]
// Backward: x.grad[row, col, :] += out.grad[i, :]  (scatter-add)
func indexWithTensor2dBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	idxPair := node.Metadata().([2]Tensor)
	rowIndices := idxPair[0].(*tensor)
	colIndices := idxPair[1].(*tensor)

	cCtx := (*C.Context)(noGraphCtx.UnsafePtr())
	res := C.wrap_IndexAccumulate2d(cCtx, x.Grad().(*tensor).cTensor, rowIndices.cTensor, colIndices.cTensor,
		node.Grad().(*tensor).cTensor)
	if res != C.OK {
		panic(fmt.Sprintf("shapes: IndexAccumulate2d failed: %s", resultString(uint32(res))))
	}

	// Free up these tensors. Context does not mark metadata for deletion
	noGraphCtx.Mark(rowIndices)
	noGraphCtx.Mark(colIndices)
}
