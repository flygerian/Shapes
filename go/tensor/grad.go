package tensor

/*
#cgo CFLAGS: -I../../base
#cgo LDFLAGS: -L../../base/build -L../../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
*/
import "C"
import (
	"unsafe"

	shapes "github.com/flygerian/shapes"
)

// BackwardFn is the signature for backward pass functions.
type BackwardFn func(ctx *shapes.Context, node *ComputationGraphNode)

// OpType identifies the operation that produced a computation graph node.
type OpType int

const (
	OpNone       OpType = iota
	OpAdd               // +
	OpSubtract          // -
	OpMultiply          // *
	OpDivide            // /
	OpPow               // pow
	OpExp               // exp
	OpTanh              // tanh
	OpDense             // wx + b
	OpReshape           // reshape
	OpTranspose         // transpose
	OpSqueeze           // squeeze
	OpSqueezeDim        // squeeze_dim
	OpUnSqueeze         // unsqueeze
	OpNegate            // negate
	OpSum               // sum
)

func (op OpType) String() string {
	switch op {
	case OpAdd:
		return "+"
	case OpSubtract:
		return "-"
	case OpMultiply:
		return "*"
	case OpDivide:
		return "/"
	case OpPow:
		return "pow"
	case OpExp:
		return "exp"
	case OpTanh:
		return "tanh"
	case OpDense:
		return "@w + b"
	case OpReshape:
		return "reshape"
	case OpTranspose:
		return "transpose"
	case OpSqueeze, OpSqueezeDim:
		return "squeeze"
	case OpUnSqueeze:
		return "unsqueeze"
	case OpNegate:
		return "negate"
	case OpSum:
		return "sum"
	default:
		return "?"
	}
}

// ComputationGraphNode represents a node in the autograd computation graph.
type ComputationGraphNode struct {
	Output     *Tensor
	Grad       *Tensor
	Inputs     []*Tensor
	Backward   BackwardFn
	Op         OpType
	Parameters []*Tensor
	Metadata   any
}

// ComputationGraph holds topologically sorted graph nodes.
type ComputationGraph struct {
	Nodes []*ComputationGraphNode
}

// leafNode attaches an empty GraphNode (nil backward) to a tensor.
// Called at creation time when grad is enabled so that .Grad() is always available.
func leafNode(t *Tensor) {
	ctx := t.ctx.NoGrad()
	t.Computation = &ComputationGraphNode{
		Output: t,
		Grad:   Zeros(ctx, shapeOf(t)),
	}
}

// attachNode creates a GraphNode and attaches it to the result tensor.
// Only called when grad is enabled.
func attachNode(result *Tensor, op OpType, backward BackwardFn, inputs ...*Tensor) {
	ctx := result.ctx.NoGrad()
	node := &ComputationGraphNode{
		Output:   result,
		Grad:     Zeros(ctx, shapeOf(result)),
		Inputs:   inputs,
		Backward: backward,
		Op:       op,
	}
	result.Computation = node
}

// markIntermediate registers a tensor for freeing after the backward pass.
func markIntermediate(ctx *shapes.Context, t *Tensor) {
	ctx.MarkIntermediate(unsafe.Pointer(t.cTensor))
}

// markIfIntermediate marks t only if it is not the same tensor as origin.
// ReduceBroadcast may return its input unchanged when no reduction is needed;
// in that case we must not free it.
func markIfIntermediate(ctx *shapes.Context, t *Tensor, origin *Tensor) {
	if t != origin {
		markIntermediate(ctx, t)
	}
}

// FreeIntermediates frees all tensors marked as intermediate during backward.
func FreeIntermediates(ctx *shapes.Context) {
	cCtx := (*C.Context)(ctx.UnsafePtr())
	for _, p := range ctx.Intermediates() {
		ct := (*C.Tensor)(p)
		if ct.isView {
			C.FreeViewTensor(cCtx, ct)
		} else {
			C.FreeTensor(cCtx, ct)
		}
	}
	ctx.ClearIntermediates()
}

// shapeOf extracts the shape from a tensor's C representation.
func shapeOf(t *Tensor) Shape {
	numDims := int(t.cTensor.shape.numOfDims)
	shape := make(Shape, numDims)
	dims := t.cTensor.shape.dims
	for i := 0; i < numDims; i++ {
		shape[i] = *(*uint32)(ptrOffset(dims, i))
	}
	return shape
}

// buildGraph performs a topological sort starting from the output tensor.
func buildGraph(t *Tensor) *ComputationGraph {
	graph := &ComputationGraph{}
	visited := make(map[*ComputationGraphNode]bool)
	topo(graph, visited, t.Computation)
	return graph
}

func topo(graph *ComputationGraph, visited map[*ComputationGraphNode]bool, node *ComputationGraphNode) {
	if node == nil || visited[node] {
		return
	}
	visited[node] = true
	for _, inp := range node.Inputs {
		topo(graph, visited, inp.Computation)
	}
	graph.Nodes = append(graph.Nodes, node)
}

// Backward runs backpropagation from tensor t through the computation graph.
func (t *Tensor) Backward(ctx *shapes.Context) *ComputationGraph {
	if t.Computation == nil {
		panic("shapes: cannot call Backward on a tensor with no computation graph")
	}

	noGraphCtx := ctx.NoGraph()
	// Seed the output gradient with ones.
	onesShape := shapeOf(t)
	t.Computation.Grad = Float(noGraphCtx, onesShape, 1.0)

	graph := buildGraph(t)

	// Walk in reverse topological order.
	for i := len(graph.Nodes) - 1; i >= 0; i-- {
		node := graph.Nodes[i]
		if node.Backward != nil {
			node.Backward(noGraphCtx, node)
		}
	}

	// Free all tensors marked as intermediate during the backprop
	FreeIntermediates(noGraphCtx)

	return graph
}

// Grad returns the gradient tensor. Panics if this tensor has no computation node.
func (t *Tensor) Grad() *Tensor {
	if t.Computation == nil {
		panic("shapes: tensor has no computation graph node")
	}
	return t.Computation.Grad
}

// RequiresGrad returns true if this tensor is part of a computation graph.
func (t *Tensor) RequiresGrad() bool {
	return t.Computation != nil
}
