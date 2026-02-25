package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "tensor/tensor.h"
#include "common.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// OpType identifies the operation that produced a computation graph node.
type OpType int

const (
	OpNone              OpType = iota
	OpAdd                      // +
	OpSubtract                 // -
	OpMultiply                 // *
	OpDivide                   // /
	OpPow                      // pow
	OpExp                      // exp
	OpTanh                     // tanh
	OpDense                    // wx + b
	OpReshape                  // reshape
	OpTranspose                // transpose
	OpSqueeze                  // squeeze
	OpSqueezeDim               // squeeze_dim
	OpUnSqueeze                // unsqueeze
	OpNegate                   // negate
	OpSum                      // sum
	OpMse                      // mse loss
	OpCrossEntropy             // cross-entropy loss
	OpOneHot                   // one-hot encoding
	OpGetTensorAt              // view row-selection (Get with scalar index)
	OpSlice                    // view slice
	OpIndexWithTensor          // advanced 1D gather
	OpIndexWithTensor2d        // advanced 2D gather
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
	case OpMse:
		return "mse"
	case OpCrossEntropy:
		return "cross_entropy"
	case OpOneHot:
		return "one_hot"
	case OpGetTensorAt:
		return "get_tensor_at"
	case OpSlice:
		return "slice"
	case OpIndexWithTensor:
		return "index_with_tensor"
	case OpIndexWithTensor2d:
		return "index_with_tensor2d"
	default:
		return "?"
	}
}

type GradTensor interface {
	HasNonMutatingBinaryOps
	HasShapeOps
	Accumulate(ctx *Context, operaandB *Tensor)
}

type ComputationGraphNode interface {
	Inputs() []*Tensor
	Grad() GradTensor
	Metadata() []*Tensor
}

// BackwardFn is the signature for backward pass functions.
type BackwardFn func(ctx *Context, node ComputationGraphNode)

// Computation represents a node in the autograd computation graph.
type Computation struct {
	Inputs     []*Tensor
	Backward   BackwardFn
	Op         OpType
	Parameters []*Tensor
	Metadata   []*Tensor

	grad *Tensor
}

// ComputationGraph holds topologically sorted graph nodes.
type ComputationGraph struct {
	Nodes []*Computation
}

// leafNode attaches an empty GraphNode (nil backward) to a tensor.
// Called at creation time when grad is enabled so that .Grad() is always available.
func leafNode(ctx *Context, t *Tensor) {
	noGradCtx := ctx.NoGrad()
	t.Computation = &Computation{
		grad: Zeros(noGradCtx, shapeOf(t)),
	}
}

// NewComputationGraphNode attaches a computation graph node to result.
// Used by external packages (e.g., layer, activation) to register custom backward passes.
func (c *Context) NewComputationGraphNode(result *Tensor, op OpType, backward BackwardFn, inputs []*Tensor, parameters []*Tensor, metadata []*Tensor) {
	c.newNode(result, op, backward, nil, inputs, parameters)
}

// newNode creates a GraphNode and attaches it to the result tensor.
// Internal helper used by both NewComputationGraphNode and NewComputationGraphNodeSaved.
// When called on a fused context, sweeps locals to mark forward intermediates.
func (c *Context) newNode(result *Tensor, op OpType, backward BackwardFn, saved []*Tensor, inputs []*Tensor, parameters []*Tensor) {
	node := &Computation{
		grad:       Zeros(c.NoGrad(), shapeOf(result)),
		Inputs:     inputs,
		Backward:   backward,
		Op:         op,
		Parameters: parameters,
	}
	result.Computation = node
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
	visited := make(map[*Computation]bool)
	topo(graph, visited, t.Computation)
	return graph
}

func topo(graph *ComputationGraph, visited map[*Computation]bool, node *Computation) {
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
func (t *Tensor) Backward(ctx *Context) *ComputationGraph {
	if t.Computation == nil {
		panic("shapes: cannot call Backward on a tensor with no computation graph")
	}

	noGraphCtx := ctx.NoGraph()
	// Seed the output gradient with ones.
	onesShape := shapeOf(t)
	t.Grad().Accumulate(noGraphCtx, Float(noGraphCtx, onesShape, 1.0))

	graph := buildGraph(t)

	// Walk in reverse topological order.
	for i := len(graph.Nodes) - 1; i >= 0; i-- {
		node := graph.Nodes[i]
		if node.Backward != nil {
			// Create fresh op subcontext per node for sweeping
			opCtx := noGraphCtx.OpSubcontext()
			node.Backward(opCtx, t)
		}
	}

	// Free all tensors marked as intermediate during the backprop
	FreeIntermediates(noGraphCtx)

	return graph
}

// Grad returns the gradient tensor. Panics if this tensor has no computation node.
func (t *Tensor) Grad() GradTensor {
	if t.Computation == nil {
		err := "shapes: tensor has no computation graph node"
		if t.Label != "" {
			err = err + fmt.Sprintf(" %s", t.Label)
		}
		panic(err)
	}
	return t.Computation.grad
}

func (t *Tensor) Accumulate(ctx *Context, operaandB *Tensor) {
	t.AddInPlace(ctx, operaandB)
}

func (t *Tensor) Metadata() []*Tensor {
	return t.Computation.Metadata
}

func (t *Tensor) Inputs() []*Tensor {
	return t.Computation.Inputs
}

// RequiresGrad returns true if this tensor is part of a computation graph.
func (t *Tensor) RequiresGrad() bool {
	return t.Computation != nil
}

// Backward runs backpropagation from this WrappedTensor through the computation graph.
func (wt *WrappedTensor) Backward() *ComputationGraph {
	return wt.tensor.Backward(wt.context)
}

// Grad returns the gradient WrappedTensor. Panics if this tensor has no computation node.
func (wt *WrappedTensor) Grad() GradTensor {
	return wt.tensor.Grad()
}

// RequiresGrad returns true if this WrappedTensor is part of a computation graph.
func (wt *WrappedTensor) RequiresGrad() bool {
	return wt.tensor.RequiresGrad()
}
