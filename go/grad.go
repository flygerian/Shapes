package shapes

/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/build/openblas/lib -lshapes_core -lshapes_memory -lopenblas -lm

#include "shapes.h"
#include "common.h"
*/
import "C"
import (
	"fmt"
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
	OpBatchNorm                // batch normalization
	OpReshape                  // reshape
	OpTranspose                // transpose
	OpPermute                  // permute
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
	OpConv
	OpConvTranspose
	OpMaxPool
	OpAdaptiveAvgPool
	OpConcat // concat
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
	case OpBatchNorm:
		return "batch_norm"
	case OpReshape:
		return "reshape"
	case OpTranspose:
		return "transpose"
	case OpPermute:
		return "permute"
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
	case OpConv:
		return "conv"
	case OpConvTranspose:
		return "conv_transpose"
	case OpMaxPool:
		return "max_pool"
	case OpAdaptiveAvgPool:
		return "adaptive_avg_pool"
	case OpConcat:
		return "concat"
	default:
		return "?"
	}
}

type GradTensor interface {
	hasNonMutatingBinaryOps
	hasShapeOps
	hadReductionOps
	hasUnaryOps

	Accumulate(ctx Context, operandB Tensor)
	Get(ctx Context, indices ...interface{}) Tensor
}

type ComputationGraphNode interface {
	Inputs() []Tensor
	Grad() GradTensor
	Metadata() any
	HiddenState() []Tensor
	SetValuesToZero()
	AttachHiddenState(tensors ...Tensor)
	Label() string
	Op() OpType
}

// BackwardFn is the signature for backward pass functions.
type BackwardFn func(ctx Context, node ComputationGraphNode)

type hasBackward interface {
	Backward(ctx Context) ComputationGraph
}

// Computation represents a node in the autograd computation graph.
type Computation struct {
	inputs      []Tensor
	backward    BackwardFn
	op          OpType
	hiddenState []Tensor
	meta        any

	grad Tensor
}

// ComputationGraph holds topologically sorted graph nodes.
type ComputationGraph []ComputationGraphNode

// leafNode attaches an empty GraphNode (nil backward) to a tensor.
// Called at creation time when grad is enabled so that .Grad() is always available.
func leafNode(ctx Context, t *tensor) {
	t.computation = &Computation{
		grad: Zeros(ctx.NoGrad(), t.Shape()),
	}
}

// shapeOf extracts the shape from a tensor's C representation.
func shapeOf(t *tensor) Shape {
	numDims := int(t.cTensor.shape.numOfDims)
	shape := make(Shape, numDims)
	dims := t.cTensor.shape.dims
	for i := range numDims {
		shape[i] = *(*uint)(ptrOffset(dims, i))
	}
	return shape
}

// buildGraph performs a topological sort starting from the output tensor.
func buildGraph(t *tensor) ComputationGraph {
	var graph []ComputationGraphNode
	visited := make(map[ComputationGraphNode]bool)
	topo(&graph, visited, t)
	return graph
}

func topo(graph *[]ComputationGraphNode, visited map[ComputationGraphNode]bool, node ComputationGraphNode) {
	if node == nil || visited[node] {
		return
	}

	visited[node] = true
	for _, inp := range node.Inputs() {
		topo(graph, visited, inp)
	}

	*graph = append(*graph, node)
}

// Backward runs backpropagation from tensor t through the computation graph.
func (t *tensor) Backward(ctx Context) ComputationGraph {
	if t.computation == nil {
		panic("shapes: cannot call Backward on a tensor with no computation graph")
	}

	noGraphCtx := ctx.NoGraph()
	defer noGraphCtx.Finish()
	// Seed the output gradient with ones.
	onesShape := t.Shape()
	t.Grad().Accumulate(noGraphCtx, Float(noGraphCtx, onesShape, 1.0))

	graph := buildGraph(t)

	// Walk in reverse topological order.
	for i := len(graph) - 1; i >= 0; i-- {
		node := graph[i]
		if node.(*tensor).computation.backward != nil {
			node.(*tensor).computation.backward(noGraphCtx, node)
		}
	}

	// Mark all forward-pass graph nodes and their grads for sweeping.
	// Only non-leaf nodes (those with a backward fn) are marked — leaf inputs were created
	// outside the forward pass and may be reused by the caller.
	// Hidden state tensors (parameters) are not in the graph so they are naturally excluded.
	for _, node := range graph {
		n := node.(*tensor)
		if n.computation.backward == nil {
			continue
		}
		ctx.Mark(n)
		if node.Grad() != nil {
			ctx.Mark(node.Grad().(Tensor))
		}
	}

	return graph
}

// Grad returns the gradient tensor. Panics if this tensor has no computation node.
func (t *tensor) Grad() GradTensor {
	if t.computation == nil {
		err := "shapes: tensor has no computation graph node"
		if t.label != "" {
			err = err + fmt.Sprintf(" %s", t.label)
		}
		panic(err)
	}
	return t.computation.grad
}

func (t *tensor) Accumulate(ctx Context, operandB Tensor) {
	t.AddInPlace(ctx, operandB)
}

func (t *tensor) Metadata() any {
	return t.computation.meta
}

func (t *tensor) Inputs() []Tensor {
	return t.computation.inputs
}

// RequiresGrad returns true if this tensor is part of a computation graph.
func (t *tensor) RequiresGrad() bool {
	return t.computation != nil
}

func (t *tensor) HiddenState() []Tensor {
	return t.computation.hiddenState
}

func (t *tensor) AttachHiddenState(tensors ...Tensor) {
	if t.computation == nil {
		panic("shapes: cannot attach hidden state to a tensor with no computation node")
	}
	t.computation.hiddenState = append(t.computation.hiddenState, tensors...)
}
