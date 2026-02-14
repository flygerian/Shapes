package tensor

import shapes "github.com/flygerian/shapes"

// BackwardFn is the signature for backward pass functions.
type BackwardFn func(node *ComputationGraphNode)

// OpType identifies the operation that produced a computation graph node.
type OpType int

const (
	OpNone     OpType = iota
	OpAdd             // +
	OpSubtract        // -
	OpMultiply        // *
	OpDivide          // /
	OpPow             // pow
	OpExp             // exp
	OpTanh            // tanh
	OpDense           // wx + b
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
	default:
		return "?"
	}
}

// ComputationGraphNode represents a node in the autograd computation graph.
type ComputationGraphNode struct {
	Output   *Tensor
	Grad     *Tensor
	Inputs   []*Tensor
	Backward BackwardFn
	Op       OpType
	Metadata any
}

// computationGraph holds topologically sorted graph nodes.
type computationGraph struct {
	nodes []*ComputationGraphNode
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
func buildGraph(t *Tensor) *computationGraph {
	graph := &computationGraph{}
	visited := make(map[*ComputationGraphNode]bool)
	topo(graph, visited, t.Computation)
	return graph
}

func topo(graph *computationGraph, visited map[*ComputationGraphNode]bool, node *ComputationGraphNode) {
	if node == nil || visited[node] {
		return
	}
	visited[node] = true
	for _, inp := range node.Inputs {
		topo(graph, visited, inp.Computation)
	}
	graph.nodes = append(graph.nodes, node)
}

// Backward runs backpropagation from tensor t through the computation graph.
func (t *Tensor) Backward() {
	if t.Computation == nil {
		panic("shapes: cannot call Backward on a tensor with no computation graph")
	}

	// Seed the output gradient with ones.
	ctx := t.ctx.NoGrad()
	onesShape := shapeOf(t)
	t.Computation.Grad = Float(ctx, onesShape, 1.0)

	graph := buildGraph(t)

	// Walk in reverse topological order.
	for i := len(graph.nodes) - 1; i >= 0; i-- {
		node := graph.nodes[i]
		if node.Backward != nil {
			node.Backward(node)
		}
	}
}

// RequiresGrad returns true if this tensor is part of a computation graph.
func (t *Tensor) RequiresGrad() bool {
	return t.Computation != nil
}

// noGradCtx returns a no-grad context from the node's first input.
// Used inside backward functions.
func noGradCtx(node *ComputationGraphNode) *shapes.Context {
	return node.Inputs[0].ctx.NoGrad()
}
