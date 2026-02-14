package tensor

import shapes "github.com/flygerian/shapes"

// BackwardFn is the signature for backward pass functions.
type BackwardFn func(node *GraphNode)

// GraphNode represents a node in the autograd computation graph.
type GraphNode struct {
	output   *Tensor
	grad     *Tensor
	inputs   []*Tensor
	backward BackwardFn
	metadata any
}

// computationGraph holds topologically sorted graph nodes.
type computationGraph struct {
	nodes []*GraphNode
}

// leafNode attaches an empty GraphNode (nil backward) to a tensor.
// Called at creation time when grad is enabled so that .Grad() is always available.
func leafNode(t *Tensor) {
	ctx := t.ctx.NoGrad()
	t.node = &GraphNode{
		output: t,
		grad:   Zeros(ctx, shapeOf(t)),
	}
}

// attachNode creates a GraphNode and attaches it to the result tensor.
// Only called when grad is enabled.
func attachNode(result *Tensor, backward BackwardFn, inputs ...*Tensor) {
	ctx := result.ctx.NoGrad()
	node := &GraphNode{
		output:   result,
		grad:     Zeros(ctx, shapeOf(result)),
		inputs:   inputs,
		backward: backward,
	}
	result.node = node
}

// shapeOf extracts the shape from a tensor's C representation.
func shapeOf(t *Tensor) Shape {
	numDims := int(t.cTensor.shape.numOfDims)
	shape := make(Shape, numDims)
	dims := t.cTensor.shape.dims
	for i := range numDims {
		shape[i] = *(*uint32)(ptrOffset(dims, i))
	}
	return shape
}

// buildGraph performs a topological sort starting from the output tensor.
func buildGraph(t *Tensor) *computationGraph {
	graph := &computationGraph{}
	visited := make(map[*GraphNode]bool)
	topo(graph, visited, t.node)
	return graph
}

func topo(graph *computationGraph, visited map[*GraphNode]bool, node *GraphNode) {
	if node == nil || visited[node] {
		return
	}
	visited[node] = true
	for _, inp := range node.inputs {
		topo(graph, visited, inp.node)
	}
	graph.nodes = append(graph.nodes, node)
}

// Backward runs backpropagation from tensor t through the computation graph.
func Backward(t *Tensor) {
	if t.node == nil {
		panic("shapes: cannot call Backward on a tensor with no computation graph")
	}

	// Seed the output gradient with ones.
	ctx := t.ctx.NoGrad()
	onesShape := shapeOf(t)
	t.node.grad = Float(ctx, onesShape, 1.0)

	graph := buildGraph(t)

	// Walk in reverse topological order.
	for i := len(graph.nodes) - 1; i >= 0; i-- {
		node := graph.nodes[i]
		if node.backward != nil {
			node.backward(node)
		}
	}
}

// Grad returns the accumulated gradient for this tensor.
// Panics if the tensor has no graph node.
func (t *Tensor) Grad() *Tensor {
	if t.node == nil {
		panic("shapes: tensor has no gradient (not part of a computation graph)")
	}
	return t.node.grad
}

// RequiresGrad returns true if this tensor is part of a computation graph.
func (t *Tensor) RequiresGrad() bool {
	return t.node != nil
}

// noGradCtx returns a no-grad context from the node's first input.
// Used inside backward functions.
func noGradCtx(node *GraphNode) *shapes.Context {
	return node.inputs[0].ctx.NoGrad()
}
