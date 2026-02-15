package tensor

import shapes "github.com/flygerian/shapes"

// addBackward computes gradients for element-wise addition.
// d(a+b)/da = 1, d(a+b)/db = 1
// grad_a += output_grad, grad_b += output_grad (with broadcast reduction).
func addBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := ReduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)

	gradB := ReduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}

// subtractBackward computes gradients for element-wise subtraction.
// d(a-b)/da = 1, d(a-b)/db = -1
func subtractBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := ReduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)

	negOnes := Float(ctx, shapeOf(node.Grad), -1.0)
	negGrad := node.Grad.Times(ctx, negOnes)
	gradB := ReduceBroadcast(ctx, b, negGrad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}

// multiplyBackward computes gradients for element-wise multiplication.
// d(a*b)/da = b, d(a*b)/db = a
func multiplyBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad * b
	gradA := ReduceBroadcast(ctx, a, node.Grad.Times(ctx, b))
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)

	// grad_b = output_grad * a
	gradB := ReduceBroadcast(ctx, b, node.Grad.Times(ctx, a))
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}

// divideBackward computes gradients for element-wise division.
// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
func divideBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad / b
	gradA := ReduceBroadcast(ctx, a, node.Grad.Divide(ctx, b))
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)

	// grad_b = output_grad * (-a / b^2)
	negA := a.Times(ctx, Float(ctx, shapeOf(a), -1.0))
	bSquared := b.Times(ctx, b)
	gradB := ReduceBroadcast(ctx, b, node.Grad.Times(ctx, negA.Divide(ctx, bSquared)))
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
}

// powBackward computes gradients for element-wise power.
// d(x^n)/dx = n * x^(n-1)
func powBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	n := node.Metadata.(float32)

	coeff := Float(ctx, shapeOf(x), n)
	gradX := node.Grad.Times(ctx, coeff.Times(ctx, x.Pow(ctx, n-1)))
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// negateBackward computes gradients for element-wise negation.
// d(-t)/dt = -1
func negateBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	t := node.Inputs[0]
	negOnes := Float(ctx, shapeOf(node.Grad), -1.0)
	gradT := node.Grad.Times(ctx, negOnes)
	t.Computation.Grad = t.Computation.Grad.Plus(ctx, gradT)
}

// sumBackward computes gradients for sum reduction along a dimension.
// d(sum(x, dim))/dx = 1 for all elements; grad is broadcast from reduced shape.
func sumBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	inputShape := shapeOf(x)
	ones := Float(ctx, inputShape, 1.0)
	gradX := ones.Times(ctx, node.Grad)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
}

// ReduceBroadcast sums the gradient along dimensions that were broadcast
// to match the input tensor's shape.
func ReduceBroadcast(ctx *shapes.Context, input *Tensor, grad *Tensor) *Tensor {
	inputShape := shapeOf(input)
	gradShape := shapeOf(grad)

	current := grad

	// Handle dimension mismatch: sum along leading dimensions.
	dimDiff := len(gradShape) - len(inputShape)
	if dimDiff < 0 {
		return current
	}

	for range dimDiff {
		current = current.Sum(ctx, 0).SqueezeDim(ctx, 0)
	}

	// Sum along dimensions where input has size 1 but grad has size > 1.
	currentShape := shapeOf(current)
	for d := range inputShape {
		if inputShape[d] == 1 && currentShape[d] > 1 {
			current = current.Sum(ctx, uint32(d))
			currentShape = shapeOf(current)
		}
	}

	return current
}
