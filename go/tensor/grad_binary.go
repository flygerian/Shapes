package tensor

import shapes "github.com/flygerian/shapes"

// addBackward computes gradients for element-wise addition.
// d(a+b)/da = 1, d(a+b)/db = 1
// grad_a += output_grad, grad_b += output_grad (with broadcast reduction).
func addBackward(node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]
	ctx := noGradCtx(node)

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(gradA)

	gradB := reduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad = b.Computation.Grad.Plus(gradB)
}

// subtractBackward computes gradients for element-wise subtraction.
// d(a-b)/da = 1, d(a-b)/db = -1
func subtractBackward(node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]
	ctx := noGradCtx(node)

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(gradA)

	negOnes := Float(ctx, shapeOf(node.Grad), -1.0)
	negGrad := node.Grad.Times(negOnes)
	gradB := reduceBroadcast(ctx, b, negGrad)
	b.Computation.Grad = b.Computation.Grad.Plus(gradB)
}

// multiplyBackward computes gradients for element-wise multiplication.
// d(a*b)/da = b, d(a*b)/db = a
func multiplyBackward(node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]
	ctx := noGradCtx(node)

	// grad_a = output_grad * b
	gradA := reduceBroadcast(ctx, a, node.Grad.Times(b))
	a.Computation.Grad = a.Computation.Grad.Plus(gradA)

	// grad_b = output_grad * a
	gradB := reduceBroadcast(ctx, b, node.Grad.Times(a))
	b.Computation.Grad = b.Computation.Grad.Plus(gradB)
}

// divideBackward computes gradients for element-wise division.
// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
func divideBackward(node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]
	ctx := noGradCtx(node)

	// grad_a = output_grad / b
	gradA := reduceBroadcast(ctx, a, node.Grad.Divide(b))
	a.Computation.Grad = a.Computation.Grad.Plus(gradA)

	// grad_b = output_grad * (-a / b^2)
	negA := a.Times(Float(ctx, shapeOf(a), -1.0))
	bSquared := b.Times(b)
	gradB := reduceBroadcast(ctx, b, node.Grad.Times(negA.Divide(bSquared)))
	b.Computation.Grad = b.Computation.Grad.Plus(gradB)
}

// reduceBroadcast sums the gradient along dimensions that were broadcast
// to match the input tensor's shape.
func reduceBroadcast(ctx *shapes.Context, input *Tensor, grad *Tensor) *Tensor {
	inputShape := shapeOf(input)
	gradShape := shapeOf(grad)

	current := grad

	// Handle dimension mismatch: sum along leading dimensions.
	dimDiff := len(gradShape) - len(inputShape)
	for i := 0; i < dimDiff; i++ {
		current = current.Sum(0)
	}

	// Sum along dimensions where input has size 1 but grad has size > 1.
	currentShape := shapeOf(current)
	for d := 0; d < len(inputShape); d++ {
		if inputShape[d] == 1 && currentShape[d] > 1 {
			current = current.Sum(uint32(d))
			currentShape = shapeOf(current)
		}
	}

	return current
}
