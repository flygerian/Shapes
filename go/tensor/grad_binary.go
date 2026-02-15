package tensor

import shapes "github.com/flygerian/shapes"

// addBackward computes gradients for element-wise addition.
// d(a+b)/da = 1, d(a+b)/db = 1
// grad_a += output_grad, grad_b += output_grad (with broadcast reduction).
func addBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)
	markIfIntermediate(ctx, gradA, node.Grad)

	gradB := reduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
	markIfIntermediate(ctx, gradB, node.Grad)
}

// subtractBackward computes gradients for element-wise subtraction.
// d(a-b)/da = 1, d(a-b)/db = -1
func subtractBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)
	markIfIntermediate(ctx, gradA, node.Grad)

	negGrad := node.Grad.Negate(ctx)
	gradB := reduceBroadcast(ctx, b, negGrad)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
	markIntermediate(ctx, negGrad)
	markIfIntermediate(ctx, gradB, negGrad)
}

// multiplyBackward computes gradients for element-wise multiplication.
// d(a*b)/da = b, d(a*b)/db = a
func multiplyBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad * b
	gradTimesB := node.Grad.Times(ctx, b)
	gradA := reduceBroadcast(ctx, a, gradTimesB)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)
	markIntermediate(ctx, gradTimesB)
	markIfIntermediate(ctx, gradA, gradTimesB)

	// grad_b = output_grad * a
	gradTimesA := node.Grad.Times(ctx, a)
	gradB := reduceBroadcast(ctx, b, gradTimesA)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
	markIntermediate(ctx, gradTimesA)
	markIfIntermediate(ctx, gradB, gradTimesA)
}

// divideBackward computes gradients for element-wise division.
// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
func divideBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad / b
	gradDivB := node.Grad.Divide(ctx, b)
	gradA := reduceBroadcast(ctx, a, gradDivB)
	a.Computation.Grad = a.Computation.Grad.Plus(ctx, gradA)
	markIntermediate(ctx, gradDivB)
	markIfIntermediate(ctx, gradA, gradDivB)

	// grad_b = output_grad * (-a / b^2)
	negA := a.Negate(ctx)
	bSquared := b.Times(ctx, b)
	negADivB2 := negA.Divide(ctx, bSquared)
	gradTimesNeg := node.Grad.Times(ctx, negADivB2)
	gradB := reduceBroadcast(ctx, b, gradTimesNeg)
	b.Computation.Grad = b.Computation.Grad.Plus(ctx, gradB)
	markIntermediate(ctx, negA)
	markIntermediate(ctx, bSquared)
	markIntermediate(ctx, negADivB2)
	markIntermediate(ctx, gradTimesNeg)
	markIfIntermediate(ctx, gradB, gradTimesNeg)
}

// powBackward computes gradients for element-wise power.
// d(x^n)/dx = n * x^(n-1)
func powBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	n := node.Metadata.(float32)

	coeff := Float(ctx, shapeOf(x), n)
	xPow := x.Pow(ctx, n-1)
	localGrad := coeff.Times(ctx, xPow)
	gradX := node.Grad.Times(ctx, localGrad)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
	markIntermediate(ctx, coeff)
	markIntermediate(ctx, xPow)
	markIntermediate(ctx, localGrad)
	markIntermediate(ctx, gradX)
}

// negateBackward computes gradients for element-wise negation.
// d(-t)/dt = -1
func negateBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	t := node.Inputs[0]
	gradT := node.Grad.Negate(ctx)
	t.Computation.Grad = t.Computation.Grad.Plus(ctx, gradT)
	markIntermediate(ctx, gradT)
}

// sumBackward computes gradients for sum reduction along a dimension.
// d(sum(x, dim))/dx = 1 for all elements; grad is broadcast from reduced shape.
func sumBackward(ctx *shapes.Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	inputShape := shapeOf(x)
	ones := Float(ctx, inputShape, 1.0)
	gradX := ones.Times(ctx, node.Grad)
	x.Computation.Grad = x.Computation.Grad.Plus(ctx, gradX)
	markIntermediate(ctx, ones)
	markIntermediate(ctx, gradX)
}

// ReduceBroadcast sums the gradient along dimensions that were broadcast
// to match the input tensor's shape.
func ReduceBroadcast(ctx *shapes.Context, input *Tensor, grad *Tensor) *Tensor {
	return reduceBroadcast(ctx, input, grad)
}

func reduceBroadcast(ctx *shapes.Context, input *Tensor, grad *Tensor) *Tensor {
	inputShape := shapeOf(input)
	gradShape := shapeOf(grad)

	current := grad

	// Handle dimension mismatch: sum along leading dimensions.
	dimDiff := len(gradShape) - len(inputShape)
	if dimDiff < 0 {
		return current
	}

	for range dimDiff {
		prev := current
		summed := current.Sum(ctx, 0)
		current = summed.SqueezeDim(ctx, 0)
		if prev != grad {
			markIntermediate(ctx, prev)
		}
		markIntermediate(ctx, summed)
	}

	// Sum along dimensions where input has size 1 but grad has size > 1.
	currentShape := shapeOf(current)
	for d := range inputShape {
		if inputShape[d] == 1 && currentShape[d] > 1 {
			prev := current
			current = current.Sum(ctx, uint32(d))
			currentShape = shapeOf(current)
			if prev != grad {
				markIntermediate(ctx, prev)
			}
		}
	}

	return current
}
