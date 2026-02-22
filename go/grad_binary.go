package shapes

// addBackward computes gradients for element-wise addition.
// d(a+b)/da = 1, d(a+b)/db = 1
// grad_a += output_grad, grad_b += output_grad (with broadcast reduction).
func addBackward(ctx *Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Grad().Plus(ctx, gradA)

	gradB := reduceBroadcast(ctx, b, node.Grad)
	b.Computation.Grad = b.Grad().Plus(ctx, gradB)
}

// subtractBackward computes gradients for element-wise subtraction.
// d(a-b)/da = 1, d(a-b)/db = -1
func subtractBackward(ctx *Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	gradA := reduceBroadcast(ctx, a, node.Grad)
	a.Computation.Grad = a.Grad().Plus(ctx, gradA)

	negGrad := node.Grad.Negate(ctx)
	gradB := reduceBroadcast(ctx, b, negGrad)
	b.Computation.Grad = b.Grad().Plus(ctx, gradB)
}

// multiplyBackward computes gradients for element-wise multiplication.
// d(a*b)/da = b, d(a*b)/db = a
func multiplyBackward(ctx *Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad * b
	gradTimesB := node.Grad.Times(ctx, b)
	gradA := reduceBroadcast(ctx, a, gradTimesB)
	a.Computation.Grad = a.Grad().Plus(ctx, gradA)

	// grad_b = output_grad * a
	gradTimesA := node.Grad.Times(ctx, a)
	gradB := reduceBroadcast(ctx, b, gradTimesA)
	b.Computation.Grad = b.Grad().Plus(ctx, gradB)
}

// divideBackward computes gradients for element-wise division.
// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
func divideBackward(ctx *Context, node *ComputationGraphNode) {
	a := node.Inputs[0]
	b := node.Inputs[1]

	// grad_a = output_grad / b
	gradDivB := node.Grad.Divide(ctx, b)
	gradA := reduceBroadcast(ctx, a, gradDivB)
	a.Computation.Grad = a.Grad().Plus(ctx, gradA)

	// grad_b = output_grad * (-a / b^2)
	negA := a.Negate(ctx)
	bSquared := b.Times(ctx, b)
	negADivB2 := negA.Divide(ctx, bSquared)
	gradTimesNeg := node.Grad.Times(ctx, negADivB2)
	gradB := reduceBroadcast(ctx, b, gradTimesNeg)
	b.Computation.Grad = b.Grad().Plus(ctx, gradB)
}

// powBackward computes gradients for element-wise power.
// d(x^n)/dx = n * x^(n-1)
func powBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	n := node.Metadata.(float32)

	coeff := Float(ctx, shapeOf(x), n)
	xPow := x.Pow(ctx, n-1)
	localGrad := coeff.Times(ctx, xPow)
	gradX := node.Grad.Times(ctx, localGrad)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// negateBackward computes gradients for element-wise negation.
// d(-t)/dt = -1
func negateBackward(ctx *Context, node *ComputationGraphNode) {
	t := node.Inputs[0]
	gradT := node.Grad.Negate(ctx)
	t.Computation.Grad = t.Grad().Plus(ctx, gradT)
}

// sumBackward computes gradients for sum reduction along a dimension.
// d(sum(x, dim))/dx = 1 for all elements; grad is broadcast from reduced shape.
func sumBackward(ctx *Context, node *ComputationGraphNode) {
	x := node.Inputs[0]
	inputShape := shapeOf(x)
	ones := Float(ctx, inputShape, 1.0)
	gradX := ones.Times(ctx, node.Grad)
	x.Computation.Grad = x.Grad().Plus(ctx, gradX)
}

// ReduceBroadcast sums the gradient along dimensions that were broadcast
// to match the input tensor's shape.
func ReduceBroadcast(ctx *Context, input *Tensor, grad *Tensor) *Tensor {
	return reduceBroadcast(ctx, input, grad)
}

func reduceBroadcast(ctx *Context, input *Tensor, grad *Tensor) *Tensor {
	inputShape := shapeOf(input)
	gradShape := shapeOf(grad)

	current := grad

	// Handle dimension mismatch: sum along leading dimensions.
	dimDiff := len(gradShape) - len(inputShape)
	if dimDiff < 0 {
		return current
	}

	for range dimDiff {
		summed := current.Sum(ctx, 0)
		current = summed.SqueezeDim(ctx, 0)
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
