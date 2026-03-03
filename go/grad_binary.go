package shapes

// addBackward computes gradients for element-wise addition.
// d(a+b)/da = 1, d(a+b)/db = 1
// grad_a += output_grad, grad_b += output_grad (with broadcast reduction).
func addBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	a := node.Inputs()[0]
	b := node.Inputs()[1]

	gradA := reduceBroadcast(noGraphCtx, a.(*tensor), node.Grad())
	a.Grad().Accumulate(noGraphCtx, gradA)

	gradB := reduceBroadcast(noGraphCtx, b.(*tensor), node.Grad())
	b.Grad().Accumulate(noGraphCtx, gradB)
}

// subtractBackward computes gradients for element-wise subtraction.
// d(a-b)/da = 1, d(a-b)/db = -1
func subtractBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	a := node.Inputs()[0]
	b := node.Inputs()[1]

	gradA := reduceBroadcast(noGraphCtx, a.(*tensor), node.Grad())
	a.Grad().Accumulate(noGraphCtx, gradA)

	negGrad := node.Grad().Negate(noGraphCtx)
	gradB := reduceBroadcast(noGraphCtx, b.(*tensor), negGrad.(GradTensor))
	b.Grad().Accumulate(noGraphCtx, gradB)
}

// multiplyBackward computes gradients for element-wise multiplication.
// d(a*b)/da = b, d(a*b)/db = a
func multiplyBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	a := node.Inputs()[0]
	b := node.Inputs()[1]

	// grad_a = output_grad * b
	gradTimesB := node.Grad().Times(noGraphCtx, b)
	gradA := reduceBroadcast(noGraphCtx, a.(*tensor), gradTimesB.(GradTensor))
	a.Grad().Accumulate(noGraphCtx, gradA)

	// grad_b = output_grad * a
	gradTimesA := node.Grad().Times(noGraphCtx, a)
	gradB := reduceBroadcast(noGraphCtx, b.(*tensor), gradTimesA.(GradTensor))
	b.Grad().Accumulate(noGraphCtx, gradB)
}

// divideBackward computes gradients for element-wise division.
// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
func divideBackward(ctx Context, node ComputationGraphNode) {

	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	a := node.Inputs()[0]
	b := node.Inputs()[1]

	// grad_a = output_grad / b
	gradDivB := node.Grad().Divide(noGraphCtx, b)
	gradA := reduceBroadcast(noGraphCtx, a.(*tensor), gradDivB.(GradTensor))
	a.Grad().Accumulate(noGraphCtx, gradA)

	// grad_b = output_grad * (-a / b^2)
	negA := a.Negate(noGraphCtx)
	bSquared := b.Times(noGraphCtx, b)
	negADivB2 := negA.Divide(noGraphCtx, bSquared)
	gradTimesNeg := node.Grad().Times(noGraphCtx, negADivB2)
	gradB := reduceBroadcast(noGraphCtx, b.(*tensor), gradTimesNeg.(GradTensor))
	b.Grad().Accumulate(noGraphCtx, gradB)
}

// powBackward computes gradients for element-wise power.
// d(x^n)/dx = n * x^(n-1)
func powBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	n := node.Metadata().(float32)

	coeff := Float(ctx, x.Shape(), n)
	xPow := x.Pow(noGraphCtx, n-1)
	localGrad := coeff.Times(noGraphCtx, xPow)
	gradX := node.Grad().Times(noGraphCtx, localGrad)
	x.Grad().Accumulate(noGraphCtx, gradX)
}

// negateBackward computes gradients for element-wise negation.
// d(-t)/dt = -1
func negateBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	t := node.Inputs()[0]
	gradT := node.Grad().Negate(noGraphCtx)
	t.Grad().Accumulate(noGraphCtx, gradT)
}

// sumBackward computes gradients for sum reduction along a dimension.
// d(sum(x, dim))/dx = 1 for all elements; grad is broadcast from reduced shape.
func sumBackward(ctx Context, node ComputationGraphNode) {
	noGraphCtx := ctx.Backward()
	defer noGraphCtx.Finish()

	x := node.Inputs()[0]
	inputShape := x.Shape()
	ones := Float(noGraphCtx, inputShape, 1.0)
	gradX := ones.Times(noGraphCtx, node.Grad().(Tensor))
	x.Grad().Accumulate(noGraphCtx, gradX)
}

// ReduceBroadcast sums the gradient along dimensions that were broadcast
// to match the input tensor's shape.
func ReduceBroadcast(ctx Context, input Tensor, grad GradTensor) Tensor {
	return reduceBroadcast(ctx, input.(*tensor), grad)
}

func reduceBroadcast(ctx Context, input *tensor, grad GradTensor) Tensor {
	inputShape := input.Shape()
	gradShape := grad.Shape()

	current := grad.(Tensor)

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
	currentShape := current.Shape()
	for d := range inputShape {
		if inputShape[d] == 1 && currentShape[d] > 1 {
			current = current.Sum(ctx, uint(d))
			currentShape = current.Shape()
		}
	}

	return current
}
