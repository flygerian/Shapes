package shapes

// WrappedTensor wraps a Tensor and its Context to provide a fluent API
// where operations don't require passing the context as the first parameter.
type WrappedTensor struct {
	tensor  *Tensor
	context *Context
}

// Wrap creates a WrappedTensor from a Tensor and Context.
// The tensor and context are captured, allowing fluent chaining of operations.
func (c *Context) Wrap(t *Tensor) *WrappedTensor {
	return &WrappedTensor{
		tensor:  t,
		context: c,
	}
}

// Tensor returns the underlying *Tensor.
func (wt *WrappedTensor) Tensor() *Tensor {
	return wt.tensor
}

// Context returns the underlying *Context.
func (wt *WrappedTensor) Context() *Context {
	return wt.context
}

// validateSameContext panics if the other WrappedTensor doesn't share the same underlying
// C context (i.e. the same arena/memory). Derived contexts (NoGrad, Fused, NoGraph) all
// share the same C context pointer, so they pass this check.
func (wt *WrappedTensor) validateSameContext(other *WrappedTensor) {
	if wt.context.UnsafePtr() != other.context.UnsafePtr() {
		panic("shapes: WrappedTensor operation requires both tensors to belong to the same context")
	}
}
