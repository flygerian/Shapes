package shapes

// WrappedTensor wraps a Tensor and its Context to provide a fluent API
// where operations don't require passing the context as the first parameter.
type WrappedTensor struct {
	tensor  Tensor
	context Context
}
