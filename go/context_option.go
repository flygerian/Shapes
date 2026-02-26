package shapes

// mainContextOption is a function that configures a Context.
type mainContextOption func(*mainContext)

// WithGrad enables or disables gradient tracking and backward passes.
func WithGrad(enabled bool) mainContextOption {
	return func(c *mainContext) {
		c.GradEnabled = enabled
		c.BackwardEnabled = enabled
	}
}

func WithArenaSize(size int) mainContextOption {
	return func(c *mainContext) {
		c.arenaSize = size
	}
}

type subContextOption func(*subContext)

func panicIfNoGradients(tensors ...*Tensor) {
	for _, it := range tensors {
		if it.Computation == nil || it.Computation.grad == nil {
			panic("Inputs tensors into a fused context have to track gradients")
		}
	}
}

func WithInputs(inputTensors ...*Tensor) subContextOption {
	return func(sc *subContext) {
		panicIfNoGradients(inputTensors...)
		sc.inputs = append(sc.inputs, inputTensors...)
	}
}

func WithHiddenState(hiddenStateTensors ...*Tensor) subContextOption {
	return func(sc *subContext) {
		panicIfNoGradients(hiddenStateTensors...)
		sc.hiddenState = append(sc.hiddenState, hiddenStateTensors...)
	}
}

func WithResult(result *Tensor) subContextOption {
	return func(sc *subContext) {
		panicIfNoGradients(result)
		sc.result = result
	}
}

func WithMetadata(metadata any) subContextOption {
	return func(sc *subContext) {
		sc.metadata = metadata
	}
}
