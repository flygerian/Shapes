package shapes

// mainContextOption is a function that configures a Context.
type mainContextOption func(*mainContext)

// WithGrad enables or disables gradient tracking and backward passes.
func WithGrad(enabled bool) mainContextOption {
	return func(c *mainContext) {
		c.gradEnabled = enabled
		c.backwardEnabled = enabled
	}
}

func WithArenaSize(size int) mainContextOption {
	return func(c *mainContext) {
		c.arenaSize = size
	}
}

type subContextOption func(*subContext)

func panicIfNoGradients(tensors ...Tensor) {
	for _, it := range tensors {
		if inputTensor, ok := it.(*tensor); !ok || inputTensor.computation == nil || inputTensor.computation.grad == nil {
			panic("Inputs tensors into a fused context have to track gradients")
		}
	}
}

func WithInputs(inputTensors ...Tensor) subContextOption {
	return func(sc *subContext) {
		panicIfNoGradients(inputTensors...)
		sc.inputs = append(sc.inputs, inputTensors...)
	}
}

func WithOpType(opType OpType) subContextOption {
	return func(sc *subContext) {
		sc.op = opType
	}
}

func WithHiddenState(hiddenStateTensors ...Tensor) subContextOption {
	return func(sc *subContext) {
		panicIfNoGradients(hiddenStateTensors...)
		sc.hiddenState = append(sc.hiddenState, hiddenStateTensors...)
	}
}

func WithResult(result Tensor) subContextOption {
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

func WithBackward(backwardFn BackwardFn) subContextOption {
	return func(sc *subContext) {
		sc.backward = backwardFn
	}
}

func WithPersistence() subContextOption {
	return func(sc *subContext) {
		sc.persistant = true
	}
}
