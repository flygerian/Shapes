package activation

import (
	shapes "github.com/flygerian/shapes"
)

type reluActivation struct{}

func (r *reluActivation) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	forwardCtx := ctx.Forward(shapes.WithInputs(x))
	out := shapes.Relu(ctx, x)

	forwardCtx.Finish(shapes.WithResult(out), shapes.WithBackward(shapes.ReluBackward))
	return out
}

type tanhActivation struct{}

func (t *tanhActivation) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	return shapes.Tanh(ctx, x)
}

// Relu returns a stateless activation layer that can be used inside Sequential.
func Relu() *reluActivation {
	return &reluActivation{}
}

// Tanh returns a stateless activation layer that can be used inside Sequential.
func Tanh() *tanhActivation {
	return &tanhActivation{}
}
