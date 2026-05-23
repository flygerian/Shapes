package layer

import "github.com/flygerian/shapes"

type HasForward interface {
	Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor
}

type Layer interface {
	HasForward
	SetBiasEnabled(isBiasEnabled bool)
}

type Sequential struct {
	Ctx    shapes.Context
	Layers []HasForward
}

func NewSequential(ctx shapes.Context, layers ...HasForward) *Sequential {
	return &Sequential{
		Ctx:    ctx,
		Layers: layers,
	}
}

type layerOption func(layer Layer)

func WithBias(isBiasApplied bool) layerOption {
	return func(layer Layer) {
		layer.SetBiasEnabled(isBiasApplied)
	}
}

func (s *Sequential) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	out := x
	for _, l := range s.Layers {
		out = l.Forward(ctx, out)
	}

	return out
}
