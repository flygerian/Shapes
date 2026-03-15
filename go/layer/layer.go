package layer

import (
	"fmt"

	"github.com/flygerian/shapes"
)

type HasForward interface {
	Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor
}

type Layer interface {
	HasForward
	SetBiasEnabled(isBiasEnabled bool)
}

type Sequential struct {
	Layers []HasForward
}

type layerOption func(layer Layer)

func WithBias(isBiasApplied bool) layerOption {
	return func(layer Layer) {
		layer.SetBiasEnabled(isBiasApplied)
	}
}

func (s *Sequential) Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor {
	out := x.Clone(ctx)

	fmt.Println()
	for _, l := range s.Layers {
		out = l.Forward(ctx, out)
	}

	return out
}
