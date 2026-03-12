package layer

import "github.com/flygerian/shapes"

type Layer interface {
	Forward(ctx shapes.Context, x shapes.Tensor) shapes.Tensor
	SetBiasEnabled(isBiasEnabled bool)
}

type layerOption func(layer Layer)

func WithBias(isBiasApplied bool) layerOption {
	return func(layer Layer) {
		layer.SetBiasEnabled(isBiasApplied)
	}
}
